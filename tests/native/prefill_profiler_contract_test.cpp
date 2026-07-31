#include "tatara/runtime/prefill_profiler.h"
#include "tatara/runtime/prefill_step.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <type_traits>

namespace {

using tatara::backend::metal::CounterSampleError;
using tatara::backend::metal::CounterSamplingMode;
using tatara::backend::metal::CounterStageSampleError;
using tatara::backend::metal::MetalComputePass;
using tatara::backend::metal::MetalCounterSampleBuffer;
using tatara::runtime::encode_prefill;
using tatara::runtime::encode_prefill_unit;
using tatara::runtime::kInvalidPrefillProfileEventIndex;
using tatara::runtime::kNoPrefillProfileLayerIndex;
using tatara::runtime::kPrefillProfileDispatchTaxonomy;
using tatara::runtime::PrefillProfileDispatchTicket;
using tatara::runtime::PrefillProfileEvent;
using tatara::runtime::PrefillProfileEventClass;
using tatara::runtime::PrefillProfileSampleWindow;
using tatara::runtime::PrefillProfiler;
using tatara::runtime::PrefillProfilerError;
using tatara::runtime::PrefillProfilerState;
using tatara::runtime::ProfiledPrefillEncodeResult;
using tatara::runtime::ProfiledPrefillProgressResult;

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

constexpr PrefillProfileEvent kGuard{
    .event_class = PrefillProfileEventClass::MoeResidualOutput,
    .layer_index = 0x123456789abcdef0ULL,
    .chunk_ordinal = 0x13579bdfU,
    .chunk_offset = 0x2468ace0U,
    .chunk_rows = 0x11223344U,
    .operation_row_begin = 0x55667788U,
    .operation_row_count = 0x99aabbccU,
};

constexpr PrefillProfileEvent kEmbedding{
    .event_class = PrefillProfileEventClass::Embedding,
    .layer_index = kNoPrefillProfileLayerIndex,
    .chunk_ordinal = 0,
    .chunk_offset = 0,
    .chunk_rows = 8,
    .operation_row_begin = 0,
    .operation_row_count = 8,
};

constexpr PrefillProfileEvent kNormalization{
    .event_class = PrefillProfileEventClass::LayerInputNormalization,
    .layer_index = 0,
    .chunk_ordinal = 0,
    .chunk_offset = 0,
    .chunk_rows = 8,
    .operation_row_begin = 0,
    .operation_row_count = 8,
};

consteval bool taxonomy_is_exact() {
    if (kPrefillProfileDispatchTaxonomy.size() != 35) {
        return false;
    }
    for (std::size_t index = 0; index < kPrefillProfileDispatchTaxonomy.size(); ++index) {
        if (static_cast<std::size_t>(kPrefillProfileDispatchTaxonomy[index]) != index ||
            !tatara::runtime::is_prefill_profile_dispatch_class(
                kPrefillProfileDispatchTaxonomy[index])) {
            return false;
        }
    }
    return !tatara::runtime::is_prefill_profile_dispatch_class(
        static_cast<PrefillProfileEventClass>(kPrefillProfileDispatchTaxonomy.size()));
}

void check(bool condition, const char* message) noexcept {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void test_default_off_and_capacity() {
    PrefillProfiler disabled;
    const auto disabled_status = disabled.status();
    check(disabled_status.state == PrefillProfilerState::Disabled &&
              disabled_status.error == PrefillProfilerError::None &&
              disabled_status.sampling_mode == CounterSamplingMode::DispatchBoundary &&
              disabled_status.event_count == 0 && disabled_status.required_sample_count == 0,
          "the default recorder is inert and error-free");
    check(disabled.finalize().error == PrefillProfilerError::Disabled,
          "finalizing the default-off recorder returns a typed exit");

    constexpr std::array<PrefillProfileEvent, 2> kEvents{
        kEmbedding,
        kNormalization,
    };
    PrefillProfiler short_capacity(kEvents, 3);
    const auto short_status = short_capacity.status();
    check(short_status.state == PrefillProfilerState::Failed &&
              short_status.error == PrefillProfilerError::SampleCapacityInsufficient &&
              short_status.required_sample_count == 4 && short_status.sample_capacity == 3,
          "two counter samples per event are required before recording");

    PrefillProfiler mismatched_capacity(kEvents, 4);
    check(mismatched_capacity.validate_sample_capacity(5) ==
                  PrefillProfilerError::SampleCapacityMismatch &&
              mismatched_capacity.status().state == PrefillProfilerState::Failed,
          "a recorder cannot be rebound to a different sample capacity");

    PrefillProfiler empty({}, 0);
    check(empty.status().error == PrefillProfilerError::EmptyEventPlan,
          "an empty enabled profile is rejected");
}

void test_stage_windows_and_mode_binding() {
    constexpr std::array<PrefillProfileEvent, 2> kEvents{
        kEmbedding,
        kNormalization,
    };
    PrefillProfiler stage(kEvents, 2,
                          CounterSamplingMode::StageBoundaryEncoderSplit);
    check(stage.status() &&
              stage.status().sampling_mode ==
                  CounterSamplingMode::StageBoundaryEncoderSplit &&
              stage.status().required_sample_count == 4 &&
              stage.status().sample_capacity == 2,
          "stage mode accepts a fixed native window smaller than the persistent request");
    check(stage.validate_sampling_mode(CounterSamplingMode::DispatchBoundary) ==
              PrefillProfilerError::SamplingModeMismatch,
          "a dispatch sample buffer cannot be rebound to a stage recorder");

    PrefillProfiler recorder(kEvents, 2,
                             CounterSamplingMode::StageBoundaryEncoderSplit);
    PrefillProfileDispatchTicket ticket;
    check(recorder.begin_dispatch(kEmbedding, ticket) ==
              PrefillProfilerError::SampleWindowRequired,
          "stage recording requires an explicit bounded sample window");

    PrefillProfiler windows(kEvents, 2,
                            CounterSamplingMode::StageBoundaryEncoderSplit);
    PrefillProfileSampleWindow window;
    check(windows.begin_sample_window() == PrefillProfilerError::None &&
              windows.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              ticket.event_index == 0 && ticket.samples.start == 0 &&
              ticket.samples.end == 1 &&
              windows.complete_dispatch(ticket) == PrefillProfilerError::None &&
              windows.finish_sample_window(window) == PrefillProfilerError::None &&
              window.event_begin == 0 && window.event_count == 1 &&
              window.sample_count == 2,
          "the first stage window maps local pair zero to the persistent first event");
    check(windows.begin_sample_window() == PrefillProfilerError::None &&
              windows.begin_dispatch(kNormalization, ticket) ==
                  PrefillProfilerError::None &&
              ticket.event_index == 1 && ticket.samples.start == 0 &&
              ticket.samples.end == 1 &&
              windows.complete_dispatch(ticket) == PrefillProfilerError::None &&
              windows.finish_sample_window(window) == PrefillProfilerError::None &&
              window.event_begin == 1 && window.event_count == 1 &&
              window.sample_count == 2 && windows.finalize(),
          "a later window reuses local samples while preserving the global event cursor");

    PrefillProfiler bounded(kEvents, 2,
                            CounterSamplingMode::StageBoundaryEncoderSplit);
    tatara::backend::metal::CounterSamplePair pair;
    check(bounded.begin_sample_window() == PrefillProfilerError::None &&
              bounded.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              bounded.complete_dispatch(ticket) == PrefillProfilerError::None &&
              bounded.next_stage_sample_pair(pair) ==
                  PrefillProfilerError::SampleWindowCapacityExceeded,
          "a bounded stage window fails before indexing beyond native sample capacity");

    PrefillProfiler stage_failure(kEvents, 2,
                                  CounterSamplingMode::StageBoundaryEncoderSplit);
    check(stage_failure.fail_stage(CounterStageSampleError::EncoderCreationFailed) ==
                  PrefillProfilerError::StageEncoderSplitFailed &&
              stage_failure.status().stage_error ==
                  CounterStageSampleError::EncoderCreationFailed,
          "encoder-split failures retain their typed stage cause");
}

void test_stage_public_state_precedence() {
    constexpr std::array<PrefillProfileEvent, 1> kEvents{kEmbedding};
    PrefillProfileSampleWindow window{
        .event_begin = 99,
        .event_count = 99,
        .sample_count = 99,
    };
    tatara::backend::metal::CounterSamplePair pair{99, 99};

    PrefillProfiler disabled_finish;
    check(disabled_finish.finish_sample_window(window) ==
                  PrefillProfilerError::Disabled &&
              disabled_finish.status().state == PrefillProfilerState::Failed &&
              window.event_begin == kInvalidPrefillProfileEventIndex &&
              window.event_count == 0 && window.sample_count == 0,
          "disabled state precedes window-state validation when finishing");
    PrefillProfiler disabled_next;
    check(disabled_next.next_stage_sample_pair(pair) ==
                  PrefillProfilerError::Disabled &&
              disabled_next.status().state == PrefillProfilerState::Failed &&
              pair.start == 0 && pair.end == 0,
          "disabled state precedes mode validation when requesting a pair");

    PrefillProfiler dispatch_finish(kEvents, 2);
    check(dispatch_finish.finish_sample_window(window) ==
                  PrefillProfilerError::InvalidSamplingMode,
          "dispatch mode precedes inactive-window validation when finishing");
    PrefillProfiler dispatch_next(kEvents, 2);
    check(dispatch_next.next_stage_sample_pair(pair) ==
                  PrefillProfilerError::InvalidSamplingMode,
          "dispatch mode is rejected before stage window-state validation");

    PrefillProfiler finalized_finish(
        kEvents, 2, CounterSamplingMode::StageBoundaryEncoderSplit);
    PrefillProfileDispatchTicket ticket;
    check(finalized_finish.begin_sample_window() == PrefillProfilerError::None &&
              finalized_finish.begin_dispatch(kEmbedding, ticket) ==
                  PrefillProfilerError::None &&
              finalized_finish.complete_dispatch(ticket) ==
                  PrefillProfilerError::None &&
              finalized_finish.finish_sample_window(window) ==
                  PrefillProfilerError::None &&
              finalized_finish.finalize(),
          "finalized-stage finish fixture reaches its terminal state");
    check(finalized_finish.finish_sample_window(window) ==
                  PrefillProfilerError::AlreadyFinalized,
          "finalized state precedes inactive-window validation when finishing");

    PrefillProfiler finalized_next(
        kEvents, 2, CounterSamplingMode::StageBoundaryEncoderSplit);
    check(finalized_next.begin_sample_window() == PrefillProfilerError::None &&
              finalized_next.begin_dispatch(kEmbedding, ticket) ==
                  PrefillProfilerError::None &&
              finalized_next.complete_dispatch(ticket) ==
                  PrefillProfilerError::None &&
              finalized_next.finish_sample_window(window) ==
                  PrefillProfilerError::None &&
              finalized_next.finalize(),
          "finalized-stage pair fixture reaches its terminal state");
    check(finalized_next.next_stage_sample_pair(pair) ==
                  PrefillProfilerError::AlreadyFinalized,
          "finalized state precedes inactive-window validation for sample pairs");
}

void test_cursor_order_and_canaries() {
    std::array<PrefillProfileEvent, 4> guarded{
        kGuard,
        kEmbedding,
        kNormalization,
        kGuard,
    };
    PrefillProfiler profiler(std::span<const PrefillProfileEvent>{guarded}.subspan(1, 2), 4);
    check(profiler.validate_sample_capacity(4) == PrefillProfilerError::None,
          "the configured sample buffer is accepted");

    PrefillProfileDispatchTicket ticket;
    check(profiler.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              ticket.event_index == 0 && ticket.samples.start == 0 && ticket.samples.end == 1 &&
              profiler.status().state == PrefillProfilerState::DispatchPending &&
              profiler.status().event_cursor == 0,
          "the first event reserves the exact first sample pair");
    check(profiler.complete_dispatch(ticket) == PrefillProfilerError::None &&
              profiler.status().event_cursor == 1 &&
              profiler.status().state == PrefillProfilerState::Ready,
          "completion advances exactly one event");

    check(profiler.begin_dispatch(kNormalization, ticket) == PrefillProfilerError::None &&
              ticket.event_index == 1 && ticket.samples.start == 2 && ticket.samples.end == 3,
          "the second event receives the adjacent second sample pair");
    check(profiler.complete_dispatch(ticket) == PrefillProfilerError::None,
          "the second event completes");
    const auto finalized = profiler.finalize();
    check(finalized && finalized.state == PrefillProfilerState::Finalized &&
              finalized.event_cursor == 2 && guarded.front() == kGuard && guarded.back() == kGuard,
          "finalization preserves caller canaries and the complete cursor");
    check(profiler.finalize().error == PrefillProfilerError::AlreadyFinalized,
          "a second finalization has a typed exit");
}

void test_mismatch_pending_and_counter_exits() {
    constexpr std::array<PrefillProfileEvent, 2> kEvents{
        kEmbedding,
        kNormalization,
    };

    PrefillProfiler mismatch(kEvents, 4);
    PrefillProfileEvent actual = kEmbedding;
    actual.chunk_rows = 7;
    PrefillProfileDispatchTicket ticket;
    check(mismatch.begin_dispatch(actual, ticket) == PrefillProfilerError::EventMismatch,
          "full event identity, not only class, is validated");
    const auto mismatch_status = mismatch.status();
    check(mismatch_status.state == PrefillProfilerState::Failed &&
              mismatch_status.mismatch_index == 0 && mismatch_status.expected_event == kEmbedding &&
              mismatch_status.actual_event == actual,
          "the mismatch retains exact expected and actual identities");

    PrefillProfiler pending(kEvents, 4);
    check(pending.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              pending.finalize().error == PrefillProfilerError::Incomplete,
          "a pending dispatch cannot be finalized");
    PrefillProfileDispatchTicket forged = ticket;
    forged.samples.end = 2;
    check(pending.complete_dispatch(forged) == PrefillProfilerError::DispatchTicketMismatch,
          "a forged sample cursor fails closed");

    PrefillProfiler counter_failure(kEvents, 4);
    check(counter_failure.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              counter_failure.fail_counter(CounterSampleError::InvalidBuffer) ==
                  PrefillProfilerError::CounterSamplingFailed,
          "counter failures receive a distinct profiler exit");
    const auto counter_status = counter_failure.status();
    check(counter_status.state == PrefillProfilerState::Failed &&
              counter_status.counter_error == CounterSampleError::InvalidBuffer &&
              counter_status.event_cursor == 0,
          "counter failure preserves its typed cause and cursor");

    PrefillProfiler incomplete(kEvents, 4);
    const auto early = incomplete.finalize();
    check(early.error == PrefillProfilerError::Incomplete &&
              incomplete.status().state == PrefillProfilerState::Ready,
          "early finalization does not consume a persistent bounded recorder");
    check(incomplete.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              incomplete.complete_dispatch(ticket) == PrefillProfilerError::None &&
              incomplete.finalize().error == PrefillProfilerError::Incomplete &&
              incomplete.begin_dispatch(kNormalization, ticket) == PrefillProfilerError::None &&
              incomplete.complete_dispatch(ticket) == PrefillProfilerError::None &&
              incomplete.finalize(),
          "one recorder persists across separately completed bounded units");
}

void test_sequence_exhaustion_and_allocation() {
    constexpr std::array<PrefillProfileEvent, 1> kEvents{kEmbedding};
    const std::size_t before = allocation_count.load(std::memory_order_relaxed);
    PrefillProfiler profiler(kEvents, 2);
    PrefillProfileDispatchTicket ticket;
    const PrefillProfilerError begin = profiler.begin_dispatch(kEmbedding, ticket);
    const PrefillProfilerError complete = profiler.complete_dispatch(ticket);
    const auto finalized = profiler.finalize();
    PrefillProfiler stage(
        kEvents, 2, CounterSamplingMode::StageBoundaryEncoderSplit);
    PrefillProfileSampleWindow window;
    const PrefillProfilerError stage_window_begin = stage.begin_sample_window();
    const PrefillProfilerError stage_begin =
        stage.begin_dispatch(kEmbedding, ticket);
    const PrefillProfilerError stage_complete = stage.complete_dispatch(ticket);
    const PrefillProfilerError stage_window_finish =
        stage.finish_sample_window(window);
    const auto stage_finalized = stage.finalize();
    const std::size_t after = allocation_count.load(std::memory_order_relaxed);
    check(begin == PrefillProfilerError::None && complete == PrefillProfilerError::None &&
              finalized && stage_window_begin == PrefillProfilerError::None &&
              stage_begin == PrefillProfilerError::None &&
              stage_complete == PrefillProfilerError::None &&
              stage_window_finish == PrefillProfilerError::None &&
              stage_finalized && before == after,
          "dispatch and stage-window recording allocate no heap memory");

    PrefillProfiler exhausted(kEvents, 2);
    check(exhausted.begin_dispatch(kEmbedding, ticket) == PrefillProfilerError::None &&
              exhausted.complete_dispatch(ticket) == PrefillProfilerError::None &&
              exhausted.begin_dispatch(kEmbedding, ticket) ==
                  PrefillProfilerError::EventSequenceExhausted &&
              exhausted.status().event_cursor == 1,
          "an event beyond the frozen plan fails at the exact cursor");
}

using DiagnosticProfiledOverload = ProfiledPrefillEncodeResult (*)(
    tatara::runtime::PrefillStep&, tatara::runtime::DecodeStep&, MetalComputePass&, std::uint32_t,
    std::uint32_t, std::span<const std::uint32_t>, PrefillProfiler&,
    const MetalCounterSampleBuffer&);
using BoundedProfiledOverload = ProfiledPrefillProgressResult (*)(tatara::runtime::PrefillStep&,
                                                                  tatara::runtime::DecodeStep&,
                                                                  MetalComputePass&,
                                                                  PrefillProfiler&,
                                                                  const MetalCounterSampleBuffer&);

static_assert(taxonomy_is_exact());
static_assert(std::is_trivially_copyable_v<PrefillProfileDispatchTicket>);
static_assert(!std::is_copy_constructible_v<PrefillProfiler>);
static_assert(!std::is_move_constructible_v<PrefillProfiler>);
static_assert(std::is_same_v<decltype(static_cast<DiagnosticProfiledOverload>(&encode_prefill)),
                             DiagnosticProfiledOverload>);
static_assert(std::is_same_v<decltype(static_cast<BoundedProfiledOverload>(&encode_prefill_unit)),
                             BoundedProfiledOverload>);

} // namespace

void* operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    std::abort();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    test_default_off_and_capacity();
    test_stage_windows_and_mode_binding();
    test_stage_public_state_precedence();
    test_cursor_order_and_canaries();
    test_mismatch_pending_and_counter_exits();
    test_sequence_exhaustion_and_allocation();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("prefill_profiler_contract_test: PASS");
    return 0;
}
