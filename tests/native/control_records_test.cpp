#include "tatara/runtime/control_records.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>

namespace {

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

} // namespace

void* operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
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
    using tatara::runtime::ControlEpoch;
    using tatara::runtime::ControlParty;
    using tatara::runtime::ControlStatus;
    using tatara::runtime::DrainRecord;
    using tatara::runtime::FatalReason;
    using tatara::runtime::FatalRecord;
    using tatara::runtime::FatalSnapshot;
    using tatara::runtime::RuntimeControlRecords;
    using tatara::runtime::StopRecord;
    using tatara::runtime::TimeRecord;
    using tatara::runtime::TimeSnapshot;
    using tatara::runtime::WakeRecord;

    static_assert(!std::is_copy_constructible_v<RuntimeControlRecords>);
    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();

    WakeRecord wake(ControlParty::Transport, ControlParty::Preparation);
    std::uint64_t observed_epoch = 0;
    check(wake.state() == ControlStatus::Idle, "wake begins idle");
    check(wake.publish(ControlParty::Engine, 0) == ControlStatus::WrongWriter,
          "wake rejects a foreign publisher");
    check(wake.publish(ControlParty::Transport, 0) == ControlStatus::Applied,
          "wake publication applies");
    check(wake.state() == ControlStatus::Pending, "published wake is pending");
    check(wake.publish(ControlParty::Transport, 0) == ControlStatus::Stale,
          "repeated stale publisher epoch is rejected");
    check(wake.observe(ControlParty::Preparation, 0, observed_epoch) == ControlStatus::Observed &&
              observed_epoch == 1,
          "consumer acquire-observes wake epoch");
    check(wake.observe(ControlParty::Preparation, 2, observed_epoch) == ControlStatus::Conflict,
          "future consumer epoch fails closed");
    check(wake.acknowledge(ControlParty::Preparation, 2) == ControlStatus::Conflict,
          "future wake acknowledgment fails closed");
    check(wake.acknowledge(ControlParty::Preparation, 1) == ControlStatus::Applied,
          "wake acknowledgment applies");
    check(wake.acknowledge(ControlParty::Preparation, 1) == ControlStatus::Repeated,
          "wake acknowledgment is idempotent");
    check(wake.state() == ControlStatus::Acknowledged, "wake reaches acknowledged");
    check(wake.publish(ControlParty::Transport, 1) == ControlStatus::Applied &&
              wake.publish(ControlParty::Transport, 2) == ControlStatus::Applied,
          "multiple wake publications may coalesce");
    check(wake.observe(ControlParty::Preparation, 1, observed_epoch) == ControlStatus::Observed &&
              observed_epoch == 3,
          "consumer observes the newest coalesced epoch");
    check(wake.acknowledge(ControlParty::Preparation, 3) == ControlStatus::Applied,
          "consumer may acknowledge coalesced epochs");
    const auto wake_evidence = wake.evidence();
    check(wake_evidence.evidence_valid && wake_evidence.conservation_holds &&
              wake_evidence.wake_acknowledged == 1 && wake_evidence.published_total == 3 &&
              wake_evidence.acknowledged_total == 3 && wake_evidence.pending_count == 0 &&
              wake_evidence.failed_retained_now == 0,
          "wake current/history evidence conserves acknowledged publications");

    WakeRecord retained_wake(ControlParty::Transport, ControlParty::Preparation);
    check(retained_wake.publish(ControlParty::Transport, 0) == ControlStatus::Applied &&
              retained_wake.retain_failed(ControlParty::Preparation, 1) == ControlStatus::Applied &&
              retained_wake.state() == ControlStatus::FailedRetained,
          "wake may retain an exact failed pending owner");
    const auto retained_evidence = retained_wake.evidence();
    check(retained_evidence.evidence_valid && retained_evidence.conservation_holds &&
              retained_evidence.wake_failed_retained == 1 &&
              retained_evidence.failed_retained_now == 1 &&
              retained_evidence.published_total == 1 && retained_evidence.acknowledged_total == 0 &&
              retained_wake.publish(ControlParty::Transport, 1) == ControlStatus::FailedRetained,
          "wake failed-retained state closes publication without losing history");

    WakeRecord exhausted_wake(ControlParty::Transport, ControlParty::Engine, maximum - 1);
    check(exhausted_wake.publish(ControlParty::Transport, maximum - 1) == ControlStatus::Applied,
          "last wake epoch is representable");
    check(exhausted_wake.publish(ControlParty::Transport, maximum) == ControlStatus::Exhausted,
          "wake epoch never wraps");
    check(exhausted_wake.acknowledge(ControlParty::Engine, maximum) == ControlStatus::Applied,
          "final wake remains acknowledgeable");
    check(exhausted_wake.state() == ControlStatus::Exhausted, "wake exhaustion remains latched");
    const auto exhausted_wake_evidence = exhausted_wake.evidence();
    check(!exhausted_wake_evidence.evidence_valid && exhausted_wake_evidence.conservation_holds &&
              exhausted_wake_evidence.wake_exhausted == 1 &&
              exhausted_wake_evidence.exhaustion_latched &&
              exhausted_wake_evidence.exhaustion_without_owner &&
              !exhausted_wake_evidence.exhaustion_owner_retained,
          "wake exhaustion invalidates evidence with exact owner disposition");

    TimeRecord time(ControlParty::Transport, ControlParty::Engine, 7);
    TimeSnapshot time_snapshot{};
    check(time.state() == ControlStatus::Idle, "time begins unpublished");
    check(time.publish(ControlParty::Preparation, 100) == ControlStatus::WrongWriter,
          "time rejects a foreign publisher");
    check(time.publish(ControlParty::Transport, 100) == ControlStatus::Applied, "time publishes");
    check(time.publish(ControlParty::Transport, 100) == ControlStatus::Repeated,
          "repeated time is explicit");
    check(time.publish(ControlParty::Transport, 99) == ControlStatus::Stale,
          "regressing time fails closed");
    check(time.read(ControlParty::Engine, 8, 100, 0, time_snapshot) == ControlStatus::ForeignEpoch,
          "foreign time epoch fails closed");
    check(time.read(ControlParty::Engine, 7, 105, 5, time_snapshot) == ControlStatus::Observed &&
              time_snapshot.epoch_id == 7 && time_snapshot.monotonic_units == 100,
          "fresh time snapshot is observed");
    check(time.read(ControlParty::Engine, 7, 106, 5, time_snapshot) == ControlStatus::Stale,
          "aged time snapshot fails closed");
    check(time.read(ControlParty::Engine, 7, 99, 5, time_snapshot) == ControlStatus::Stale,
          "observer time regression fails closed");
    check(time.stop(ControlParty::Transport) == ControlStatus::Applied &&
              time.stop(ControlParty::Transport) == ControlStatus::Repeated,
          "time stop is monotonic and idempotent");
    check(time.read(ControlParty::Engine, 7, 100, 0, time_snapshot) == ControlStatus::Stopped,
          "stopped time closes progress");

    TimeRecord exhausted_time(ControlParty::Transport, ControlParty::Engine, 9, maximum - 3);
    check(exhausted_time.publish(ControlParty::Transport, 1) == ControlStatus::Applied,
          "final even time sequence is published");
    check(exhausted_time.publish(ControlParty::Transport, 2) == ControlStatus::Exhausted,
          "time sequence never wraps");
    TimeRecord corrupt_time(ControlParty::Transport, ControlParty::Engine, 9, 3);
    check(corrupt_time.state() == ControlStatus::Corrupt, "odd time sequence is corrupt");

    DrainRecord drain(ControlParty::Transport, ControlParty::Preparation, ControlParty::Engine);
    check(drain.request(ControlParty::Transport, 0) == ControlStatus::Applied,
          "drain request applies");
    check(drain.state() == ControlStatus::Pending, "drain awaits both workers");
    check(drain.request(ControlParty::Transport, 1) == ControlStatus::Repeated,
          "repeat signal coalesces while drain is pending");
    check(drain.observe(ControlParty::Preparation, 0, observed_epoch) == ControlStatus::Observed &&
              observed_epoch == 1,
          "P observes drain");
    check(drain.acknowledge(ControlParty::Preparation, 2) == ControlStatus::Conflict,
          "future drain acknowledgment fails closed");
    check(drain.acknowledge(ControlParty::Preparation, 1) == ControlStatus::Applied,
          "P acknowledges drain");
    check(drain.state() == ControlStatus::Pending, "E acknowledgment is still required");
    check(drain.acknowledge(ControlParty::Engine, 1) == ControlStatus::Applied,
          "E acknowledges drain");
    check(drain.state() == ControlStatus::Acknowledged,
          "drain reaches acknowledged only after P and E");
    check(drain.acknowledge(ControlParty::Engine, 1) == ControlStatus::Repeated,
          "repeated drain acknowledgment is idempotent");

    DrainRecord exhausted_drain(ControlParty::Transport, ControlParty::Preparation,
                                ControlParty::Engine, maximum - 1);
    check(exhausted_drain.request(ControlParty::Transport, maximum - 1) == ControlStatus::Applied,
          "last drain epoch is representable");
    check(exhausted_drain.acknowledge(ControlParty::Preparation, maximum) ==
                  ControlStatus::Applied &&
              exhausted_drain.acknowledge(ControlParty::Engine, maximum) == ControlStatus::Applied,
          "last drain epoch is acknowledgeable");
    check(exhausted_drain.request(ControlParty::Transport, maximum) == ControlStatus::Exhausted,
          "drain epoch never wraps");

    FatalRecord fatal(ControlParty::Engine, ControlParty::Transport);
    FatalSnapshot fatal_snapshot{};
    check(fatal.publish(ControlParty::Preparation, 0, FatalReason::WorkerFailure) ==
              ControlStatus::WrongWriter,
          "fatal rejects a foreign publisher");
    check(fatal.publish(ControlParty::Engine, 0, FatalReason::None) == ControlStatus::InvalidValue,
          "fatal None cannot be published");
    check(fatal.publish(ControlParty::Engine, 0, static_cast<FatalReason>(255)) ==
              ControlStatus::InvalidValue,
          "fatal rejects values outside its closed reason domain");
    check(fatal.publish(ControlParty::Engine, 0, FatalReason::OwnershipInvariant) ==
              ControlStatus::Applied,
          "fatal latch applies");
    check(fatal.observe(ControlParty::Transport, fatal_snapshot) == ControlStatus::Observed &&
              fatal_snapshot.epoch == 1 && fatal_snapshot.reason == FatalReason::OwnershipInvariant,
          "transport observes exact fatal reason");
    check(fatal.publish(ControlParty::Engine, 1, FatalReason::OwnershipInvariant) ==
              ControlStatus::Repeated,
          "same fatal is idempotent");
    check(fatal.publish(ControlParty::Engine, 1, FatalReason::WorkerFailure) ==
              ControlStatus::Conflict,
          "a competing fatal cannot overwrite the first");
    check(fatal.publish(ControlParty::Engine, 0, FatalReason::WorkerFailure) ==
              ControlStatus::Stale,
          "stale fatal epoch has priority over reason conflict");
    FatalRecord exhausted_fatal(ControlParty::Engine, ControlParty::Transport, maximum);
    check(exhausted_fatal.publish(ControlParty::Engine, maximum, FatalReason::CounterExhausted) ==
              ControlStatus::Exhausted,
          "fatal generation never wraps");
    FatalSnapshot exhausted_fatal_snapshot{};
    check(exhausted_fatal.observe(ControlParty::Transport, exhausted_fatal_snapshot) ==
                  ControlStatus::Exhausted &&
              !exhausted_fatal_snapshot.evidence_valid &&
              exhausted_fatal_snapshot.exhaustion_latched &&
              exhausted_fatal_snapshot.exhausted_after_epoch == maximum &&
              exhausted_fatal_snapshot.exhausted_attempt_reason == FatalReason::CounterExhausted &&
              exhausted_fatal_snapshot.reason == FatalReason::CounterExhausted &&
              !exhausted_fatal_snapshot.exhaustion_without_owner &&
              exhausted_fatal_snapshot.exhaustion_owner_retained,
          "fatal exhaustion retains the first attempted event identity and owner");

    StopRecord stop(ControlParty::Supervisor, ControlParty::Preparation);
    check(stop.state() == ControlStatus::Idle, "stop begins idle");
    check(stop.request(ControlParty::Preparation, 0) == ControlStatus::WrongWriter,
          "worker cannot publish its own stop request");
    check(stop.request(ControlParty::Supervisor, 0) == ControlStatus::Applied,
          "supervisor publishes stop");
    check(stop.state() == ControlStatus::Pending, "stop is pending");
    check(stop.request(ControlParty::Supervisor, 1) == ControlStatus::Repeated,
          "repeat stop request is idempotent");
    check(stop.publish_stopped(ControlParty::Engine, 1) == ControlStatus::WrongWriter,
          "foreign worker cannot publish stopped");
    check(stop.publish_stopped(ControlParty::Preparation, 2) == ControlStatus::Conflict,
          "future stopped epoch fails closed");
    check(stop.publish_stopped(ControlParty::Preparation, 1) == ControlStatus::Applied,
          "worker publishes stopped for the exact request");
    check(stop.state() == ControlStatus::Stopped, "stop reaches stopped");
    check(stop.acknowledge_join(ControlParty::Preparation, 1) == ControlStatus::WrongWriter,
          "worker cannot acknowledge its own join");
    check(stop.acknowledge_join(ControlParty::Supervisor, 1) == ControlStatus::Applied,
          "supervisor acknowledges join");
    check(stop.state() == ControlStatus::Joined, "stop reaches joined");
    check(stop.acknowledge_join(ControlParty::Supervisor, 1) == ControlStatus::Repeated,
          "join acknowledgment is idempotent");

    StopRecord exhausted_stop(ControlParty::Supervisor, ControlParty::Engine, maximum - 1);
    check(exhausted_stop.request(ControlParty::Supervisor, maximum - 1) == ControlStatus::Applied &&
              exhausted_stop.publish_stopped(ControlParty::Engine, maximum) ==
                  ControlStatus::Applied &&
              exhausted_stop.acknowledge_join(ControlParty::Supervisor, maximum) ==
                  ControlStatus::Applied,
          "final stop/stopped/join generation completes");
    check(exhausted_stop.request(ControlParty::Supervisor, maximum) == ControlStatus::Exhausted,
          "stop generation never wraps");

    const std::size_t before = allocation_count.load(std::memory_order_relaxed);
    RuntimeControlRecords controls(17);
    check(controls.t_to_p.publish(ControlParty::Transport, 0) == ControlStatus::Applied,
          "fixed T-to-P wake exists");
    check(controls.p_to_t.publish(ControlParty::Preparation, 0) == ControlStatus::Applied,
          "fixed P-to-T wake exists");
    check(controls.t_to_e.publish(ControlParty::Transport, 0) == ControlStatus::Applied,
          "fixed T-to-E wake exists");
    check(controls.e_to_t.publish(ControlParty::Engine, 0) == ControlStatus::Applied,
          "fixed E-to-T wake exists");
    check(controls.time.publish(ControlParty::Transport, 1) == ControlStatus::Applied,
          "fixed time record operates");
    check(controls.drain.request(ControlParty::Transport, 0) == ControlStatus::Applied,
          "fixed drain record operates");
    check(controls.p_fatal.publish(ControlParty::Preparation, 0, FatalReason::WorkerFailure) ==
              ControlStatus::Applied,
          "fixed P fatal operates");
    check(controls.e_fatal.publish(ControlParty::Engine, 0, FatalReason::WorkerFailure) ==
              ControlStatus::Applied,
          "fixed E fatal operates");
    check(controls.p_stop.request(ControlParty::Supervisor, 0) == ControlStatus::Applied &&
              controls.e_stop.request(ControlParty::Supervisor, 0) == ControlStatus::Applied &&
              controls.t_stop.request(ControlParty::Supervisor, 0) == ControlStatus::Applied,
          "fixed P/E/T stop records operate");
    check(allocation_count.load(std::memory_order_relaxed) == before,
          "control construction and transitions allocate no heap memory");

    constexpr std::uint64_t kConcurrentEvents = 20000;
    constexpr std::uint64_t kConcurrentAttempts = kConcurrentEvents * 500;

    WakeRecord concurrent_wake(ControlParty::Transport, ControlParty::Preparation);
    std::atomic<bool> wake_begin{false};
    std::atomic<std::uint32_t> wake_failures{0};
    std::atomic<std::uint64_t> wake_last_seen{0};
    std::thread wake_publisher([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (wake_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            wake_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        for (std::uint64_t epoch = 0; epoch != kConcurrentEvents; ++epoch) {
            if (concurrent_wake.publish(ControlParty::Transport, epoch) != ControlStatus::Applied) {
                wake_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    });
    std::thread wake_consumer([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (wake_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            wake_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        ControlEpoch last_seen = 0;
        for (std::uint64_t attempt = 0;
             attempt != kConcurrentAttempts && last_seen != kConcurrentEvents; ++attempt) {
            ControlEpoch observed = 0;
            const ControlStatus status =
                concurrent_wake.observe(ControlParty::Preparation, last_seen, observed);
            if (status == ControlStatus::Idle) {
                if (observed != last_seen) {
                    wake_failures.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                continue;
            }
            if (status != ControlStatus::Observed || observed <= last_seen ||
                observed > kConcurrentEvents ||
                concurrent_wake.acknowledge(ControlParty::Preparation, observed) !=
                    ControlStatus::Applied) {
                wake_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            last_seen = observed;
        }
        wake_last_seen.store(last_seen, std::memory_order_release);
    });
    wake_begin.store(true, std::memory_order_release);
    wake_publisher.join();
    wake_consumer.join();
    const auto concurrent_wake_evidence = concurrent_wake.evidence();
    check(wake_failures.load(std::memory_order_relaxed) == 0 &&
              wake_last_seen.load(std::memory_order_acquire) == kConcurrentEvents &&
              concurrent_wake_evidence.evidence_valid &&
              concurrent_wake_evidence.conservation_holds &&
              concurrent_wake_evidence.published_total == kConcurrentEvents &&
              concurrent_wake_evidence.acknowledged_total == kConcurrentEvents,
          "two-thread wake epochs tolerate spurious polls and lose no publication");

    TimeRecord concurrent_time(ControlParty::Transport, ControlParty::Engine, 41);
    std::atomic<bool> time_begin{false};
    std::atomic<std::uint32_t> time_failures{0};
    std::atomic<std::uint64_t> time_last_seen{0};
    std::thread time_publisher([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (time_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            time_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        for (std::uint64_t units = 1; units <= kConcurrentEvents; ++units) {
            if (concurrent_time.publish(ControlParty::Transport, units) != ControlStatus::Applied) {
                time_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    });
    std::thread time_consumer([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (time_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            time_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::uint64_t last_units = 0;
        for (std::uint64_t attempt = 0;
             attempt != kConcurrentAttempts && last_units != kConcurrentEvents; ++attempt) {
            TimeSnapshot snapshot{};
            const ControlStatus status =
                concurrent_time.read(ControlParty::Engine, 41, kConcurrentEvents,
                                     std::numeric_limits<std::uint64_t>::max(), snapshot);
            if (status == ControlStatus::Idle || status == ControlStatus::Busy) {
                continue;
            }
            if (status != ControlStatus::Observed || snapshot.monotonic_units < last_units ||
                snapshot.monotonic_units > kConcurrentEvents ||
                (snapshot.publication_sequence & 1U) != 0) {
                time_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            last_units = snapshot.monotonic_units;
        }
        time_last_seen.store(last_units, std::memory_order_release);
    });
    time_begin.store(true, std::memory_order_release);
    time_publisher.join();
    time_consumer.join();
    check(time_failures.load(std::memory_order_relaxed) == 0 &&
              time_last_seen.load(std::memory_order_acquire) == kConcurrentEvents,
          "two-thread time seqlock overlap yields only coherent even snapshots");

    struct FatalCell {
        FatalRecord record{ControlParty::Engine, ControlParty::Transport};
    };
    constexpr std::size_t kFatalCells = 1024;
    std::array<FatalCell, kFatalCells> fatal_cells{};
    std::atomic<bool> fatal_begin{false};
    std::atomic<std::uint32_t> fatal_failures{0};
    std::atomic<std::size_t> fatal_observed{0};
    std::thread fatal_publisher([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (fatal_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            fatal_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        for (std::size_t index = 0; index != fatal_cells.size(); ++index) {
            const FatalReason reason =
                (index & 1U) == 0 ? FatalReason::WorkerFailure : FatalReason::OwnershipInvariant;
            if (fatal_cells[index].record.publish(ControlParty::Engine, 0, reason) !=
                ControlStatus::Applied) {
                fatal_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    });
    std::thread fatal_consumer([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (fatal_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            fatal_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::size_t index = 0;
        for (std::uint64_t attempt = 0;
             attempt != kConcurrentAttempts && index != fatal_cells.size(); ++attempt) {
            FatalSnapshot snapshot{};
            const ControlStatus status =
                fatal_cells[index].record.observe(ControlParty::Transport, snapshot);
            if (status == ControlStatus::Idle) {
                continue;
            }
            const FatalReason expected =
                (index & 1U) == 0 ? FatalReason::WorkerFailure : FatalReason::OwnershipInvariant;
            if (status != ControlStatus::Observed || snapshot.epoch != 1 ||
                snapshot.reason != expected || !snapshot.evidence_valid ||
                snapshot.exhaustion_latched) {
                fatal_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            ++index;
        }
        fatal_observed.store(index, std::memory_order_release);
    });
    fatal_begin.store(true, std::memory_order_release);
    fatal_publisher.join();
    fatal_consumer.join();
    check(fatal_failures.load(std::memory_order_relaxed) == 0 &&
              fatal_observed.load(std::memory_order_acquire) == kFatalCells,
          "two-thread fatal publication never exposes a mixed or corrupt snapshot");

    DrainRecord concurrent_drain(ControlParty::Transport, ControlParty::Preparation,
                                 ControlParty::Engine);
    StopRecord concurrent_stop(ControlParty::Supervisor, ControlParty::Preparation);
    std::atomic<bool> lifecycle_begin{false};
    std::atomic<std::uint32_t> lifecycle_failures{0};
    std::thread lifecycle_worker([&] {
        bool started = false;
        for (std::uint64_t spin = 0; spin != kConcurrentAttempts; ++spin) {
            if (lifecycle_begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            lifecycle_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        bool drain_done = false;
        bool stop_done = false;
        for (std::uint64_t attempt = 0;
             attempt != kConcurrentAttempts && (!drain_done || !stop_done); ++attempt) {
            if (!drain_done) {
                ControlEpoch preparation_epoch = 0;
                ControlEpoch engine_epoch = 0;
                const ControlStatus preparation_status =
                    concurrent_drain.observe(ControlParty::Preparation, 0, preparation_epoch);
                const ControlStatus engine_status =
                    concurrent_drain.observe(ControlParty::Engine, 0, engine_epoch);
                if (preparation_status == ControlStatus::Observed &&
                    engine_status == ControlStatus::Observed && preparation_epoch == 1 &&
                    engine_epoch == 1) {
                    drain_done = concurrent_drain.acknowledge(ControlParty::Preparation, 1) ==
                                     ControlStatus::Applied &&
                                 concurrent_drain.acknowledge(ControlParty::Engine, 1) ==
                                     ControlStatus::Applied;
                }
            }
            if (!stop_done && concurrent_stop.requested_epoch() == 1) {
                stop_done = concurrent_stop.publish_stopped(ControlParty::Preparation, 1) ==
                            ControlStatus::Applied;
            }
        }
        if (!drain_done || !stop_done) {
            lifecycle_failures.fetch_add(1, std::memory_order_relaxed);
        }
    });
    lifecycle_begin.store(true, std::memory_order_release);
    check(concurrent_drain.request(ControlParty::Transport, 0) == ControlStatus::Applied &&
              concurrent_stop.request(ControlParty::Supervisor, 0) == ControlStatus::Applied,
          "controller release-publishes concurrent drain and stop");
    bool lifecycle_complete = false;
    for (std::uint64_t attempt = 0; attempt != kConcurrentAttempts; ++attempt) {
        if (concurrent_drain.state() == ControlStatus::Acknowledged &&
            concurrent_stop.state() == ControlStatus::Stopped) {
            lifecycle_complete = true;
            break;
        }
    }
    lifecycle_worker.join();
    check(lifecycle_failures.load(std::memory_order_relaxed) == 0 && lifecycle_complete &&
              concurrent_stop.acknowledge_join(ControlParty::Supervisor, 1) ==
                  ControlStatus::Applied &&
              concurrent_stop.state() == ControlStatus::Joined,
          "two-thread drain acknowledgments and stop/stopped/join transitions complete");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("control_records_test: PASS");
    return 0;
}
