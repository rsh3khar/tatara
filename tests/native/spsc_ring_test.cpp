#include "tatara/runtime/spsc_ring.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>

namespace {

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

struct BytePayload {
    std::uint8_t value;
};

struct alignas(8) EightBytePayload {
    std::uint64_t value;
};

struct alignas(32) ThirtyTwoBytePayload {
    std::uint64_t values[4];
};

struct ConcurrentPayload {
    std::uint64_t sequence;
    std::uint64_t inverse;
    std::uint64_t pattern[4];
};

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
    using tatara::runtime::plan_spsc_ring_bytes;
    using tatara::runtime::RingOperationStatus;
    using tatara::runtime::RingPlanStatus;
    using tatara::runtime::RingState;
    using tatara::runtime::SpscInitialState;
    using tatara::runtime::SpscRing;

    static_assert(!std::is_copy_constructible_v<SpscRing<std::uint32_t, 1>>);
    static_assert(!std::is_move_constructible_v<SpscRing<std::uint32_t, 1>>);
    static_assert(SpscRing<std::uint32_t, 1>::kUsableCells == 1);
    static_assert(SpscRing<std::uint32_t, 1>::kPhysicalCells == 2);

    constexpr auto one_cell_plan = plan_spsc_ring_bytes<std::uint32_t, 1>();
    static_assert(one_cell_plan.status == RingPlanStatus::Ok);
    static_assert(one_cell_plan.usable_cells == 1);
    static_assert(one_cell_plan.physical_cells == 2);
    static_assert(one_cell_plan.cell_stride == sizeof(std::uint32_t));
    static_assert(one_cell_plan.bytes == 192);
    static_assert(sizeof(SpscRing<std::uint32_t, 1>) == one_cell_plan.bytes);
    static_assert(sizeof(SpscRing<std::uint64_t, 3>) ==
                  plan_spsc_ring_bytes<std::uint64_t, 3>().bytes);
    static_assert(std::is_standard_layout_v<SpscRing<BytePayload, 1>>);
    static_assert(SpscRing<BytePayload, 1>::head_offset() == 0);
    static_assert(SpscRing<BytePayload, 1>::tail_offset() == 64);
    static_assert(SpscRing<BytePayload, 1>::cells_offset() == 128);
    static_assert(SpscRing<BytePayload, 1>::head_offset() ==
                  plan_spsc_ring_bytes<BytePayload, 1>().head_offset);
    static_assert(SpscRing<BytePayload, 1>::tail_offset() ==
                  plan_spsc_ring_bytes<BytePayload, 1>().tail_offset);
    static_assert(SpscRing<BytePayload, 1>::cells_offset() ==
                  plan_spsc_ring_bytes<BytePayload, 1>().cells_offset);
    static_assert(sizeof(SpscRing<BytePayload, 1>) == plan_spsc_ring_bytes<BytePayload, 1>().bytes);
    static_assert(sizeof(SpscRing<EightBytePayload, 3>) ==
                  plan_spsc_ring_bytes<EightBytePayload, 3>().bytes);
    static_assert(sizeof(SpscRing<ThirtyTwoBytePayload, 7>) ==
                  plan_spsc_ring_bytes<ThirtyTwoBytePayload, 7>().bytes);
    static_assert(SpscRing<ThirtyTwoBytePayload, 7>::head_offset() == 0);
    static_assert(SpscRing<ThirtyTwoBytePayload, 7>::tail_offset() == 64);
    static_assert(SpscRing<ThirtyTwoBytePayload, 7>::cells_offset() == 128);

    const auto zero_cells = plan_spsc_ring_bytes(0, 4, 4);
    check(zero_cells.status == RingPlanStatus::ZeroUsableCells, "zero usable cells are rejected");
    const auto zero_payload = plan_spsc_ring_bytes(1, 0, 1);
    check(zero_payload.status == RingPlanStatus::ZeroPayloadSize, "zero payload size is rejected");
    const auto bad_alignment = plan_spsc_ring_bytes(1, 4, 3);
    check(bad_alignment.status == RingPlanStatus::InvalidAlignment,
          "non-power-of-two alignment is rejected");
    const auto excessive_alignment = plan_spsc_ring_bytes(1, 128, 128);
    check(excessive_alignment.status == RingPlanStatus::InvalidAlignment,
          "payload alignment beyond the cache-line layout is rejected");
    const auto physical_overflow =
        plan_spsc_ring_bytes(std::numeric_limits<std::size_t>::max(), 1, 1);
    check(physical_overflow.status == RingPlanStatus::Overflow,
          "physical-cell addition overflow is rejected");
    const auto product_overflow =
        plan_spsc_ring_bytes(std::numeric_limits<std::size_t>::max() / 2, 16, 8);
    check(product_overflow.status == RingPlanStatus::Overflow,
          "cell byte multiplication overflow is rejected");

    SpscRing<std::uint32_t, 1> single;
    std::uint32_t value = 0;
    const auto initial_evidence = single.evidence();
    check(initial_evidence.evidence_valid && initial_evidence.conservation_holds &&
              initial_evidence.free_cells == 1 && initial_evidence.published_cells == 0 &&
              initial_evidence.published_total == 0 && initial_evidence.consumed_total == 0,
          "empty ring evidence conserves its one usable cell");
    check(single.try_pop(value) == RingOperationStatus::Empty, "N=1 begins empty");
    check(single.try_push(11) == RingOperationStatus::Published,
          "N=1 publishes its one usable cell");
    check(single.snapshot().state == RingState::Full, "N=1 distinguishes full");
    check(single.try_push(12) == RingOperationStatus::Full, "N=1 refuses a second publication");
    check(single.try_pop(value) == RingOperationStatus::Consumed && value == 11,
          "N=1 consumes the published value");
    check(single.snapshot().state == RingState::Empty, "N=1 becomes empty");
    check(single.try_push(12) == RingOperationStatus::Published &&
              single.try_pop(value) == RingOperationStatus::Consumed && value == 12,
          "N=1 reuses the sentinel layout after wrap");
    const auto single_evidence = single.evidence();
    check(single_evidence.evidence_valid && single_evidence.conservation_holds &&
              single_evidence.free_cells == 1 && single_evidence.published_cells == 0 &&
              single_evidence.failed_retained_cells == 0 && single_evidence.published_total == 2 &&
              single_evidence.consumed_total == 2 && single_evidence.failed_retained_total == 0,
          "ring current/history evidence remains exact after wrap");

    SpscRing<std::uint32_t, 3> fifo;
    check(fifo.try_push(1) == RingOperationStatus::Published &&
              fifo.try_push(2) == RingOperationStatus::Published &&
              fifo.try_push(3) == RingOperationStatus::Published,
          "three-cell ring fills exactly");
    check(fifo.try_pop(value) == RingOperationStatus::Consumed && value == 1, "FIFO 1");
    check(fifo.try_push(4) == RingOperationStatus::Published,
          "publication wraps through the physical sentinel");
    check(fifo.try_pop(value) == RingOperationStatus::Consumed && value == 2, "FIFO 2");
    check(fifo.try_pop(value) == RingOperationStatus::Consumed && value == 3, "FIFO 3");
    check(fifo.try_pop(value) == RingOperationStatus::Consumed && value == 4, "FIFO 4");

    SpscRing<std::uint64_t, 7> cycling;
    for (std::uint64_t expected = 0; expected != 100000; ++expected) {
        check(cycling.try_push(expected) == RingOperationStatus::Published,
              "bounded long-cycle publish");
        std::uint64_t observed = 0;
        check(cycling.try_pop(observed) == RingOperationStatus::Consumed && observed == expected,
              "bounded long-cycle FIFO consume");
    }

    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    SpscRing<std::uint32_t, 2> near_exhaustion{SpscInitialState{maximum - 1, maximum - 1}};
    check(near_exhaustion.try_push(91) == RingOperationStatus::Published,
          "the final representable publication succeeds");
    check(near_exhaustion.try_push(92) == RingOperationStatus::Exhausted,
          "producer sequence never wraps");
    check(near_exhaustion.try_pop(value) == RingOperationStatus::Consumed && value == 91,
          "the final published value remains consumable");
    check(near_exhaustion.try_pop(value) == RingOperationStatus::Exhausted,
          "drained maximum sequence is explicitly exhausted");
    check(near_exhaustion.snapshot().state == RingState::Exhausted,
          "snapshot exposes terminal exhaustion");
    const auto exhausted_evidence = near_exhaustion.evidence();
    check(!exhausted_evidence.evidence_valid && exhausted_evidence.conservation_holds &&
              exhausted_evidence.exhaustion_latched &&
              exhausted_evidence.exhaustion_without_owner &&
              !exhausted_evidence.exhaustion_owner_retained,
          "ring exhaustion invalidates evidence and records owner disposition");

    SpscRing<std::uint32_t, 2> reversed{SpscInitialState{8, 7}};
    check(reversed.try_push(1) == RingOperationStatus::Corrupt &&
              reversed.try_pop(value) == RingOperationStatus::Corrupt,
          "reversed cursors fail closed");
    SpscRing<std::uint32_t, 2> overfull{SpscInitialState{1, 4}};
    check(overfull.snapshot().state == RingState::Corrupt,
          "distance beyond usable capacity fails closed");

    const std::size_t before = allocation_count.load(std::memory_order_relaxed);
    SpscRing<std::uint64_t, 4> allocation_free;
    std::uint64_t allocation_value = 0;
    for (std::uint64_t index = 0; index != 1000; ++index) {
        check(allocation_free.try_push(index) == RingOperationStatus::Published,
              "allocation-free publish succeeds");
        check(allocation_free.try_pop(allocation_value) == RingOperationStatus::Consumed,
              "allocation-free consume succeeds");
        (void)allocation_free.snapshot();
    }
    check(allocation_count.load(std::memory_order_relaxed) == before,
          "ring construction and operations allocate no heap memory");

    constexpr std::uint64_t kConcurrentItems = 50000;
    constexpr std::uint64_t kMaximumAttempts = kConcurrentItems * 200;
    SpscRing<ConcurrentPayload, 1> concurrent;
    std::atomic<bool> begin{false};
    std::atomic<bool> producer_started{false};
    std::atomic<bool> consumer_started{false};
    std::atomic<std::uint64_t> produced{0};
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<std::uint32_t> concurrent_failures{0};

    std::thread producer([&] {
        producer_started.store(true, std::memory_order_release);
        bool started = false;
        for (std::uint64_t spin = 0; spin != kMaximumAttempts; ++spin) {
            if (begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            concurrent_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::uint64_t next = 0;
        for (std::uint64_t attempt = 0; attempt != kMaximumAttempts && next != kConcurrentItems;
             ++attempt) {
            ConcurrentPayload payload{
                next,
                ~next,
                {next ^ 0x1111111111111111ULL, next ^ 0x2222222222222222ULL,
                 next ^ 0x4444444444444444ULL, next ^ 0x8888888888888888ULL},
            };
            const RingOperationStatus status = concurrent.try_push(payload);
            if (status == RingOperationStatus::Published) {
                ++next;
            } else if (status != RingOperationStatus::Full) {
                concurrent_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        produced.store(next, std::memory_order_release);
    });

    std::thread consumer([&] {
        consumer_started.store(true, std::memory_order_release);
        bool started = false;
        for (std::uint64_t spin = 0; spin != kMaximumAttempts; ++spin) {
            if (begin.load(std::memory_order_acquire)) {
                started = true;
                break;
            }
        }
        if (!started) {
            concurrent_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::uint64_t expected = 0;
        for (std::uint64_t attempt = 0; attempt != kMaximumAttempts && expected != kConcurrentItems;
             ++attempt) {
            ConcurrentPayload payload{};
            const RingOperationStatus status = concurrent.try_pop(payload);
            if (status == RingOperationStatus::Empty) {
                continue;
            }
            if (status != RingOperationStatus::Consumed || payload.sequence != expected ||
                payload.inverse != ~expected ||
                payload.pattern[0] != (expected ^ 0x1111111111111111ULL) ||
                payload.pattern[1] != (expected ^ 0x2222222222222222ULL) ||
                payload.pattern[2] != (expected ^ 0x4444444444444444ULL) ||
                payload.pattern[3] != (expected ^ 0x8888888888888888ULL)) {
                concurrent_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            ++expected;
        }
        consumed.store(expected, std::memory_order_release);
    });

    for (std::uint64_t spin = 0;
         spin != kMaximumAttempts && (!producer_started.load(std::memory_order_acquire) ||
                                      !consumer_started.load(std::memory_order_acquire));
         ++spin) {
    }
    begin.store(true, std::memory_order_release);
    producer.join();
    consumer.join();
    check(concurrent_failures.load(std::memory_order_relaxed) == 0 &&
              produced.load(std::memory_order_acquire) == kConcurrentItems &&
              consumed.load(std::memory_order_acquire) == kConcurrentItems,
          "two-thread N=1 release/acquire payload handoff is lossless and bounded");
    const auto concurrent_evidence = concurrent.evidence();
    check(concurrent_evidence.evidence_valid && concurrent_evidence.conservation_holds &&
              concurrent_evidence.published_total == kConcurrentItems &&
              concurrent_evidence.consumed_total == kConcurrentItems &&
              concurrent_evidence.free_cells == 1,
          "two-thread wrap history and conservation are exact");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("spsc_ring_test: PASS");
    return 0;
}
