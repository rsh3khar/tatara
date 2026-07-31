#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tatara::runtime {

inline constexpr std::size_t kRuntimeCacheLineBytes = 64;

enum class RingPlanStatus : std::uint8_t {
    Ok,
    ZeroUsableCells,
    ZeroPayloadSize,
    InvalidAlignment,
    Overflow,
};

struct RingBytePlan {
    RingPlanStatus status{RingPlanStatus::Ok};
    std::size_t usable_cells{0};
    std::size_t physical_cells{0};
    std::size_t cell_stride{0};
    std::size_t head_offset{0};
    std::size_t tail_offset{0};
    std::size_t cells_offset{0};
    std::size_t bytes{0};

    constexpr explicit operator bool() const noexcept {
        return status == RingPlanStatus::Ok;
    }
};

namespace detail {

constexpr bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

constexpr bool checked_add(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    result = left + right;
    return true;
}

constexpr bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

constexpr bool checked_align_up(std::size_t value, std::size_t alignment,
                                std::size_t& result) noexcept {
    if (!is_power_of_two(alignment)) {
        return false;
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        return false;
    }
    result = (value + mask) & ~mask;
    return true;
}

enum class ExhaustionOwnerDisposition : std::uint8_t {
    None,
    WithoutOwner,
    OwnerRetained,
};

struct alignas(kRuntimeCacheLineBytes) RingConsumerCursor {
    std::atomic<std::uint64_t> sequence{0};
    std::uint64_t initial_sequence{0};
};

struct alignas(kRuntimeCacheLineBytes) RingProducerCursor {
    std::atomic<std::uint64_t> sequence{0};
    std::uint64_t initial_sequence{0};
    std::atomic<ExhaustionOwnerDisposition> exhaustion{ExhaustionOwnerDisposition::None};
};

static_assert(sizeof(RingConsumerCursor) == kRuntimeCacheLineBytes);
static_assert(sizeof(RingProducerCursor) == kRuntimeCacheLineBytes);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<ExhaustionOwnerDisposition>::is_always_lock_free);

} // namespace detail

constexpr RingBytePlan plan_spsc_ring_bytes(std::size_t usable_cells, std::size_t payload_size,
                                            std::size_t payload_alignment) noexcept {
    RingBytePlan plan{};
    plan.usable_cells = usable_cells;
    if (usable_cells == 0) {
        plan.status = RingPlanStatus::ZeroUsableCells;
        return plan;
    }
    if (payload_size == 0) {
        plan.status = RingPlanStatus::ZeroPayloadSize;
        return plan;
    }
    if (!detail::is_power_of_two(payload_alignment) || payload_alignment > kRuntimeCacheLineBytes) {
        plan.status = RingPlanStatus::InvalidAlignment;
        return plan;
    }
    if (!detail::checked_add(usable_cells, 1, plan.physical_cells) ||
        !detail::checked_align_up(payload_size, payload_alignment, plan.cell_stride)) {
        plan.status = RingPlanStatus::Overflow;
        return plan;
    }

    std::size_t cursor_bytes = 0;
    std::size_t cell_bytes = 0;
    std::size_t unaligned_bytes = 0;
    if (!detail::checked_multiply(kRuntimeCacheLineBytes, 2, cursor_bytes) ||
        !detail::checked_multiply(plan.physical_cells, plan.cell_stride, cell_bytes) ||
        !detail::checked_add(cursor_bytes, cell_bytes, unaligned_bytes) ||
        !detail::checked_align_up(unaligned_bytes, kRuntimeCacheLineBytes, plan.bytes)) {
        plan.status = RingPlanStatus::Overflow;
        plan.bytes = 0;
        return plan;
    }
    plan.head_offset = 0;
    plan.tail_offset = kRuntimeCacheLineBytes;
    plan.cells_offset = 2 * kRuntimeCacheLineBytes;
    return plan;
}

template <typename Payload, std::size_t UsableCells>
constexpr RingBytePlan plan_spsc_ring_bytes() noexcept {
    return plan_spsc_ring_bytes(UsableCells, sizeof(Payload), alignof(Payload));
}

enum class RingOperationStatus : std::uint8_t {
    Published,
    Consumed,
    Empty,
    Full,
    Exhausted,
    Corrupt,
};

enum class RingState : std::uint8_t {
    Ready,
    Empty,
    Full,
    Exhausted,
    Corrupt,
};

struct RingSnapshot {
    RingState state{RingState::Empty};
    std::size_t size{0};
    std::uint64_t head_sequence{0};
    std::uint64_t tail_sequence{0};
};

struct RingEvidenceSnapshot {
    std::size_t usable_cells{0};
    std::size_t free_cells{0};
    std::size_t published_cells{0};
    std::size_t failed_retained_cells{0};
    std::uint64_t published_total{0};
    std::uint64_t consumed_total{0};
    std::uint64_t failed_retained_total{0};
    bool evidence_valid{false};
    bool exhaustion_latched{false};
    bool exhaustion_without_owner{false};
    bool exhaustion_owner_retained{false};
    bool conservation_holds{false};
};

struct SpscInitialState {
    std::uint64_t head_sequence{0};
    std::uint64_t tail_sequence{0};
};

template <typename Payload, std::size_t UsableCells>
class alignas(kRuntimeCacheLineBytes) SpscRing {
    static_assert(UsableCells > 0, "an SPSC ring must have at least one usable cell");
    static_assert(UsableCells < std::numeric_limits<std::size_t>::max(),
                  "the sentinel cell must be representable");
    static_assert(std::is_trivially_copyable_v<Payload> &&
                      std::is_trivially_destructible_v<Payload>,
                  "ring payloads must have allocation-free value ownership");
    static_assert(std::is_trivially_default_constructible_v<Payload> &&
                      std::is_trivially_copy_assignable_v<Payload>,
                  "ring payload cells must support allocation-free construction and assignment");
    static_assert(alignof(Payload) <= kRuntimeCacheLineBytes,
                  "payload alignment must fit the frozen cache-line layout");

  public:
    using value_type = Payload;
    static constexpr std::size_t kUsableCells = UsableCells;
    static constexpr std::size_t kPhysicalCells = UsableCells + 1;
    static constexpr std::size_t kHeadOffset = 0;
    static constexpr std::size_t kTailOffset = kRuntimeCacheLineBytes;
    static constexpr std::size_t kCellsOffset = 2 * kRuntimeCacheLineBytes;

    SpscRing() noexcept = default;

    explicit SpscRing(SpscInitialState initial) noexcept {
        head_.sequence.store(initial.head_sequence, std::memory_order_relaxed);
        head_.initial_sequence = initial.head_sequence;
        tail_.sequence.store(initial.tail_sequence, std::memory_order_relaxed);
        tail_.initial_sequence = initial.tail_sequence;
    }

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;

    RingOperationStatus try_push(const Payload& payload) noexcept {
        const std::uint64_t tail = tail_.sequence.load(std::memory_order_relaxed);
        const std::uint64_t head = head_.sequence.load(std::memory_order_acquire);
        if (head > tail || tail - head > UsableCells) {
            return RingOperationStatus::Corrupt;
        }
        if (tail == std::numeric_limits<std::uint64_t>::max()) {
            tail_.exhaustion.store(detail::ExhaustionOwnerDisposition::WithoutOwner,
                                   std::memory_order_release);
            return RingOperationStatus::Exhausted;
        }
        if (tail - head == UsableCells) {
            return RingOperationStatus::Full;
        }

        cells_[static_cast<std::size_t>(tail % kPhysicalCells)] = payload;
        tail_.sequence.store(tail + 1, std::memory_order_release);
        return RingOperationStatus::Published;
    }

    RingOperationStatus try_pop(Payload& payload) noexcept {
        const std::uint64_t head = head_.sequence.load(std::memory_order_relaxed);
        const std::uint64_t tail = tail_.sequence.load(std::memory_order_acquire);
        if (head > tail || tail - head > UsableCells) {
            return RingOperationStatus::Corrupt;
        }
        if (head == tail) {
            return head == std::numeric_limits<std::uint64_t>::max()
                       ? RingOperationStatus::Exhausted
                       : RingOperationStatus::Empty;
        }

        payload = cells_[static_cast<std::size_t>(head % kPhysicalCells)];
        head_.sequence.store(head + 1, std::memory_order_release);
        return RingOperationStatus::Consumed;
    }

    RingSnapshot snapshot() const noexcept {
        const std::uint64_t head = head_.sequence.load(std::memory_order_acquire);
        const std::uint64_t tail = tail_.sequence.load(std::memory_order_acquire);
        RingSnapshot result{RingState::Ready, 0, head, tail};
        if (head > tail || tail - head > UsableCells) {
            result.state = RingState::Corrupt;
            return result;
        }
        result.size = static_cast<std::size_t>(tail - head);
        if (head == tail) {
            result.state = head == std::numeric_limits<std::uint64_t>::max() ? RingState::Exhausted
                                                                             : RingState::Empty;
        } else if (result.size == UsableCells) {
            result.state = RingState::Full;
        }
        return result;
    }

    RingEvidenceSnapshot evidence() const noexcept {
        const std::uint64_t head = head_.sequence.load(std::memory_order_acquire);
        const std::uint64_t tail = tail_.sequence.load(std::memory_order_acquire);
        const detail::ExhaustionOwnerDisposition exhaustion =
            tail_.exhaustion.load(std::memory_order_acquire);
        RingEvidenceSnapshot result;
        result.usable_cells = UsableCells;
        result.exhaustion_latched = exhaustion != detail::ExhaustionOwnerDisposition::None;
        result.exhaustion_without_owner =
            exhaustion == detail::ExhaustionOwnerDisposition::WithoutOwner;
        result.exhaustion_owner_retained =
            exhaustion == detail::ExhaustionOwnerDisposition::OwnerRetained;
        if (head > tail || tail - head > UsableCells || head < head_.initial_sequence ||
            tail < tail_.initial_sequence || head_.initial_sequence != tail_.initial_sequence) {
            return result;
        }
        result.published_cells = static_cast<std::size_t>(tail - head);
        result.free_cells = UsableCells - result.published_cells;
        result.published_total = tail - tail_.initial_sequence;
        result.consumed_total = head - head_.initial_sequence;
        result.conservation_holds =
            result.free_cells + result.published_cells + result.failed_retained_cells ==
                UsableCells &&
            result.published_total >= result.consumed_total &&
            result.published_total - result.consumed_total ==
                result.published_cells + result.failed_retained_total;
        result.evidence_valid = result.conservation_holds && !result.exhaustion_latched;
        return result;
    }

    static constexpr std::size_t head_offset() noexcept {
        return offsetof(SpscRing, head_);
    }

    static constexpr std::size_t tail_offset() noexcept {
        return offsetof(SpscRing, tail_);
    }

    static constexpr std::size_t cells_offset() noexcept {
        return offsetof(SpscRing, cells_);
    }

  private:
    detail::RingConsumerCursor head_{};
    detail::RingProducerCursor tail_{};
    std::array<Payload, kPhysicalCells> cells_{};
};

} // namespace tatara::runtime
