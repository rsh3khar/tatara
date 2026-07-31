#include "tatara/service/prefix_cache.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>

namespace {

std::size_t allocation_count = 0;
bool track_allocations = false;

} // namespace

void* operator new(std::size_t size) {
    if (track_allocations) {
        ++allocation_count;
    }
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

using namespace tatara::service;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

PrefixCacheDigest digest_value(std::uint8_t value) {
    PrefixCacheDigest digest{};
    digest.fill(value);
    return digest;
}

template <std::size_t Size>
PrefixCacheKey key(const std::array<std::uint32_t, Size>& tokens) {
    return {
        .token_digest = digest_prefix_tokens(tokens),
        .tokens = tokens,
    };
}

template <std::size_t Size>
PrefixCacheKey key_with_digest(const std::array<std::uint32_t, Size>& tokens,
                               const PrefixCacheDigest& digest) {
    return {
        .token_digest = digest,
        .tokens = tokens,
    };
}

PrefixCache make_cache(std::uint32_t entries = 3, std::uint64_t blocks = 6,
                       std::uint32_t max_tokens = 8, std::uint32_t origin = 1,
                       std::uint32_t stride = 1,
                       std::uint64_t generation_limit =
                           std::numeric_limits<std::uint64_t>::max()) {
    auto result = create_prefix_cache({
        .state_capacity_bytes = blocks * 64,
        .state_block_bytes = 64,
        .max_entries = entries,
        .max_tokens_per_entry = max_tokens,
        .boundary_origin_tokens = origin,
        .boundary_stride_tokens = stride,
        .domain = {.diagnostic_digest = digest_value(0xA1), .generation = 17},
        .generation_limit = generation_limit,
    });
    check(static_cast<bool>(result), "cache configuration constructs");
    return std::move(*result.cache);
}

PrefixCacheReservation reserve_and_commit(PrefixCache& cache, const PrefixCacheKey& prefix,
                                          std::uint64_t state_bytes) {
    auto result = cache.reserve_snapshot(cache.domain(), prefix, state_bytes);
    check(static_cast<bool>(result), "reservation succeeds");
    const PrefixCacheReservation reservation = *result.reservation;
    check(cache.mark_snapshot_pending_publication(reservation) == PrefixCacheError::None,
          "observed snapshot becomes pending publication");
    check(cache.commit_snapshot(reservation) == PrefixCacheError::None,
          "successful request publishes snapshot");
    return reservation;
}

template <std::size_t Size>
PrefixCacheLookupResult lookup(PrefixCache& cache,
                               const std::array<std::uint32_t, Size>& tokens,
                               std::uint32_t maximum_position, std::uint64_t request,
                               std::uint64_t slot) {
    return cache.lookup_longest(
        cache.domain(), tokens, maximum_position,
        {.owner_index = request, .owner_generation = 1},
        {.slot_index = slot, .slot_generation = 1});
}

void invalid_configuration_domain_and_boundaries_are_typed() {
    check(create_prefix_cache({}).error == PrefixCacheError::InvalidConfiguration,
          "zero configuration is invalid");
    auto zero_domain = create_prefix_cache({
        .state_capacity_bytes = 64,
        .state_block_bytes = 64,
        .max_entries = 1,
        .max_tokens_per_entry = 4,
        .boundary_origin_tokens = 2,
        .boundary_stride_tokens = 2,
        .domain = {.diagnostic_digest = {}, .generation = 1},
    });
    check(zero_domain.error == PrefixCacheError::InvalidConfiguration,
          "zero diagnostic domain is invalid");

    PrefixCache cache = make_cache(2, 2, 4, 2, 2);
    const std::array<std::uint32_t, 0> empty{};
    const std::array<std::uint32_t, 1> one{1};
    const std::array<std::uint32_t, 2> two{1, 2};
    const std::array<std::uint32_t, 3> three{1, 2, 3};
    const std::array<std::uint32_t, 5> five{1, 2, 3, 4, 5};
    check(cache.reserve_snapshot(cache.domain(), key(empty), 1).error ==
              PrefixCacheError::EmptyPrefix,
          "empty prefix is typed");
    check(cache.reserve_snapshot(cache.domain(), key(one), 1).error ==
              PrefixCacheError::BoundaryViolation,
          "position before origin is typed");
    check(cache.reserve_snapshot(cache.domain(), key(three), 1).error ==
              PrefixCacheError::BoundaryViolation,
          "origin-relative stride is enforced");
    check(cache.reserve_snapshot(cache.domain(), key(five), 1).error ==
              PrefixCacheError::TokenLimitExceeded,
          "token limit is typed before boundary");
    check(cache.reserve_snapshot({.generation = 18}, key(two), 1).error ==
              PrefixCacheError::ForeignDomain,
          "foreign domain generation is rejected");
    check(cache.reserve_snapshot(cache.domain(), key(two), 129).error ==
              PrefixCacheError::StateTooLarge,
          "state larger than usable arena is typed");
    check(!lookup(cache, two, 3, 1, 1) &&
              lookup(cache, two, 3, 1, 1).error ==
                  PrefixCacheError::InvalidMaximumPosition,
          "lookup maximum cannot exceed prepared tokens");
    check(cache.lookup_longest(cache.domain(), two, 2, {}, {.slot_generation = 1}).error ==
              PrefixCacheError::InvalidOwner,
          "zero request handle is rejected");
}

void publication_is_invisible_until_successful_terminal() {
    PrefixCache cache = make_cache();
    const std::array<std::uint32_t, 3> tokens{11, 22, 33};
    const auto reserved = cache.reserve_snapshot(cache.domain(), key(tokens), 65);
    check(reserved && reserved.reservation->state_offset_bytes == 0 &&
              reserved.reservation->allocated_state_bytes == 128,
          "reservation returns block-rounded state extent");
    PrefixCacheEvidence evidence = cache.evidence();
    check(evidence.reservation_entries == 1 && evidence.ready_entries == 0 &&
              evidence.conserved,
          "fresh snapshot reservation is owned and conserved");
    check(!lookup(cache, tokens, 3, 1, 1), "unobserved reservation is invisible");

    check(cache.mark_snapshot_pending_publication(*reserved.reservation) ==
              PrefixCacheError::None,
          "successful state copy waits for request terminal");
    evidence = cache.evidence();
    check(evidence.pending_publication_entries == 1 && evidence.ready_entries == 0,
          "pending publication remains invisible");
    check(!lookup(cache, tokens, 3, 1, 1), "pending publication cannot restore");
    check(cache.commit_snapshot(*reserved.reservation) == PrefixCacheError::None,
          "successful request terminal publishes");

    auto hit = lookup(cache, tokens, 3, 1, 1);
    check(hit && hit.lease->position_tokens == 3 && hit.lease->state_bytes == 65,
          "ready exact prefix returns a restore lease");
    check(cache.release_restore(*hit.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::None,
          "observed successful restore releases and touches");
    check(cache.reserve_snapshot(cache.domain(), key(tokens), 65).error ==
              PrefixCacheError::AlreadyPresent,
          "duplicate ready publication is typed");

    const std::array<std::uint32_t, 2> cancelled{44, 55};
    auto cancelled_reservation =
        cache.reserve_snapshot(cache.domain(), key(cancelled), 64);
    check(cancelled_reservation &&
              cache.mark_snapshot_pending_publication(*cancelled_reservation.reservation) ==
                  PrefixCacheError::None &&
              cache.abort_snapshot(*cancelled_reservation.reservation) ==
                  PrefixCacheError::None,
          "cancelled request aborts an observed unpublished snapshot");
    check(!lookup(cache, cancelled, 2, 2, 1), "cancelled request never publishes");
    check(cache.abort_snapshot(*cancelled_reservation.reservation) ==
              PrefixCacheError::StaleReservation,
          "aborted snapshot handle cannot be reused");
}

void longest_lookup_uses_exact_tokens_not_digest_aliases() {
    PrefixCache cache = make_cache(4, 8, 8);
    const PrefixCacheDigest collision = digest_value(0x5A);
    const std::array<std::uint32_t, 2> short_tokens{1, 2};
    const std::array<std::uint32_t, 4> long_tokens{1, 2, 3, 4};
    const std::array<std::uint32_t, 4> colliding_tokens{9, 8, 7, 6};
    reserve_and_commit(cache, key_with_digest(short_tokens, collision), 64);
    reserve_and_commit(cache, key_with_digest(long_tokens, collision), 64);
    reserve_and_commit(cache, key_with_digest(colliding_tokens, collision), 64);

    const std::array<std::uint32_t, 6> request{1, 2, 3, 4, 5, 6};
    auto longest = lookup(cache, request, 6, 1, 1);
    check(longest && longest.lease->position_tokens == 4,
          "greatest exact prefix wins despite forced digest collision");
    check(cache.release_restore(*longest.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::None,
          "longest lease releases");

    auto bounded = lookup(cache, request, 3, 2, 1);
    check(bounded && bounded.lease->position_tokens == 2,
          "maximum position bounds longest selection");
    check(cache.release_restore(*bounded.lease,
                                PrefixCacheRestoreDisposition::ObservedFailure) ==
              PrefixCacheError::None,
          "observed failed copy safely releases without a touch");

    const std::array<std::uint32_t, 4> first_mutation{0, 2, 3, 4};
    const std::array<std::uint32_t, 4> middle_mutation{1, 2, 0, 4};
    const std::array<std::uint32_t, 4> last_mutation{1, 2, 3, 0};
    check(!lookup(cache, first_mutation, 4, 3, 1), "first-token mutation misses");
    auto middle = lookup(cache, middle_mutation, 4, 4, 1);
    check(middle && middle.lease->position_tokens == 2,
          "middle mutation falls back only to the exact shorter prefix");
    check(cache.release_restore(*middle.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::None,
          "shorter fallback lease releases");
    auto last = lookup(cache, last_mutation, 4, 5, 1);
    check(last && last.lease->position_tokens == 2,
          "last mutation cannot alias the longer entry");
    check(cache.release_restore(*last.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::None,
          "last-mutation fallback lease releases");
}

void restore_leases_are_generation_owner_and_lru_checked() {
    PrefixCache cache = make_cache(2, 2);
    const std::array<std::uint32_t, 1> a{1};
    const std::array<std::uint32_t, 1> b{2};
    const std::array<std::uint32_t, 1> c{3};
    reserve_and_commit(cache, key(a), 64);
    reserve_and_commit(cache, key(b), 64);

    auto lease = lookup(cache, a, 1, 41, 7);
    check(lease && cache.evidence().restore_leased_entries == 1,
          "lookup atomically leases the extent");
    check(!lookup(cache, a, 1, 42, 8), "leased entry is invisible to a second restore");
    PrefixCacheRestoreLease foreign = *lease.lease;
    foreign.request.owner_index = 99;
    check(cache.release_restore(foreign, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::StaleRestoreLease,
          "foreign request cannot release a lease");

    auto blocked = cache.reserve_snapshot(cache.domain(), key(c), 128);
    check(blocked.error == PrefixCacheError::StateArenaExhausted,
          "leased entry is not an eviction victim");
    check(cache.release_restore(*lease.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::None,
          "matching lease releases");
    check(cache.release_restore(*lease.lease, PrefixCacheRestoreDisposition::Success) ==
              PrefixCacheError::StaleRestoreLease,
          "restore lease cannot be released twice");

    auto replacement = cache.reserve_snapshot(cache.domain(), key(c), 64);
    check(replacement && replacement.reservation->entry != lease.lease->entry,
          "successful restore touch protects the entry from next LRU eviction");
    check(cache.abort_snapshot(*replacement.reservation) == PrefixCacheError::None,
          "replacement abort releases its extent");
}

void victim_planning_is_atomic_and_deterministic() {
    {
        PrefixCache cache = make_cache(4, 3);
        const std::array<std::uint32_t, 1> a{1};
        const std::array<std::uint32_t, 1> b{2};
        const std::array<std::uint32_t, 1> c{3};
        const std::array<std::uint32_t, 1> d{4};
        auto first = cache.reserve_snapshot(cache.domain(), key(a), 64);
        reserve_and_commit(cache, key(b), 64);
        auto third = cache.reserve_snapshot(cache.domain(), key(c), 64);
        check(first && third, "pending extents bracket the ready block");
        const PrefixCacheEvidence before = cache.evidence();
        const auto failed = cache.reserve_snapshot(cache.domain(), key(d), 128);
        const PrefixCacheEvidence after = cache.evidence();
        check(failed.error == PrefixCacheError::StateArenaExhausted,
              "fragmented non-evictable arena is typed");
        check(before.state_digest == after.state_digest && before.lru_epoch == after.lru_epoch &&
                  after.ready_entries == 1 && after.reservation_entries == 2 &&
                  after.conserved,
              "failed victim plan mutates no authoritative cache state");
        check(cache.abort_snapshot(*first.reservation) == PrefixCacheError::None &&
                  cache.abort_snapshot(*third.reservation) == PrefixCacheError::None,
              "failed plan retained both pending owners");
    }

    {
        PrefixCache cache = make_cache(3, 4);
        const std::array<std::uint32_t, 1> a{1};
        const std::array<std::uint32_t, 1> b{2};
        const std::array<std::uint32_t, 1> c{3};
        const std::array<std::uint32_t, 1> d{4};
        reserve_and_commit(cache, key(a), 64);
        reserve_and_commit(cache, key(b), 64);
        reserve_and_commit(cache, key(c), 64);
        auto touch_a = lookup(cache, a, 1, 1, 1);
        check(touch_a &&
                  cache.release_restore(*touch_a.lease,
                                        PrefixCacheRestoreDisposition::Success) ==
                      PrefixCacheError::None,
              "A becomes newest only after restore success");

        auto large = cache.reserve_snapshot(cache.domain(), key(d), 128);
        check(large && large.reservation->state_offset_bytes == 64,
              "atomic LRU plan evicts B then C for blocks 1-2");
        check(cache.mark_snapshot_pending_publication(*large.reservation) ==
                      PrefixCacheError::None &&
                  cache.commit_snapshot(*large.reservation) == PrefixCacheError::None,
              "planned replacement publishes");
        auto a_hit = lookup(cache, a, 1, 2, 1);
        check(static_cast<bool>(a_hit),
              "newest A survives deterministic multi-victim plan");
        check(cache.release_restore(*a_hit.lease, PrefixCacheRestoreDisposition::Success) ==
                  PrefixCacheError::None,
              "surviving A lease releases");
        check(!lookup(cache, b, 1, 3, 1) && !lookup(cache, c, 1, 4, 1),
              "two oldest entries were evicted");
    }
}

void retained_failures_and_generation_retirement_never_reenter() {
    {
        PrefixCache cache = make_cache(1, 1);
        const std::array<std::uint32_t, 1> a{1};
        const std::array<std::uint32_t, 1> b{2};
        auto reservation = cache.reserve_snapshot(cache.domain(), key(a), 64);
        check(reservation &&
                  cache.retain_snapshot_failure(*reservation.reservation) ==
                      PrefixCacheError::None,
              "unobserved snapshot failure retains ownership");
        PrefixCacheEvidence evidence = cache.evidence();
        check(evidence.failed_retained_entries == 1 && evidence.free_state_blocks == 0 &&
                  evidence.conserved,
              "failed snapshot bytes never become reusable");
        check(cache.reserve_snapshot(cache.domain(), key(b), 64).error ==
                  PrefixCacheError::NoEvictableEntry,
              "failed-retained snapshot is not evictable");
    }

    {
        PrefixCache cache = make_cache(1, 1);
        const std::array<std::uint32_t, 1> a{1};
        const std::array<std::uint32_t, 1> b{2};
        reserve_and_commit(cache, key(a), 64);
        auto lease = lookup(cache, a, 1, 1, 1);
        check(lease &&
                  cache.release_restore(*lease.lease,
                                        PrefixCacheRestoreDisposition::RetainedFailure) ==
                      PrefixCacheError::None,
              "unobserved restore failure retains entry and extent");
        check(cache.evidence().failed_retained_entries == 1 &&
                  cache.reserve_snapshot(cache.domain(), key(b), 64).error ==
                      PrefixCacheError::NoEvictableEntry,
              "retained restore cannot be published or evicted");
    }

    {
        PrefixCache cache = make_cache(1, 1, 4, 1, 1, 2);
        const std::array<std::uint32_t, 1> a{1};
        const std::array<std::uint32_t, 1> b{2};
        const std::array<std::uint32_t, 1> c{3};
        reserve_and_commit(cache, key(a), 64);
        auto second = cache.reserve_snapshot(cache.domain(), key(b), 64);
        check(second && second.reservation->generation == 2,
              "evicted record advances its own generation");
        check(cache.abort_snapshot(*second.reservation) == PrefixCacheError::None,
              "final generation can be aborted safely");
        PrefixCacheEvidence evidence = cache.evidence();
        check(evidence.exhausted_entries == 1 && evidence.free_state_blocks == 1 &&
                  evidence.conserved,
              "generation limit retires record without wrapping");
        check(cache.reserve_snapshot(cache.domain(), key(c), 64).error ==
                  PrefixCacheError::GenerationExhausted,
              "retired record never silently reuses a generation");
    }
}

void token_digest_and_budget_are_explicit() {
    const std::array<std::uint32_t, 2> tokens{1, 0x01020304U};
    const PrefixCacheDigest expected{
        0xCD, 0x48, 0xBD, 0x37, 0xD3, 0x01, 0x64, 0xFF, 0xFC, 0x22, 0xB2,
        0xF6, 0xDB, 0x0A, 0x7F, 0x71, 0xD0, 0x99, 0x7F, 0x5E, 0xDA, 0x19,
        0x50, 0x95, 0x0E, 0xFB, 0xB3, 0xC5, 0x54, 0x68, 0x45, 0x76,
    };
    check(digest_prefix_tokens(tokens) == expected,
          "token digest is stable little-endian SHA-256");

    auto result = create_prefix_cache({
        .state_capacity_bytes = 130,
        .state_block_bytes = 64,
        .max_entries = 2,
        .max_tokens_per_entry = 8,
        .boundary_origin_tokens = 0,
        .boundary_stride_tokens = 1,
        .domain = {.diagnostic_digest = digest_value(1), .generation = 1},
    });
    check(static_cast<bool>(result), "budget fixture constructs");
    const PrefixCacheBudget& budget = result.cache->budget();
    check(budget.state_arena_bytes == 130 && budget.usable_state_bytes == 128 &&
              budget.unused_state_tail_bytes == 2,
          "state arena tail is explicit");
    check(budget.token_arena_bytes == 64, "fixed exact-token slots are charged");
    check(budget.metadata_payload_bytes != 0 &&
              budget.total_payload_bytes == budget.state_arena_bytes +
                                                budget.token_arena_bytes +
                                                budget.metadata_payload_bytes,
          "state, token, and all planning metadata are charged");
}

void total_budget_derives_metadata_entries_and_state_arena() {
    constexpr std::uint64_t kTotalBudget = 4096;
    auto result = create_prefix_cache_for_budget({
        .total_budget_bytes = kTotalBudget,
        .state_block_bytes = 64,
        .minimum_entry_state_bytes = 64,
        .max_tokens_per_entry = 8,
        .boundary_origin_tokens = 1,
        .boundary_stride_tokens = 1,
        .domain = {.diagnostic_digest = digest_value(1), .generation = 1},
    });
    check(static_cast<bool>(result), "complete cache budget derives a cache");
    const PrefixCacheBudget& budget = result.cache->budget();
    const PrefixCacheEvidence evidence = result.cache->evidence();
    check(budget.configured_total_budget_bytes == kTotalBudget &&
              budget.state_arena_bytes >= 64 && evidence.free_entries != 0 &&
              evidence.conserved,
          "budget derives nonzero entry and arena capacity");
    check(budget.total_payload_bytes + budget.unallocated_budget_bytes ==
              budget.configured_total_budget_bytes,
          "actual payload and explicit slack conserve total cache budget");

    auto insufficient = create_prefix_cache_for_budget({
        .total_budget_bytes = 32,
        .state_block_bytes = 64,
        .minimum_entry_state_bytes = 64,
        .max_tokens_per_entry = 8,
        .boundary_origin_tokens = 1,
        .boundary_stride_tokens = 1,
        .domain = {.diagnostic_digest = digest_value(1), .generation = 1},
    });
    check(insufficient.error == PrefixCacheError::MetadataBudgetInsufficient,
          "budget too small for one useful entry is typed at boot");
}

void operations_allocate_nothing_after_creation() {
    PrefixCache cache = make_cache(3, 4);
    const std::array<std::uint32_t, 2> a{1, 2};
    const std::array<std::uint32_t, 2> b{3, 4};
    const std::size_t before = allocation_count;
    track_allocations = true;

    auto first = cache.reserve_snapshot(cache.domain(), key(a), 64);
    cache.mark_snapshot_pending_publication(*first.reservation);
    cache.commit_snapshot(*first.reservation);
    auto hit = lookup(cache, a, 2, 1, 1);
    cache.release_restore(*hit.lease, PrefixCacheRestoreDisposition::Success);
    auto second = cache.reserve_snapshot(cache.domain(), key(b), 65);
    cache.abort_snapshot(*second.reservation);
    cache.evidence();

    track_allocations = false;
    check(allocation_count == before,
          "reserve/publish/lookup/lease/release/abort/evidence allocate nothing");
}

} // namespace

int main() {
    check(prefix_cache_error_name(PrefixCacheError::None) == "none" &&
              prefix_cache_error_name(
                  PrefixCacheError::StateArenaExhausted) ==
                  "state-arena-exhausted" &&
              prefix_cache_error_name(
                  PrefixCacheError::InvalidTransition) ==
                  "invalid-transition",
          "cache outcomes have stable operator-visible names");
    invalid_configuration_domain_and_boundaries_are_typed();
    publication_is_invisible_until_successful_terminal();
    longest_lookup_uses_exact_tokens_not_digest_aliases();
    restore_leases_are_generation_owner_and_lru_checked();
    victim_planning_is_atomic_and_deterministic();
    retained_failures_and_generation_retirement_never_reenter();
    token_digest_and_budget_are_explicit();
    total_budget_derives_metadata_entries_and_state_arena();
    operations_allocate_nothing_after_creation();
    if (failures == 0) {
        std::printf("prefix_cache: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
