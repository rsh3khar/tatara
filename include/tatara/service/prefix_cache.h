#pragma once

#include "tatara/model/sha256.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace tatara::service {

using PrefixCacheDigest = model::Sha256Digest;

struct PrefixCacheRequestHandle {
    std::uint64_t owner_index{0};
    std::uint64_t owner_generation{0};
    friend constexpr bool operator==(const PrefixCacheRequestHandle&,
                                     const PrefixCacheRequestHandle&) = default;
};

struct PrefixCacheSlotHandle {
    std::uint64_t slot_index{0};
    std::uint64_t slot_generation{0};
    friend constexpr bool operator==(const PrefixCacheSlotHandle&,
                                     const PrefixCacheSlotHandle&) = default;
};

inline constexpr std::uint64_t kPrefixCacheNoGeneration = 0;

struct PrefixCacheDomain {
    PrefixCacheDigest diagnostic_digest{};
    std::uint64_t generation{kPrefixCacheNoGeneration};
};

struct PrefixCacheDomainHandle {
    std::uint64_t generation{kPrefixCacheNoGeneration};
};

// The digest is a shortlist only. Exact token comparison is always required.
struct PrefixCacheKey {
    PrefixCacheDigest token_digest{};
    std::span<const std::uint32_t> tokens;
};

PrefixCacheDigest digest_prefix_tokens(std::span<const std::uint32_t> tokens) noexcept;

struct PrefixCacheConfiguration {
    std::uint64_t state_capacity_bytes{0};
    std::uint64_t state_block_bytes{0};
    std::uint32_t max_entries{0};
    std::uint32_t max_tokens_per_entry{0};
    std::uint32_t boundary_origin_tokens{0};
    std::uint32_t boundary_stride_tokens{1};
    PrefixCacheDomain domain{};
    // Production uses uint64 max. A lower value exists only to make generation
    // retirement directly testable without undefined wraparound.
    std::uint64_t generation_limit{std::numeric_limits<std::uint64_t>::max()};
};

struct PrefixCacheBudget {
    std::uint64_t configured_total_budget_bytes{0};
    std::uint64_t state_arena_bytes{0};
    std::uint64_t usable_state_bytes{0};
    std::uint64_t unused_state_tail_bytes{0};
    std::uint64_t token_arena_bytes{0};
    std::uint64_t metadata_payload_bytes{0};
    std::uint64_t total_payload_bytes{0};
    std::uint64_t unallocated_budget_bytes{0};
};

enum class PrefixCacheError : std::uint8_t {
    None,
    InvalidConfiguration,
    AllocationFailed,
    MetadataBudgetInsufficient,
    StateArenaBudgetInsufficient,
    ForeignDomain,
    InvalidOwner,
    EmptyPrefix,
    TokenLimitExceeded,
    BoundaryViolation,
    InvalidMaximumPosition,
    StateTooLarge,
    AlreadyPresent,
    PublicationInFlight,
    NoEvictableEntry,
    StateArenaExhausted,
    GenerationExhausted,
    StaleReservation,
    StaleRestoreLease,
    InvalidTransition,
};

std::string_view prefix_cache_error_name(PrefixCacheError error) noexcept;

struct PrefixCacheReservation {
    std::uint64_t domain_generation{kPrefixCacheNoGeneration};
    std::uint32_t entry{0};
    std::uint64_t generation{kPrefixCacheNoGeneration};
    std::uint32_t position_tokens{0};
    std::uint64_t state_offset_bytes{0};
    std::uint64_t state_bytes{0};
    std::uint64_t allocated_state_bytes{0};
};

struct PrefixCacheReserveResult {
    PrefixCacheError error{PrefixCacheError::None};
    std::optional<PrefixCacheReservation> reservation;

    explicit operator bool() const noexcept {
        return error == PrefixCacheError::None && reservation.has_value();
    }
};

struct PrefixCacheRestoreLease {
    std::uint64_t domain_generation{kPrefixCacheNoGeneration};
    std::uint32_t entry{0};
    std::uint64_t generation{kPrefixCacheNoGeneration};
    PrefixCacheRequestHandle request{};
    PrefixCacheSlotHandle slot{};
    std::uint32_t position_tokens{0};
    std::uint64_t state_offset_bytes{0};
    std::uint64_t state_bytes{0};
    std::uint64_t allocated_state_bytes{0};
};

struct PrefixCacheLookupResult {
    PrefixCacheError error{PrefixCacheError::None};
    bool hit{false};
    std::optional<PrefixCacheRestoreLease> lease;

    explicit operator bool() const noexcept {
        return error == PrefixCacheError::None && hit && lease.has_value();
    }
};

enum class PrefixCacheRestoreDisposition : std::uint8_t {
    Success,
    ObservedFailure,
    RetainedFailure,
};

struct PrefixCacheEvidence {
    std::size_t free_entries{0};
    std::size_t reservation_entries{0};
    std::size_t pending_publication_entries{0};
    std::size_t ready_entries{0};
    std::size_t restore_leased_entries{0};
    std::size_t failed_retained_entries{0};
    std::size_t exhausted_entries{0};
    std::size_t free_state_blocks{0};
    std::uint64_t lru_epoch{0};
    PrefixCacheDigest state_digest{};
    bool conserved{false};
};

class PrefixCache;
struct PrefixCacheCreateResult;
struct PrefixCacheBudgetCreateResult;

struct PrefixCacheBudgetConfiguration {
    std::uint64_t total_budget_bytes{0};
    std::uint64_t state_block_bytes{0};
    std::uint64_t minimum_entry_state_bytes{0};
    std::uint32_t max_tokens_per_entry{0};
    std::uint32_t boundary_origin_tokens{0};
    std::uint32_t boundary_stride_tokens{1};
    PrefixCacheDomain domain{};
    std::uint64_t generation_limit{std::numeric_limits<std::uint64_t>::max()};
};

// Creates all host metadata and token storage for an externally owned,
// startup-allocated Metal state arena. Request-path methods do not allocate.
class PrefixCache {
  public:
    PrefixCache(PrefixCache&&) noexcept = default;
    PrefixCache& operator=(PrefixCache&&) noexcept = default;
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    PrefixCacheDomainHandle domain() const noexcept;
    const PrefixCacheDigest& diagnostic_domain_digest() const noexcept;

    PrefixCacheReserveResult
    reserve_snapshot(PrefixCacheDomainHandle domain, const PrefixCacheKey& key,
                     std::uint64_t state_bytes) noexcept;
    PrefixCacheError
    mark_snapshot_pending_publication(const PrefixCacheReservation& reservation) noexcept;
    PrefixCacheError commit_snapshot(const PrefixCacheReservation& reservation) noexcept;
    PrefixCacheError abort_snapshot(const PrefixCacheReservation& reservation) noexcept;
    PrefixCacheError retain_snapshot_failure(const PrefixCacheReservation& reservation) noexcept;

    PrefixCacheLookupResult
    lookup_longest(PrefixCacheDomainHandle domain,
                   std::span<const std::uint32_t> request_prefill_tokens,
                   std::uint32_t maximum_position, PrefixCacheRequestHandle request,
                   PrefixCacheSlotHandle slot) noexcept;
    PrefixCacheError
    release_restore(const PrefixCacheRestoreLease& lease,
                    PrefixCacheRestoreDisposition disposition) noexcept;

    const PrefixCacheBudget& budget() const noexcept;
    PrefixCacheEvidence evidence() const noexcept;

  private:
    enum class EntryState : std::uint8_t {
        Free,
        Reservation,
        PendingPublication,
        Ready,
        RestoreLeased,
        FailedRetained,
        Exhausted,
    };

    struct Entry {
        EntryState state{EntryState::Free};
        PrefixCacheDigest token_digest{};
        std::uint32_t token_count{0};
        std::uint32_t first_block{0};
        std::uint32_t block_count{0};
        std::uint64_t state_bytes{0};
        std::uint64_t last_use_epoch{0};
        std::uint64_t generation{0};
        PrefixCacheRequestHandle request{};
        PrefixCacheSlotHandle slot{};
    };

    PrefixCache(PrefixCacheConfiguration configuration, PrefixCacheBudget budget,
                std::size_t token_slots, std::size_t state_blocks);

    friend PrefixCacheCreateResult
    create_prefix_cache(const PrefixCacheConfiguration& configuration);
    friend PrefixCacheBudgetCreateResult
    create_prefix_cache_for_budget(const PrefixCacheBudgetConfiguration& configuration);

    bool valid_domain(PrefixCacheDomainHandle domain) const noexcept;
    bool valid_key(const PrefixCacheKey& key, PrefixCacheError& error) const noexcept;
    bool exact_tokens(std::uint32_t entry, std::span<const std::uint32_t> tokens) const noexcept;
    bool matching_metadata(const Entry& entry, const PrefixCacheKey& key) const noexcept;
    bool entry_in_plan(std::uint32_t entry, std::size_t plan_count) const noexcept;
    std::uint32_t find_usable_free_entry() const noexcept;
    std::size_t build_lru_victim_order() noexcept;
    std::uint32_t find_virtual_free_extent(std::uint32_t blocks,
                                           std::size_t plan_count) const noexcept;
    void evict_for_plan(std::uint32_t entry) noexcept;
    void release_blocks(const Entry& entry) noexcept;
    void assign_blocks(std::uint32_t entry, std::uint32_t first, std::uint32_t count) noexcept;
    void touch(std::uint32_t entry) noexcept;
    void rebase_epochs() noexcept;
    void clear_entry_preserving_generation(std::uint32_t entry) noexcept;
    void retain_entry(std::uint32_t entry) noexcept;
    std::span<std::uint32_t> token_slot(std::uint32_t entry) noexcept;
    std::span<const std::uint32_t> token_slot(std::uint32_t entry) const noexcept;
    bool reservation_matches(const PrefixCacheReservation& reservation,
                             EntryState expected) const noexcept;
    bool restore_lease_matches(const PrefixCacheRestoreLease& lease) const noexcept;
    PrefixCacheDigest state_digest() const noexcept;

    PrefixCacheConfiguration configuration_;
    PrefixCacheBudget budget_;
    std::vector<Entry> entries_;
    std::vector<std::uint32_t> tokens_;
    std::vector<std::uint32_t> block_owners_;
    std::vector<std::uint32_t> victim_order_;
    std::vector<std::uint32_t> victim_plan_;
    std::uint64_t epoch_{0};
    std::size_t free_state_blocks_{0};
};

struct PrefixCacheCreateResult {
    PrefixCacheError error{PrefixCacheError::None};
    std::optional<PrefixCache> cache;

    explicit operator bool() const noexcept;
};

PrefixCacheCreateResult create_prefix_cache(const PrefixCacheConfiguration& configuration);

struct PrefixCacheBudgetCreateResult {
    PrefixCacheError error{PrefixCacheError::None};
    std::optional<PrefixCache> cache;

    explicit operator bool() const noexcept {
        return error == PrefixCacheError::None && cache.has_value();
    }
};

PrefixCacheBudgetCreateResult
create_prefix_cache_for_budget(const PrefixCacheBudgetConfiguration& configuration);

} // namespace tatara::service
