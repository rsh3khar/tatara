#include "tatara/service/prefix_cache.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace tatara::service {
namespace {

constexpr std::uint32_t kNoEntry = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kDigestBatchTokens = 256;

bool checked_multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

std::uint32_t blocks_for(std::uint64_t bytes, std::uint64_t block_bytes) noexcept {
    return static_cast<std::uint32_t>(bytes / block_bytes + (bytes % block_bytes != 0 ? 1 : 0));
}

bool nonzero_digest(const PrefixCacheDigest& digest) noexcept {
    return std::any_of(digest.begin(), digest.end(),
                       [](std::uint8_t value) { return value != 0; });
}

void digest_u64(model::Sha256& digest, std::uint64_t value) noexcept {
    std::array<std::byte, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
    digest.update(bytes);
}

void digest_bytes(model::Sha256& digest, const PrefixCacheDigest& value) noexcept {
    digest.update(std::as_bytes(std::span<const std::uint8_t>(value)));
}

} // namespace

PrefixCacheDigest digest_prefix_tokens(std::span<const std::uint32_t> tokens) noexcept {
    model::Sha256 digest;
    std::array<std::byte, kDigestBatchTokens * sizeof(std::uint32_t)> bytes{};
    std::size_t cursor = 0;
    while (cursor < tokens.size()) {
        const std::size_t count = std::min(kDigestBatchTokens, tokens.size() - cursor);
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t token = tokens[cursor + index];
            bytes[index * 4] = static_cast<std::byte>(token & 0xFFU);
            bytes[index * 4 + 1] = static_cast<std::byte>((token >> 8U) & 0xFFU);
            bytes[index * 4 + 2] = static_cast<std::byte>((token >> 16U) & 0xFFU);
            bytes[index * 4 + 3] = static_cast<std::byte>((token >> 24U) & 0xFFU);
        }
        digest.update(std::span<const std::byte>(bytes.data(), count * sizeof(std::uint32_t)));
        cursor += count;
    }
    return digest.finish();
}

std::string_view prefix_cache_error_name(PrefixCacheError error) noexcept {
    switch (error) {
    case PrefixCacheError::None:
        return "none";
    case PrefixCacheError::InvalidConfiguration:
        return "invalid-configuration";
    case PrefixCacheError::AllocationFailed:
        return "allocation-failed";
    case PrefixCacheError::MetadataBudgetInsufficient:
        return "metadata-budget-insufficient";
    case PrefixCacheError::StateArenaBudgetInsufficient:
        return "state-arena-budget-insufficient";
    case PrefixCacheError::ForeignDomain:
        return "foreign-domain";
    case PrefixCacheError::InvalidOwner:
        return "invalid-owner";
    case PrefixCacheError::EmptyPrefix:
        return "empty-prefix";
    case PrefixCacheError::TokenLimitExceeded:
        return "token-limit-exceeded";
    case PrefixCacheError::BoundaryViolation:
        return "boundary-violation";
    case PrefixCacheError::InvalidMaximumPosition:
        return "invalid-maximum-position";
    case PrefixCacheError::StateTooLarge:
        return "state-too-large";
    case PrefixCacheError::AlreadyPresent:
        return "already-present";
    case PrefixCacheError::PublicationInFlight:
        return "publication-in-flight";
    case PrefixCacheError::NoEvictableEntry:
        return "no-evictable-entry";
    case PrefixCacheError::StateArenaExhausted:
        return "state-arena-exhausted";
    case PrefixCacheError::GenerationExhausted:
        return "generation-exhausted";
    case PrefixCacheError::StaleReservation:
        return "stale-reservation";
    case PrefixCacheError::StaleRestoreLease:
        return "stale-restore-lease";
    case PrefixCacheError::InvalidTransition:
        return "invalid-transition";
    }
    return "unknown";
}

PrefixCacheCreateResult::operator bool() const noexcept {
    return error == PrefixCacheError::None && cache.has_value();
}

PrefixCacheCreateResult create_prefix_cache(const PrefixCacheConfiguration& configuration) {
    if (configuration.state_capacity_bytes == 0 || configuration.state_block_bytes == 0 ||
        configuration.max_entries == 0 || configuration.max_tokens_per_entry == 0 ||
        configuration.boundary_stride_tokens == 0 ||
        configuration.boundary_origin_tokens > configuration.max_tokens_per_entry ||
        configuration.state_capacity_bytes < configuration.state_block_bytes ||
        configuration.domain.generation == kPrefixCacheNoGeneration ||
        !nonzero_digest(configuration.domain.diagnostic_digest) ||
        configuration.generation_limit == kPrefixCacheNoGeneration) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    const std::uint64_t state_blocks =
        configuration.state_capacity_bytes / configuration.state_block_bytes;
    if (state_blocks == 0 || state_blocks > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }

    std::uint64_t token_slots = 0;
    std::uint64_t usable_state_bytes = 0;
    if (!checked_multiply(configuration.max_entries, configuration.max_tokens_per_entry,
                          token_slots) ||
        token_slots > std::numeric_limits<std::size_t>::max() ||
        !checked_multiply(state_blocks, configuration.state_block_bytes, usable_state_bytes)) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }

    try {
        PrefixCache cache(configuration, {}, static_cast<std::size_t>(token_slots),
                          static_cast<std::size_t>(state_blocks));
        std::uint64_t token_bytes = 0;
        std::uint64_t entry_bytes = 0;
        std::uint64_t block_owner_bytes = 0;
        std::uint64_t victim_order_bytes = 0;
        std::uint64_t victim_plan_bytes = 0;
        std::uint64_t metadata_bytes = 0;
        std::uint64_t host_payload_bytes = 0;
        std::uint64_t total_payload_bytes = 0;
        if (!checked_multiply(cache.tokens_.capacity(), sizeof(std::uint32_t), token_bytes) ||
            !checked_multiply(cache.entries_.capacity(), sizeof(PrefixCache::Entry), entry_bytes) ||
            !checked_multiply(cache.block_owners_.capacity(), sizeof(std::uint32_t),
                              block_owner_bytes) ||
            !checked_multiply(cache.victim_order_.capacity(), sizeof(std::uint32_t),
                              victim_order_bytes) ||
            !checked_multiply(cache.victim_plan_.capacity(), sizeof(std::uint32_t),
                              victim_plan_bytes) ||
            !checked_add(entry_bytes, block_owner_bytes, metadata_bytes) ||
            !checked_add(metadata_bytes, victim_order_bytes, metadata_bytes) ||
            !checked_add(metadata_bytes, victim_plan_bytes, metadata_bytes) ||
            !checked_add(metadata_bytes, sizeof(PrefixCache), metadata_bytes) ||
            !checked_add(token_bytes, metadata_bytes, host_payload_bytes) ||
            !checked_add(configuration.state_capacity_bytes, host_payload_bytes,
                         total_payload_bytes)) {
            return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
        }
        cache.budget_ = {
            .configured_total_budget_bytes = 0,
            .state_arena_bytes = configuration.state_capacity_bytes,
            .usable_state_bytes = usable_state_bytes,
            .unused_state_tail_bytes = configuration.state_capacity_bytes - usable_state_bytes,
            .token_arena_bytes = token_bytes,
            .metadata_payload_bytes = metadata_bytes,
            .total_payload_bytes = total_payload_bytes,
            .unallocated_budget_bytes = 0,
        };
        return {.error = PrefixCacheError::None, .cache = std::move(cache)};
    } catch (const std::bad_alloc&) {
        return {.error = PrefixCacheError::AllocationFailed, .cache = std::nullopt};
    } catch (const std::length_error&) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
}

PrefixCacheBudgetCreateResult
create_prefix_cache_for_budget(const PrefixCacheBudgetConfiguration& configuration) {
    if (configuration.total_budget_bytes == 0 || configuration.state_block_bytes == 0 ||
        configuration.minimum_entry_state_bytes == 0 ||
        configuration.max_tokens_per_entry == 0 ||
        configuration.boundary_stride_tokens == 0 ||
        configuration.boundary_origin_tokens > configuration.max_tokens_per_entry ||
        configuration.domain.generation == kPrefixCacheNoGeneration ||
        !nonzero_digest(configuration.domain.diagnostic_digest) ||
        configuration.generation_limit == kPrefixCacheNoGeneration) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    const std::uint64_t minimum_blocks =
        configuration.minimum_entry_state_bytes / configuration.state_block_bytes +
        (configuration.minimum_entry_state_bytes % configuration.state_block_bytes != 0 ? 1 : 0);
    if (minimum_blocks == 0 || minimum_blocks > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    std::uint64_t token_bytes_per_entry = 0;
    std::uint64_t minimum_state_bytes = 0;
    std::uint64_t minimum_block_owner_bytes = 0;
    std::uint64_t per_entry_bytes = 0;
    if (!checked_multiply(configuration.max_tokens_per_entry, sizeof(std::uint32_t),
                          token_bytes_per_entry) ||
        !checked_multiply(minimum_blocks, configuration.state_block_bytes,
                          minimum_state_bytes) ||
        !checked_multiply(minimum_blocks, sizeof(std::uint32_t),
                          minimum_block_owner_bytes) ||
        !checked_add(token_bytes_per_entry, sizeof(PrefixCache::Entry), per_entry_bytes) ||
        !checked_add(per_entry_bytes, 2u * sizeof(std::uint32_t), per_entry_bytes) ||
        !checked_add(per_entry_bytes, minimum_state_bytes, per_entry_bytes) ||
        !checked_add(per_entry_bytes, minimum_block_owner_bytes, per_entry_bytes)) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    const std::uint64_t entry_count =
        configuration.total_budget_bytes / per_entry_bytes;
    if (entry_count == 0) {
        return {.error = PrefixCacheError::MetadataBudgetInsufficient,
                .cache = std::nullopt};
    }
    const std::uint32_t max_entries = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(entry_count, std::numeric_limits<std::uint32_t>::max()));

    std::uint64_t entry_bytes = 0;
    std::uint64_t token_bytes = 0;
    std::uint64_t plan_bytes = 0;
    std::uint64_t fixed_host_bytes = sizeof(PrefixCache);
    if (!checked_multiply(max_entries, sizeof(PrefixCache::Entry), entry_bytes) ||
        !checked_multiply(max_entries, token_bytes_per_entry, token_bytes) ||
        !checked_multiply(max_entries, 2u * sizeof(std::uint32_t), plan_bytes) ||
        !checked_add(fixed_host_bytes, entry_bytes, fixed_host_bytes) ||
        !checked_add(fixed_host_bytes, token_bytes, fixed_host_bytes) ||
        !checked_add(fixed_host_bytes, plan_bytes, fixed_host_bytes) ||
        fixed_host_bytes >= configuration.total_budget_bytes) {
        return {.error = PrefixCacheError::MetadataBudgetInsufficient,
                .cache = std::nullopt};
    }
    const std::uint64_t remaining = configuration.total_budget_bytes - fixed_host_bytes;
    std::uint64_t block_charge = 0;
    if (!checked_add(configuration.state_block_bytes, sizeof(std::uint32_t),
                     block_charge)) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    const std::uint64_t state_blocks = remaining / block_charge;
    if (state_blocks < minimum_blocks ||
        state_blocks > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixCacheError::StateArenaBudgetInsufficient,
                .cache = std::nullopt};
    }
    std::uint64_t state_capacity_bytes = 0;
    if (!checked_multiply(state_blocks, configuration.state_block_bytes,
                          state_capacity_bytes)) {
        return {.error = PrefixCacheError::InvalidConfiguration, .cache = std::nullopt};
    }
    PrefixCacheCreateResult created = create_prefix_cache({
        .state_capacity_bytes = state_capacity_bytes,
        .state_block_bytes = configuration.state_block_bytes,
        .max_entries = max_entries,
        .max_tokens_per_entry = configuration.max_tokens_per_entry,
        .boundary_origin_tokens = configuration.boundary_origin_tokens,
        .boundary_stride_tokens = configuration.boundary_stride_tokens,
        .domain = configuration.domain,
        .generation_limit = configuration.generation_limit,
    });
    if (!created) {
        return {.error = created.error, .cache = std::nullopt};
    }
    PrefixCache cache = std::move(*created.cache);
    if (cache.budget_.total_payload_bytes > configuration.total_budget_bytes) {
        return {.error = PrefixCacheError::MetadataBudgetInsufficient,
                .cache = std::nullopt};
    }
    cache.budget_.configured_total_budget_bytes = configuration.total_budget_bytes;
    cache.budget_.unallocated_budget_bytes =
        configuration.total_budget_bytes - cache.budget_.total_payload_bytes;
    return {.error = PrefixCacheError::None, .cache = std::move(cache)};
}

PrefixCache::PrefixCache(PrefixCacheConfiguration configuration, PrefixCacheBudget budget,
                         std::size_t token_slots, std::size_t state_blocks)
    : configuration_(configuration), budget_(budget), entries_(configuration.max_entries),
      tokens_(token_slots), block_owners_(state_blocks, kNoEntry),
      victim_order_(configuration.max_entries, kNoEntry),
      victim_plan_(configuration.max_entries, kNoEntry), free_state_blocks_(state_blocks) {}

PrefixCacheDomainHandle PrefixCache::domain() const noexcept {
    return {.generation = configuration_.domain.generation};
}

const PrefixCacheDigest& PrefixCache::diagnostic_domain_digest() const noexcept {
    return configuration_.domain.diagnostic_digest;
}

PrefixCacheReserveResult
PrefixCache::reserve_snapshot(PrefixCacheDomainHandle domain, const PrefixCacheKey& key,
                              std::uint64_t state_bytes) noexcept {
    if (!valid_domain(domain)) {
        return {.error = PrefixCacheError::ForeignDomain, .reservation = std::nullopt};
    }
    PrefixCacheError key_error = PrefixCacheError::None;
    if (!valid_key(key, key_error)) {
        return {.error = key_error, .reservation = std::nullopt};
    }
    if (state_bytes == 0 || state_bytes > budget_.usable_state_bytes) {
        return {.error = PrefixCacheError::StateTooLarge, .reservation = std::nullopt};
    }
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
        const Entry& entry = entries_[index];
        if (entry.state != EntryState::Free && entry.state != EntryState::Exhausted &&
            matching_metadata(entry, key) && exact_tokens(index, key.tokens)) {
            const PrefixCacheError error = entry.state == EntryState::Ready
                                               ? PrefixCacheError::AlreadyPresent
                                               : PrefixCacheError::PublicationInFlight;
            return {.error = error, .reservation = std::nullopt};
        }
    }

    const std::uint32_t required_blocks = blocks_for(state_bytes, configuration_.state_block_bytes);
    const std::size_t victim_count = build_lru_victim_order();
    std::size_t plan_count = 0;
    std::uint32_t target = find_usable_free_entry();
    std::uint32_t first_block = kNoEntry;

    for (;;) {
        if (target == kNoEntry) {
            for (std::size_t index = 0; index < plan_count; ++index) {
                const std::uint32_t candidate = victim_plan_[index];
                if (entries_[candidate].generation < configuration_.generation_limit) {
                    target = candidate;
                    break;
                }
            }
        }
        if (target != kNoEntry) {
            first_block = find_virtual_free_extent(required_blocks, plan_count);
            if (first_block != kNoEntry) {
                break;
            }
        }
        if (plan_count == victim_count) {
            if (target == kNoEntry) {
                const bool reusable_but_busy = std::any_of(
                    entries_.begin(), entries_.end(), [&](const Entry& entry) {
                        return entry.state != EntryState::Exhausted &&
                               entry.generation < configuration_.generation_limit;
                    });
                return {
                    .error = reusable_but_busy ? PrefixCacheError::NoEvictableEntry
                                               : PrefixCacheError::GenerationExhausted,
                    .reservation = std::nullopt,
                };
            }
            return {.error = PrefixCacheError::StateArenaExhausted,
                    .reservation = std::nullopt};
        }
        victim_plan_[plan_count] = victim_order_[plan_count];
        ++plan_count;
    }

    for (std::size_t index = 0; index < plan_count; ++index) {
        evict_for_plan(victim_plan_[index]);
    }
    Entry& entry = entries_[target];
    if (entry.state != EntryState::Free ||
        entry.generation >= configuration_.generation_limit) {
        return {.error = PrefixCacheError::GenerationExhausted, .reservation = std::nullopt};
    }

    const std::uint64_t generation = entry.generation + 1;
    entry = {
        .state = EntryState::Reservation,
        .token_digest = key.token_digest,
        .token_count = static_cast<std::uint32_t>(key.tokens.size()),
        .first_block = first_block,
        .block_count = required_blocks,
        .state_bytes = state_bytes,
        .last_use_epoch = 0,
        .generation = generation,
        .request = {},
        .slot = {},
    };
    std::span<std::uint32_t> stored = token_slot(target);
    std::fill(stored.begin(), stored.end(), 0);
    std::copy(key.tokens.begin(), key.tokens.end(), stored.begin());
    assign_blocks(target, first_block, required_blocks);

    const std::uint64_t state_offset =
        static_cast<std::uint64_t>(first_block) * configuration_.state_block_bytes;
    const std::uint64_t allocated_state =
        static_cast<std::uint64_t>(required_blocks) * configuration_.state_block_bytes;
    return {
        .error = PrefixCacheError::None,
        .reservation =
            PrefixCacheReservation{
                .domain_generation = configuration_.domain.generation,
                .entry = target,
                .generation = generation,
                .position_tokens = static_cast<std::uint32_t>(key.tokens.size()),
                .state_offset_bytes = state_offset,
                .state_bytes = state_bytes,
                .allocated_state_bytes = allocated_state,
            },
    };
}

PrefixCacheError PrefixCache::mark_snapshot_pending_publication(
    const PrefixCacheReservation& reservation) noexcept {
    if (!reservation_matches(reservation, EntryState::Reservation)) {
        return PrefixCacheError::StaleReservation;
    }
    entries_[reservation.entry].state = EntryState::PendingPublication;
    return PrefixCacheError::None;
}

PrefixCacheError
PrefixCache::commit_snapshot(const PrefixCacheReservation& reservation) noexcept {
    if (!reservation_matches(reservation, EntryState::PendingPublication)) {
        return PrefixCacheError::StaleReservation;
    }
    entries_[reservation.entry].state = EntryState::Ready;
    touch(reservation.entry);
    return PrefixCacheError::None;
}

PrefixCacheError PrefixCache::abort_snapshot(
    const PrefixCacheReservation& reservation) noexcept {
    if (!reservation_matches(reservation, EntryState::Reservation) &&
        !reservation_matches(reservation, EntryState::PendingPublication)) {
        return PrefixCacheError::StaleReservation;
    }
    release_blocks(entries_[reservation.entry]);
    clear_entry_preserving_generation(reservation.entry);
    return PrefixCacheError::None;
}

PrefixCacheError PrefixCache::retain_snapshot_failure(
    const PrefixCacheReservation& reservation) noexcept {
    if (!reservation_matches(reservation, EntryState::Reservation) &&
        !reservation_matches(reservation, EntryState::PendingPublication)) {
        return PrefixCacheError::StaleReservation;
    }
    retain_entry(reservation.entry);
    return PrefixCacheError::None;
}

PrefixCacheLookupResult
PrefixCache::lookup_longest(PrefixCacheDomainHandle domain,
                            std::span<const std::uint32_t> request_prefill_tokens,
                            std::uint32_t maximum_position, PrefixCacheRequestHandle request,
                            PrefixCacheSlotHandle slot) noexcept {
    if (!valid_domain(domain)) {
        return {.error = PrefixCacheError::ForeignDomain};
    }
    if (request.owner_generation == 0 || slot.slot_generation == 0) {
        return {.error = PrefixCacheError::InvalidOwner};
    }
    if (maximum_position > request_prefill_tokens.size() ||
        maximum_position > configuration_.max_tokens_per_entry) {
        return {.error = PrefixCacheError::InvalidMaximumPosition};
    }

    std::uint32_t selected = kNoEntry;
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
        const Entry& candidate = entries_[index];
        if (candidate.state != EntryState::Ready ||
            candidate.token_count > maximum_position ||
            (selected != kNoEntry &&
             (candidate.token_count < entries_[selected].token_count ||
              (candidate.token_count == entries_[selected].token_count &&
               index > selected)))) {
            continue;
        }
        const auto request_prefix = request_prefill_tokens.first(candidate.token_count);
        if (exact_tokens(index, request_prefix)) {
            selected = index;
        }
    }
    if (selected == kNoEntry) {
        return {.error = PrefixCacheError::None, .hit = false, .lease = std::nullopt};
    }

    Entry& entry = entries_[selected];
    entry.state = EntryState::RestoreLeased;
    entry.request = request;
    entry.slot = slot;
    const std::uint64_t state_offset =
        static_cast<std::uint64_t>(entry.first_block) * configuration_.state_block_bytes;
    const std::uint64_t allocated_state =
        static_cast<std::uint64_t>(entry.block_count) * configuration_.state_block_bytes;
    return {
        .error = PrefixCacheError::None,
        .hit = true,
        .lease =
            PrefixCacheRestoreLease{
                .domain_generation = configuration_.domain.generation,
                .entry = selected,
                .generation = entry.generation,
                .request = request,
                .slot = slot,
                .position_tokens = entry.token_count,
                .state_offset_bytes = state_offset,
                .state_bytes = entry.state_bytes,
                .allocated_state_bytes = allocated_state,
            },
    };
}

PrefixCacheError
PrefixCache::release_restore(const PrefixCacheRestoreLease& lease,
                             PrefixCacheRestoreDisposition disposition) noexcept {
    if (!restore_lease_matches(lease)) {
        return PrefixCacheError::StaleRestoreLease;
    }
    Entry& entry = entries_[lease.entry];
    entry.request = {};
    entry.slot = {};
    switch (disposition) {
    case PrefixCacheRestoreDisposition::Success:
        entry.state = EntryState::Ready;
        touch(lease.entry);
        return PrefixCacheError::None;
    case PrefixCacheRestoreDisposition::ObservedFailure:
        entry.state = EntryState::Ready;
        return PrefixCacheError::None;
    case PrefixCacheRestoreDisposition::RetainedFailure:
        retain_entry(lease.entry);
        return PrefixCacheError::None;
    }
    return PrefixCacheError::InvalidTransition;
}

const PrefixCacheBudget& PrefixCache::budget() const noexcept {
    return budget_;
}

PrefixCacheEvidence PrefixCache::evidence() const noexcept {
    PrefixCacheEvidence result;
    std::size_t owned_blocks = 0;
    bool blocks_valid = true;
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
        const Entry& entry = entries_[index];
        switch (entry.state) {
        case EntryState::Free:
            ++result.free_entries;
            break;
        case EntryState::Reservation:
            ++result.reservation_entries;
            break;
        case EntryState::PendingPublication:
            ++result.pending_publication_entries;
            break;
        case EntryState::Ready:
            ++result.ready_entries;
            break;
        case EntryState::RestoreLeased:
            ++result.restore_leased_entries;
            break;
        case EntryState::FailedRetained:
            ++result.failed_retained_entries;
            break;
        case EntryState::Exhausted:
            ++result.exhausted_entries;
            break;
        }
        const bool owns_blocks =
            entry.state != EntryState::Free && entry.state != EntryState::Exhausted;
        if (!owns_blocks) {
            blocks_valid = blocks_valid && entry.block_count == 0;
            continue;
        }
        owned_blocks += entry.block_count;
        if (entry.first_block > block_owners_.size() ||
            entry.block_count > block_owners_.size() - entry.first_block) {
            blocks_valid = false;
            continue;
        }
        for (std::uint32_t block = 0; block < entry.block_count; ++block) {
            blocks_valid =
                blocks_valid && block_owners_[entry.first_block + block] == index;
        }
    }
    std::size_t counted_free_blocks = 0;
    for (std::uint32_t owner : block_owners_) {
        counted_free_blocks += owner == kNoEntry ? 1U : 0U;
        blocks_valid = blocks_valid && (owner == kNoEntry || owner < entries_.size());
    }

    result.free_state_blocks = free_state_blocks_;
    result.lru_epoch = epoch_;
    result.state_digest = state_digest();
    const std::size_t entry_count =
        result.free_entries + result.reservation_entries +
        result.pending_publication_entries + result.ready_entries +
        result.restore_leased_entries + result.failed_retained_entries +
        result.exhausted_entries;
    result.conserved =
        entry_count == entries_.size() && counted_free_blocks == free_state_blocks_ &&
        owned_blocks + free_state_blocks_ == block_owners_.size() && blocks_valid;
    return result;
}

bool PrefixCache::valid_domain(PrefixCacheDomainHandle domain) const noexcept {
    return domain.generation != kPrefixCacheNoGeneration &&
           domain.generation == configuration_.domain.generation;
}

bool PrefixCache::valid_key(const PrefixCacheKey& key, PrefixCacheError& error) const noexcept {
    if (key.tokens.empty()) {
        error = PrefixCacheError::EmptyPrefix;
        return false;
    }
    if (key.tokens.size() > configuration_.max_tokens_per_entry) {
        error = PrefixCacheError::TokenLimitExceeded;
        return false;
    }
    const std::uint64_t count = key.tokens.size();
    if (count < configuration_.boundary_origin_tokens ||
        (count - configuration_.boundary_origin_tokens) %
                configuration_.boundary_stride_tokens !=
            0) {
        error = PrefixCacheError::BoundaryViolation;
        return false;
    }
    error = PrefixCacheError::None;
    return true;
}

bool PrefixCache::exact_tokens(std::uint32_t entry,
                               std::span<const std::uint32_t> tokens) const noexcept {
    const auto stored = token_slot(entry).first(tokens.size());
    return std::equal(stored.begin(), stored.end(), tokens.begin(), tokens.end());
}

bool PrefixCache::matching_metadata(const Entry& entry, const PrefixCacheKey& key) const noexcept {
    return entry.token_digest == key.token_digest && entry.token_count == key.tokens.size();
}

bool PrefixCache::entry_in_plan(std::uint32_t entry, std::size_t plan_count) const noexcept {
    for (std::size_t index = 0; index < plan_count; ++index) {
        if (victim_plan_[index] == entry) {
            return true;
        }
    }
    return false;
}

std::uint32_t PrefixCache::find_usable_free_entry() const noexcept {
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].state == EntryState::Free &&
            entries_[index].generation < configuration_.generation_limit) {
            return index;
        }
    }
    return kNoEntry;
}

std::size_t PrefixCache::build_lru_victim_order() noexcept {
    std::size_t count = 0;
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].state == EntryState::Ready) {
            victim_order_[count++] = index;
        }
    }
    for (std::size_t index = 1; index < count; ++index) {
        const std::uint32_t value = victim_order_[index];
        std::size_t position = index;
        while (position > 0) {
            const std::uint32_t prior = victim_order_[position - 1];
            if (entries_[prior].last_use_epoch < entries_[value].last_use_epoch ||
                (entries_[prior].last_use_epoch == entries_[value].last_use_epoch &&
                 prior < value)) {
                break;
            }
            victim_order_[position] = prior;
            --position;
        }
        victim_order_[position] = value;
    }
    return count;
}

std::uint32_t PrefixCache::find_virtual_free_extent(std::uint32_t blocks,
                                                    std::size_t plan_count) const noexcept {
    if (blocks == 0 || blocks > block_owners_.size()) {
        return kNoEntry;
    }
    std::uint32_t run_start = 0;
    std::uint32_t run_length = 0;
    for (std::uint32_t block = 0; block < block_owners_.size(); ++block) {
        const std::uint32_t owner = block_owners_[block];
        const bool available = owner == kNoEntry || entry_in_plan(owner, plan_count);
        if (available) {
            if (run_length == 0) {
                run_start = block;
            }
            ++run_length;
            if (run_length == blocks) {
                return run_start;
            }
        } else {
            run_length = 0;
        }
    }
    return kNoEntry;
}

void PrefixCache::evict_for_plan(std::uint32_t entry) noexcept {
    release_blocks(entries_[entry]);
    std::fill(token_slot(entry).begin(), token_slot(entry).end(), 0);
    clear_entry_preserving_generation(entry);
}

void PrefixCache::release_blocks(const Entry& entry) noexcept {
    for (std::uint32_t block = 0; block < entry.block_count; ++block) {
        block_owners_[entry.first_block + block] = kNoEntry;
    }
    free_state_blocks_ += entry.block_count;
}

void PrefixCache::assign_blocks(std::uint32_t entry, std::uint32_t first,
                                std::uint32_t count) noexcept {
    for (std::uint32_t block = 0; block < count; ++block) {
        block_owners_[first + block] = entry;
    }
    free_state_blocks_ -= count;
}

void PrefixCache::touch(std::uint32_t entry) noexcept {
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) {
        rebase_epochs();
    }
    entries_[entry].last_use_epoch = ++epoch_;
}

void PrefixCache::rebase_epochs() noexcept {
    const std::size_t count = build_lru_victim_order();
    for (std::size_t index = 0; index < count; ++index) {
        entries_[victim_order_[index]].last_use_epoch = index + 1;
    }
    epoch_ = count;
}

void PrefixCache::clear_entry_preserving_generation(std::uint32_t entry_index) noexcept {
    const std::uint64_t generation = entries_[entry_index].generation;
    entries_[entry_index] = Entry{};
    entries_[entry_index].generation = generation;
    if (generation >= configuration_.generation_limit) {
        entries_[entry_index].state = EntryState::Exhausted;
    }
}

void PrefixCache::retain_entry(std::uint32_t entry) noexcept {
    entries_[entry].state = EntryState::FailedRetained;
}

std::span<std::uint32_t> PrefixCache::token_slot(std::uint32_t entry) noexcept {
    const std::size_t start = static_cast<std::size_t>(entry) * configuration_.max_tokens_per_entry;
    return std::span<std::uint32_t>(tokens_).subspan(start, configuration_.max_tokens_per_entry);
}

std::span<const std::uint32_t> PrefixCache::token_slot(std::uint32_t entry) const noexcept {
    const std::size_t start = static_cast<std::size_t>(entry) * configuration_.max_tokens_per_entry;
    return std::span<const std::uint32_t>(tokens_).subspan(start,
                                                           configuration_.max_tokens_per_entry);
}

bool PrefixCache::reservation_matches(const PrefixCacheReservation& reservation,
                                      EntryState expected) const noexcept {
    if (reservation.domain_generation != configuration_.domain.generation ||
        reservation.entry >= entries_.size()) {
        return false;
    }
    const Entry& entry = entries_[reservation.entry];
    const std::uint64_t offset =
        static_cast<std::uint64_t>(entry.first_block) * configuration_.state_block_bytes;
    const std::uint64_t allocated =
        static_cast<std::uint64_t>(entry.block_count) * configuration_.state_block_bytes;
    return entry.state == expected && entry.generation == reservation.generation &&
           entry.token_count == reservation.position_tokens &&
           entry.state_bytes == reservation.state_bytes &&
           offset == reservation.state_offset_bytes &&
           allocated == reservation.allocated_state_bytes;
}

bool PrefixCache::restore_lease_matches(const PrefixCacheRestoreLease& lease) const noexcept {
    if (lease.domain_generation != configuration_.domain.generation ||
        lease.entry >= entries_.size() || lease.request.owner_generation == 0 ||
        lease.slot.slot_generation == 0) {
        return false;
    }
    const Entry& entry = entries_[lease.entry];
    const std::uint64_t offset =
        static_cast<std::uint64_t>(entry.first_block) * configuration_.state_block_bytes;
    const std::uint64_t allocated =
        static_cast<std::uint64_t>(entry.block_count) * configuration_.state_block_bytes;
    return entry.state == EntryState::RestoreLeased && entry.generation == lease.generation &&
           entry.request == lease.request && entry.slot == lease.slot &&
           entry.token_count == lease.position_tokens && entry.state_bytes == lease.state_bytes &&
           offset == lease.state_offset_bytes && allocated == lease.allocated_state_bytes;
}

PrefixCacheDigest PrefixCache::state_digest() const noexcept {
    model::Sha256 digest;
    digest_bytes(digest, configuration_.domain.diagnostic_digest);
    digest_u64(digest, configuration_.domain.generation);
    digest_u64(digest, configuration_.state_capacity_bytes);
    digest_u64(digest, configuration_.state_block_bytes);
    digest_u64(digest, configuration_.max_entries);
    digest_u64(digest, configuration_.max_tokens_per_entry);
    digest_u64(digest, configuration_.boundary_origin_tokens);
    digest_u64(digest, configuration_.boundary_stride_tokens);
    digest_u64(digest, configuration_.generation_limit);
    digest_u64(digest, epoch_);
    digest_u64(digest, free_state_blocks_);
    for (const Entry& entry : entries_) {
        digest_u64(digest, static_cast<std::uint8_t>(entry.state));
        digest_bytes(digest, entry.token_digest);
        digest_u64(digest, entry.token_count);
        digest_u64(digest, entry.first_block);
        digest_u64(digest, entry.block_count);
        digest_u64(digest, entry.state_bytes);
        digest_u64(digest, entry.last_use_epoch);
        digest_u64(digest, entry.generation);
        digest_u64(digest, entry.request.owner_index);
        digest_u64(digest, entry.request.owner_generation);
        digest_u64(digest, entry.slot.slot_index);
        digest_u64(digest, entry.slot.slot_generation);
    }
    for (std::uint32_t token : tokens_) {
        digest_u64(digest, token);
    }
    for (std::uint32_t owner : block_owners_) {
        digest_u64(digest, owner);
    }
    return digest.finish();
}

} // namespace tatara::service
