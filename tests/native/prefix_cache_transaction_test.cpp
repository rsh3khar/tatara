#include "tatara/engine/prefix_cache_transaction.h"

#include <array>
#include <cstdio>
#include <limits>

namespace {

using namespace tatara;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

service::PrefixCacheDigest digest(std::uint8_t value) {
    service::PrefixCacheDigest result{};
    result.fill(value);
    return result;
}

service::PrefixCache make_cache() {
    auto result = service::create_prefix_cache({
        .state_capacity_bytes = 256,
        .state_block_bytes = 64,
        .max_entries = 4,
        .max_tokens_per_entry = 8,
        .boundary_origin_tokens = 1,
        .boundary_stride_tokens = 1,
        .domain = {.diagnostic_digest = digest(1), .generation = 5},
    });
    check(static_cast<bool>(result), "cache constructs");
    return std::move(*result.cache);
}

runtime::RequestHandle request(std::uint64_t index = 1) {
    return {.owner_index = index, .owner_generation = 7};
}

runtime::SlotHandle slot(std::uint64_t index = 2) {
    return {.slot_index = index, .slot_generation = 9};
}

runtime::CommandTicket command(runtime::CommandKind kind,
                               runtime::RequestHandle request_handle = request(),
                               runtime::SlotHandle slot_handle = slot()) {
    return {
        .command_generation = 11,
        .kind = kind,
        .request = request_handle,
        .slot = slot_handle,
    };
}

void set_transfer_state(runtime::DecodeStateSlot& state,
                        runtime::DecodeStateSlotStatus status,
                        const backend::metal::MetalBuffer& arena,
                        std::uint64_t generation, std::uint32_t positions,
                        std::uint64_t offset, std::uint64_t bytes) {
    state.status = status;
    state.active_transfer_generation = generation;
    state.active_transfer_arena = &arena;
    state.active_transfer_positions = positions;
    state.active_transfer_offset_bytes = offset;
    state.active_transfer_state_bytes = bytes;
}

runtime::PrefixStateTransferTicket transfer(
    runtime::PrefixStateTransferDirection direction, runtime::DecodeStep& decode,
    runtime::DecodeStateSlot& state, const backend::metal::MetalBuffer& arena,
    std::uint64_t generation, std::uint32_t positions, std::uint64_t offset,
    std::uint64_t bytes) {
    set_transfer_state(
        state,
        direction == runtime::PrefixStateTransferDirection::Snapshot
            ? runtime::DecodeStateSlotStatus::SnapshotPending
            : runtime::DecodeStateSlotStatus::RestorePending,
        arena, generation, positions, offset, bytes);
    return {
        .direction = direction,
        .owner = &decode,
        .state_owner = &state,
        .arena_owner = &arena,
        .positions = positions,
        .arena_offset_bytes = offset,
        .state_bytes = bytes,
        .generation = generation,
        .pending = true,
    };
}

struct Fixture {
    runtime::DecodeStep decode;
    runtime::DecodeStateSlot state;
    backend::metal::MetalBuffer arena;

    Fixture() {
        decode.capacity = 8;
        decode.schedule = {model::qwen36::LayerKind::GatedDelta};
        state.capacity = decode.capacity;
        state.schedule_identity = decode.schedule.data();
        state.layers.resize(1);
    }
};

service::PrefixCacheKey key(const std::array<std::uint32_t, 2>& tokens) {
    return {
        .token_digest = service::digest_prefix_tokens(tokens),
        .tokens = tokens,
    };
}

service::PrefixCacheReservation reservation(service::PrefixCache& cache,
                                            const std::array<std::uint32_t, 2>& tokens) {
    auto result = cache.reserve_snapshot(cache.domain(), key(tokens), 64);
    check(static_cast<bool>(result), "snapshot reservation succeeds");
    return *result.reservation;
}

service::PrefixCacheRestoreLease restore_lease(
    service::PrefixCache& cache, const std::array<std::uint32_t, 2>& tokens,
    runtime::RequestHandle request_handle = request(),
    runtime::SlotHandle slot_handle = slot()) {
    const service::PrefixCacheReservation prepared = reservation(cache, tokens);
    check(cache.mark_snapshot_pending_publication(prepared) == service::PrefixCacheError::None &&
              cache.commit_snapshot(prepared) == service::PrefixCacheError::None,
          "restore fixture publishes entry");
    auto result = cache.lookup_longest(
        cache.domain(), tokens, 2,
        {.owner_index = request_handle.owner_index,
         .owner_generation = request_handle.owner_generation},
        {.slot_index = slot_handle.slot_index,
         .slot_generation = slot_handle.slot_generation});
    check(static_cast<bool>(result), "restore fixture leases entry");
    return *result.lease;
}

void binding_validation_is_exact_and_nonmutating() {
    service::PrefixCache cache = make_cache();
    Fixture fixture;
    const std::array<std::uint32_t, 2> tokens{1, 2};
    service::PrefixCacheRestoreLease lease = restore_lease(cache, tokens);
    runtime::PrefixStateTransferTicket runtime_ticket = transfer(
        runtime::PrefixStateTransferDirection::Restore, fixture.decode, fixture.state,
        fixture.arena, 1, lease.position_tokens, lease.state_offset_bytes,
        lease.state_bytes);
    const service::PrefixCacheEvidence before = cache.evidence();

    runtime::CommandTicket wrong_kind = command(runtime::CommandKind::Snapshot);
    auto rejected = engine::make_prefix_restore_transaction(
        cache, fixture.decode, fixture.state, fixture.arena, request(), slot(),
        wrong_kind, lease, runtime_ticket);
    check(!rejected &&
              rejected.status.error == engine::PrefixCacheTransactionError::InvalidCommand,
          "restore transaction rejects wrong command kind");

    runtime::PrefixStateTransferTicket wrong_offset = runtime_ticket;
    ++wrong_offset.arena_offset_bytes;
    rejected = engine::make_prefix_restore_transaction(
        cache, fixture.decode, fixture.state, fixture.arena, request(), slot(),
        command(runtime::CommandKind::Restore), lease, wrong_offset);
    check(!rejected &&
              rejected.status.error == engine::PrefixCacheTransactionError::BindingMismatch,
          "restore transaction binds exact arena extent");
    check(before.state_digest == cache.evidence().state_digest,
          "pre-submit binding failures do not mutate cache ownership");

    check(cache.release_restore(lease,
                                service::PrefixCacheRestoreDisposition::ObservedFailure) ==
              service::PrefixCacheError::None,
          "binding fixture lease releases safely");
}

void every_restore_pre_submit_fault_conserves_ownership() {
    service::PrefixCache cache = make_cache();
    Fixture fixture;
    Fixture other;
    const std::array<std::uint32_t, 2> tokens{1, 2};
    const service::PrefixCacheRestoreLease lease =
        restore_lease(cache, tokens);
    const runtime::PrefixStateTransferTicket runtime_ticket = transfer(
        runtime::PrefixStateTransferDirection::Restore, fixture.decode,
        fixture.state, fixture.arena, 12, lease.position_tokens,
        lease.state_offset_bytes, lease.state_bytes);
    const runtime::RequestHandle valid_request = request();
    const runtime::SlotHandle valid_slot = slot();
    const runtime::CommandTicket valid_command =
        command(runtime::CommandKind::Restore);
    const service::PrefixCacheEvidence before = cache.evidence();

    const auto rejected =
        [&](runtime::RequestHandle candidate_request,
            runtime::SlotHandle candidate_slot,
            runtime::CommandTicket candidate_command,
            service::PrefixCacheRestoreLease candidate_lease,
            runtime::PrefixStateTransferTicket candidate_transfer,
            engine::PrefixCacheTransactionError expected,
            const char* what) {
            const auto result = engine::make_prefix_restore_transaction(
                cache, fixture.decode, fixture.state, fixture.arena,
                candidate_request, candidate_slot, candidate_command,
                candidate_lease, candidate_transfer);
            check(!result && result.status.error == expected, what);
            check(cache.evidence().state_digest == before.state_digest,
                  "restore pre-submit fault preserves cache ownership");
        };

    runtime::RequestHandle bad_request = valid_request;
    bad_request.owner_generation = 0;
    rejected(bad_request, valid_slot, valid_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidOwner,
             "zero-generation restore request is rejected");
    runtime::SlotHandle bad_slot = valid_slot;
    bad_slot.slot_generation = 0;
    rejected(valid_request, bad_slot, valid_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidOwner,
             "zero-generation restore slot is rejected");

    runtime::CommandTicket bad_command = valid_command;
    bad_command.command_generation = 0;
    rejected(valid_request, valid_slot, bad_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "zero-generation restore command is rejected");
    bad_command = valid_command;
    bad_command.kind = runtime::CommandKind::Snapshot;
    rejected(valid_request, valid_slot, bad_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-kind restore command is rejected");
    bad_command = valid_command;
    ++bad_command.request.owner_generation;
    rejected(valid_request, valid_slot, bad_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-request restore command is rejected");
    bad_command = valid_command;
    ++bad_command.slot.slot_generation;
    rejected(valid_request, valid_slot, bad_command, lease, runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-slot restore command is rejected");

    runtime::PrefixStateTransferTicket bad_transfer = runtime_ticket;
    bad_transfer.direction =
        runtime::PrefixStateTransferDirection::Snapshot;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "wrong-direction restore transfer is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.owner = &other.decode;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign decode owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.state_owner = &other.state;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign state-slot owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.arena_owner = &other.arena;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign arena owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.generation = 0;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "zero-generation restore transfer is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.pending = false;
    rejected(valid_request, valid_slot, valid_command, lease, bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "completed restore transfer is rejected");

    service::PrefixCacheRestoreLease bad_lease = lease;
    ++bad_lease.domain_generation;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "foreign-domain restore lease is rejected");
    bad_lease = lease;
    ++bad_lease.request.owner_generation;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "foreign-request restore lease is rejected");
    bad_lease = lease;
    ++bad_lease.slot.slot_generation;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "foreign-slot restore lease is rejected");
    bad_lease = lease;
    ++bad_lease.position_tokens;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-position restore lease is rejected");
    bad_lease = lease;
    ++bad_lease.state_offset_bytes;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-offset restore lease is rejected");
    bad_lease = lease;
    ++bad_lease.state_bytes;
    rejected(valid_request, valid_slot, valid_command, bad_lease,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-length restore lease is rejected");

    check(cache.release_restore(
              lease,
              service::PrefixCacheRestoreDisposition::ObservedFailure) ==
              service::PrefixCacheError::None,
          "restore fault matrix releases the unchanged lease");
}

void every_snapshot_pre_submit_fault_conserves_ownership() {
    service::PrefixCache cache = make_cache();
    Fixture fixture;
    Fixture other;
    const std::array<std::uint32_t, 2> tokens{1, 2};
    const service::PrefixCacheReservation reserved =
        reservation(cache, tokens);
    const runtime::PrefixStateTransferTicket runtime_ticket = transfer(
        runtime::PrefixStateTransferDirection::Snapshot, fixture.decode,
        fixture.state, fixture.arena, 13, reserved.position_tokens,
        reserved.state_offset_bytes, reserved.state_bytes);
    const runtime::RequestHandle valid_request = request();
    const runtime::SlotHandle valid_slot = slot();
    const runtime::CommandTicket valid_command =
        command(runtime::CommandKind::Snapshot);
    const service::PrefixCacheEvidence before = cache.evidence();

    const auto rejected =
        [&](runtime::RequestHandle candidate_request,
            runtime::SlotHandle candidate_slot,
            runtime::CommandTicket candidate_command,
            service::PrefixCacheReservation candidate_reservation,
            runtime::PrefixStateTransferTicket candidate_transfer,
            engine::PrefixCacheTransactionError expected,
            const char* what) {
            const auto result = engine::make_prefix_snapshot_transaction(
                cache, fixture.decode, fixture.state, fixture.arena,
                candidate_request, candidate_slot, candidate_command,
                candidate_reservation, candidate_transfer);
            check(!result && result.status.error == expected, what);
            check(cache.evidence().state_digest == before.state_digest,
                  "snapshot pre-submit fault preserves cache ownership");
        };

    runtime::RequestHandle bad_request = valid_request;
    bad_request.owner_generation = 0;
    rejected(bad_request, valid_slot, valid_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidOwner,
             "zero-generation snapshot request is rejected");
    runtime::SlotHandle bad_slot = valid_slot;
    bad_slot.slot_generation = 0;
    rejected(valid_request, bad_slot, valid_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidOwner,
             "zero-generation snapshot slot is rejected");

    runtime::CommandTicket bad_command = valid_command;
    bad_command.command_generation = 0;
    rejected(valid_request, valid_slot, bad_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "zero-generation snapshot command is rejected");
    bad_command = valid_command;
    bad_command.kind = runtime::CommandKind::Restore;
    rejected(valid_request, valid_slot, bad_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-kind snapshot command is rejected");
    bad_command = valid_command;
    ++bad_command.request.owner_generation;
    rejected(valid_request, valid_slot, bad_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-request snapshot command is rejected");
    bad_command = valid_command;
    ++bad_command.slot.slot_generation;
    rejected(valid_request, valid_slot, bad_command, reserved,
             runtime_ticket,
             engine::PrefixCacheTransactionError::InvalidCommand,
             "wrong-slot snapshot command is rejected");

    runtime::PrefixStateTransferTicket bad_transfer = runtime_ticket;
    bad_transfer.direction =
        runtime::PrefixStateTransferDirection::Restore;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "wrong-direction snapshot transfer is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.owner = &other.decode;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign snapshot decode owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.state_owner = &other.state;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign snapshot state owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.arena_owner = &other.arena;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "foreign snapshot arena owner is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.generation = 0;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "zero-generation snapshot transfer is rejected");
    bad_transfer = runtime_ticket;
    bad_transfer.pending = false;
    rejected(valid_request, valid_slot, valid_command, reserved,
             bad_transfer,
             engine::PrefixCacheTransactionError::InvalidState,
             "completed snapshot transfer is rejected");

    service::PrefixCacheReservation bad_reservation = reserved;
    ++bad_reservation.domain_generation;
    rejected(valid_request, valid_slot, valid_command, bad_reservation,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "foreign-domain snapshot reservation is rejected");
    bad_reservation = reserved;
    ++bad_reservation.position_tokens;
    rejected(valid_request, valid_slot, valid_command, bad_reservation,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-position snapshot reservation is rejected");
    bad_reservation = reserved;
    ++bad_reservation.state_offset_bytes;
    rejected(valid_request, valid_slot, valid_command, bad_reservation,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-offset snapshot reservation is rejected");
    bad_reservation = reserved;
    ++bad_reservation.state_bytes;
    rejected(valid_request, valid_slot, valid_command, bad_reservation,
             runtime_ticket,
             engine::PrefixCacheTransactionError::BindingMismatch,
             "wrong-length snapshot reservation is rejected");

    check(cache.abort_snapshot(reserved) ==
              service::PrefixCacheError::None,
          "snapshot fault matrix releases the unchanged reservation");
}

void successful_restore_completes_both_owners_atomically() {
    service::PrefixCache cache = make_cache();
    Fixture fixture;
    const std::array<std::uint32_t, 2> tokens{1, 2};
    service::PrefixCacheRestoreLease lease = restore_lease(cache, tokens);
    runtime::PrefixStateTransferTicket runtime_ticket = transfer(
        runtime::PrefixStateTransferDirection::Restore, fixture.decode, fixture.state,
        fixture.arena, 2, lease.position_tokens, lease.state_offset_bytes,
        lease.state_bytes);
    fixture.state.layers[0].swapped = true;
    auto made = engine::make_prefix_restore_transaction(
        cache, fixture.decode, fixture.state, fixture.arena, request(), slot(),
        command(runtime::CommandKind::Restore), lease, runtime_ticket);
    check(static_cast<bool>(made), "matching restore transaction constructs");
    auto status = engine::observe_prefix_restore(
        *made.transaction, engine::PrefixCacheCommandObservation::Success);
    check(status && made.transaction->state == engine::PrefixCacheTransactionState::Finished,
          "successful restore completes compound transaction");
    check(fixture.state.status == runtime::DecodeStateSlotStatus::Ready &&
              !fixture.state.layers[0].swapped &&
              cache.evidence().ready_entries == 1 &&
              cache.evidence().restore_leased_entries == 0,
          "restore canonicalizes phase and releases cache lease together");
    check(!engine::observe_prefix_restore(
               *made.transaction, engine::PrefixCacheCommandObservation::Success) &&
              made.transaction->state == engine::PrefixCacheTransactionState::Finished,
          "compound restore cannot complete twice");
}

void unobserved_failures_retain_every_native_owner() {
    service::PrefixCache cache = make_cache();
    Fixture fixture;
    const std::array<std::uint32_t, 2> tokens{1, 2};
    service::PrefixCacheRestoreLease lease = restore_lease(cache, tokens);
    runtime::PrefixStateTransferTicket runtime_ticket = transfer(
        runtime::PrefixStateTransferDirection::Restore, fixture.decode, fixture.state,
        fixture.arena, 3, lease.position_tokens, lease.state_offset_bytes,
        lease.state_bytes);
    auto made = engine::make_prefix_restore_transaction(
        cache, fixture.decode, fixture.state, fixture.arena, request(), slot(),
        command(runtime::CommandKind::Restore), lease, runtime_ticket);
    auto status = engine::observe_prefix_restore(
        *made.transaction, engine::PrefixCacheCommandObservation::UnobservedFailure);
    check(status.error == engine::PrefixCacheTransactionError::OwnershipRetained &&
              made.transaction->state ==
                  engine::PrefixCacheTransactionState::FailedRetained,
          "unobserved restore selects failed-retained ownership");
    check(made.transaction->transfer.pending &&
              fixture.state.status == runtime::DecodeStateSlotStatus::RestorePending &&
              cache.evidence().failed_retained_entries == 1 &&
              cache.evidence().free_state_blocks == 3,
          "runtime ticket, slot, cache entry, and arena extent remain retained");
}

void snapshot_publication_obeys_terminal_contract() {
    const std::array<engine::PrefixCacheTerminalDisposition, 7> terminals{
        engine::PrefixCacheTerminalDisposition::SuccessfulStopToken,
        engine::PrefixCacheTerminalDisposition::SuccessfulMaximumOutput,
        engine::PrefixCacheTerminalDisposition::Cancelled,
        engine::PrefixCacheTerminalDisposition::Deadline,
        engine::PrefixCacheTerminalDisposition::RequestFailure,
        engine::PrefixCacheTerminalDisposition::EngineFailure,
        engine::PrefixCacheTerminalDisposition::AdministrativeDrain,
    };
    for (std::size_t index = 0; index < terminals.size(); ++index) {
        service::PrefixCache cache = make_cache();
        Fixture fixture;
        const std::array<std::uint32_t, 2> tokens{1, 2};
        const service::PrefixCacheReservation reserved = reservation(cache, tokens);
        runtime::PrefixStateTransferTicket runtime_ticket = transfer(
            runtime::PrefixStateTransferDirection::Snapshot, fixture.decode,
            fixture.state, fixture.arena, index + 1, reserved.position_tokens,
            reserved.state_offset_bytes, reserved.state_bytes);
        auto made = engine::make_prefix_snapshot_transaction(
            cache, fixture.decode, fixture.state, fixture.arena, request(), slot(),
            command(runtime::CommandKind::Snapshot), reserved, runtime_ticket);
        check(static_cast<bool>(made), "matching snapshot transaction constructs");
        check(static_cast<bool>(engine::observe_prefix_snapshot(
                  *made.transaction, engine::PrefixCacheCommandObservation::Success)),
              "observed snapshot waits pending publication");
        check(cache.evidence().pending_publication_entries == 1 &&
                  cache.evidence().ready_entries == 0,
              "successful copy alone does not publish");
        check(static_cast<bool>(engine::resolve_prefix_snapshot_terminal(
                  *made.transaction, terminals[index])),
              "terminal disposition resolves pending snapshot");
        const bool successful = index < 2;
        check(cache.evidence().ready_entries == (successful ? 1U : 0U) &&
                  cache.evidence().free_entries == (successful ? 3U : 4U),
              successful
                  ? "successful stop/max-output completion publishes"
                  : "cancel/deadline/failure/drain never publishes");
    }
}

} // namespace

int main() {
    binding_validation_is_exact_and_nonmutating();
    every_restore_pre_submit_fault_conserves_ownership();
    every_snapshot_pre_submit_fault_conserves_ownership();
    successful_restore_completes_both_owners_atomically();
    unobserved_failures_retain_every_native_owner();
    snapshot_publication_obeys_terminal_contract();
    if (failures == 0) {
        std::printf("prefix_cache_transaction: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
