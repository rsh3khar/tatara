#include "tatara/engine/prefix_cache_transaction.h"

namespace tatara::engine {
namespace {

using runtime::CommandKind;
using runtime::CommandTicket;
using runtime::DecodeStateSlot;
using runtime::DecodeStep;
using runtime::PrefixStateTransferDirection;
using runtime::PrefixStateTransferError;
using runtime::PrefixStateTransferTicket;
using runtime::RequestHandle;
using runtime::SlotHandle;
using service::PrefixCacheError;
using service::PrefixCacheRequestHandle;
using service::PrefixCacheSlotHandle;

bool valid_request(RequestHandle request) noexcept {
    return request.owner_generation != 0;
}

bool valid_slot(SlotHandle slot) noexcept {
    return slot.slot_generation != 0;
}

PrefixCacheRequestHandle service_request(RequestHandle request) noexcept {
    return {
        .owner_index = request.owner_index,
        .owner_generation = request.owner_generation,
    };
}

PrefixCacheSlotHandle service_slot(SlotHandle slot) noexcept {
    return {
        .slot_index = slot.slot_index,
        .slot_generation = slot.slot_generation,
    };
}

bool command_matches(const CommandTicket& command, CommandKind kind,
                     RequestHandle request, SlotHandle slot) noexcept {
    return command.command_generation != 0 && command.kind == kind &&
           command.request == request && command.slot == slot;
}

bool transfer_matches(const PrefixStateTransferTicket& transfer,
                      PrefixStateTransferDirection direction, const DecodeStep& decode,
                      DecodeStateSlot& state,
                      const backend::metal::MetalBuffer& arena) noexcept {
    return transfer.pending && transfer.direction == direction &&
           transfer.owner == &decode && transfer.state_owner == &state &&
           transfer.arena_owner == &arena && transfer.generation != 0;
}

PrefixCacheTransactionStatus failure(PrefixCacheTransactionError error) noexcept {
    return {.error = error};
}

PrefixCacheTransactionStatus transfer_failure(PrefixStateTransferError error) noexcept {
    return {
        .error = PrefixCacheTransactionError::TransferFailure,
        .transfer_error = error,
    };
}

PrefixCacheTransactionStatus cache_failure(PrefixCacheError error) noexcept {
    return {
        .error = PrefixCacheTransactionError::CacheFailure,
        .cache_error = error,
    };
}

bool successful_terminal(PrefixCacheTerminalDisposition terminal) noexcept {
    return terminal == PrefixCacheTerminalDisposition::SuccessfulStopToken ||
           terminal == PrefixCacheTerminalDisposition::SuccessfulMaximumOutput;
}

bool safe_abort_result(PrefixStateTransferError error) noexcept {
    return error == PrefixStateTransferError::None ||
           error == PrefixStateTransferError::PoisonedStateSlot;
}

} // namespace

PrefixRestoreTransactionResult make_prefix_restore_transaction(
    service::PrefixCache& cache, const DecodeStep& decode, DecodeStateSlot& state,
    const backend::metal::MetalBuffer& arena, RequestHandle request, SlotHandle slot,
    CommandTicket command, service::PrefixCacheRestoreLease cache_lease,
    PrefixStateTransferTicket transfer) noexcept {
    if (!valid_request(request) || !valid_slot(slot)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidOwner)};
    }
    if (!command_matches(command, CommandKind::Restore, request, slot)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidCommand)};
    }
    if (!transfer_matches(transfer, PrefixStateTransferDirection::Restore, decode, state,
                          arena)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidState)};
    }
    if (cache_lease.domain_generation != cache.domain().generation ||
        cache_lease.request != service_request(request) ||
        cache_lease.slot != service_slot(slot) ||
        cache_lease.position_tokens != transfer.positions ||
        cache_lease.state_offset_bytes != transfer.arena_offset_bytes ||
        cache_lease.state_bytes != transfer.state_bytes) {
        return {.status = failure(PrefixCacheTransactionError::BindingMismatch)};
    }
    return {
        .status = {},
        .transaction =
            PrefixRestoreTransaction{
                .cache_owner = &cache,
                .decode_owner = &decode,
                .state_owner = &state,
                .arena_owner = &arena,
                .request = request,
                .slot = slot,
                .command = command,
                .cache_lease = cache_lease,
                .transfer = transfer,
            },
    };
}

PrefixSnapshotTransactionResult make_prefix_snapshot_transaction(
    service::PrefixCache& cache, const DecodeStep& decode, DecodeStateSlot& state,
    const backend::metal::MetalBuffer& arena, RequestHandle request, SlotHandle slot,
    CommandTicket command, service::PrefixCacheReservation reservation,
    PrefixStateTransferTicket transfer) noexcept {
    if (!valid_request(request) || !valid_slot(slot)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidOwner)};
    }
    if (!command_matches(command, CommandKind::Snapshot, request, slot)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidCommand)};
    }
    if (!transfer_matches(transfer, PrefixStateTransferDirection::Snapshot, decode, state,
                          arena)) {
        return {.status = failure(PrefixCacheTransactionError::InvalidState)};
    }
    if (reservation.domain_generation != cache.domain().generation ||
        reservation.position_tokens != transfer.positions ||
        reservation.state_offset_bytes != transfer.arena_offset_bytes ||
        reservation.state_bytes != transfer.state_bytes) {
        return {.status = failure(PrefixCacheTransactionError::BindingMismatch)};
    }
    return {
        .status = {},
        .transaction =
            PrefixSnapshotTransaction{
                .cache_owner = &cache,
                .decode_owner = &decode,
                .state_owner = &state,
                .arena_owner = &arena,
                .request = request,
                .slot = slot,
                .command = command,
                .reservation = reservation,
                .transfer = transfer,
            },
    };
}

PrefixCacheTransactionStatus observe_prefix_restore(
    PrefixRestoreTransaction& transaction,
    PrefixCacheCommandObservation observation) noexcept {
    if (transaction.state != PrefixCacheTransactionState::Encoded ||
        transaction.cache_owner == nullptr || transaction.decode_owner == nullptr ||
        transaction.state_owner == nullptr || transaction.arena_owner == nullptr) {
        return failure(PrefixCacheTransactionError::InvalidState);
    }
    if (observation == PrefixCacheCommandObservation::UnobservedFailure) {
        const PrefixCacheError retained = transaction.cache_owner->release_restore(
            transaction.cache_lease,
            service::PrefixCacheRestoreDisposition::RetainedFailure);
        if (retained != PrefixCacheError::None) {
            return cache_failure(retained);
        }
        transaction.state = PrefixCacheTransactionState::FailedRetained;
        return failure(PrefixCacheTransactionError::OwnershipRetained);
    }

    PrefixStateTransferError transfer_error = PrefixStateTransferError::None;
    service::PrefixCacheRestoreDisposition cache_disposition =
        service::PrefixCacheRestoreDisposition::Success;
    if (observation == PrefixCacheCommandObservation::Success) {
        transfer_error = runtime::complete_prefix_state_transfer(
            transaction.transfer, *transaction.decode_owner, *transaction.state_owner,
            *transaction.arena_owner);
    } else {
        transfer_error = runtime::abort_prefix_state_transfer(
            transaction.transfer, *transaction.decode_owner, *transaction.state_owner,
            *transaction.arena_owner);
        cache_disposition = service::PrefixCacheRestoreDisposition::ObservedFailure;
    }
    if ((observation == PrefixCacheCommandObservation::Success &&
         transfer_error != PrefixStateTransferError::None) ||
        (observation == PrefixCacheCommandObservation::ObservedFailure &&
         !safe_abort_result(transfer_error))) {
        return transfer_failure(transfer_error);
    }
    const PrefixCacheError cache_error =
        transaction.cache_owner->release_restore(transaction.cache_lease, cache_disposition);
    if (cache_error != PrefixCacheError::None) {
        transaction.state = PrefixCacheTransactionState::FailedRetained;
        return cache_failure(cache_error);
    }
    transaction.state = PrefixCacheTransactionState::Finished;
    return {};
}

PrefixCacheTransactionStatus observe_prefix_snapshot(
    PrefixSnapshotTransaction& transaction,
    PrefixCacheCommandObservation observation) noexcept {
    if (transaction.state != PrefixCacheTransactionState::Encoded ||
        transaction.cache_owner == nullptr || transaction.decode_owner == nullptr ||
        transaction.state_owner == nullptr || transaction.arena_owner == nullptr) {
        return failure(PrefixCacheTransactionError::InvalidState);
    }
    if (observation == PrefixCacheCommandObservation::UnobservedFailure) {
        const PrefixCacheError retained =
            transaction.cache_owner->retain_snapshot_failure(transaction.reservation);
        if (retained != PrefixCacheError::None) {
            return cache_failure(retained);
        }
        transaction.state = PrefixCacheTransactionState::FailedRetained;
        return failure(PrefixCacheTransactionError::OwnershipRetained);
    }

    PrefixStateTransferError transfer_error = PrefixStateTransferError::None;
    if (observation == PrefixCacheCommandObservation::Success) {
        transfer_error = runtime::complete_prefix_state_transfer(
            transaction.transfer, *transaction.decode_owner, *transaction.state_owner,
            *transaction.arena_owner);
        if (transfer_error != PrefixStateTransferError::None) {
            return transfer_failure(transfer_error);
        }
        const PrefixCacheError cache_error =
            transaction.cache_owner->mark_snapshot_pending_publication(
                transaction.reservation);
        if (cache_error != PrefixCacheError::None) {
            transaction.state = PrefixCacheTransactionState::FailedRetained;
            return cache_failure(cache_error);
        }
        transaction.state = PrefixCacheTransactionState::PendingPublication;
        return {};
    }

    transfer_error = runtime::abort_prefix_state_transfer(
        transaction.transfer, *transaction.decode_owner, *transaction.state_owner,
        *transaction.arena_owner);
    if (!safe_abort_result(transfer_error)) {
        return transfer_failure(transfer_error);
    }
    const PrefixCacheError cache_error =
        transaction.cache_owner->abort_snapshot(transaction.reservation);
    if (cache_error != PrefixCacheError::None) {
        transaction.state = PrefixCacheTransactionState::FailedRetained;
        return cache_failure(cache_error);
    }
    transaction.state = PrefixCacheTransactionState::Finished;
    return {};
}

PrefixCacheTransactionStatus resolve_prefix_snapshot_terminal(
    PrefixSnapshotTransaction& transaction,
    PrefixCacheTerminalDisposition terminal) noexcept {
    if (transaction.state != PrefixCacheTransactionState::PendingPublication ||
        transaction.cache_owner == nullptr || transaction.transfer.pending) {
        return failure(PrefixCacheTransactionError::InvalidState);
    }
    const PrefixCacheError cache_error =
        successful_terminal(terminal)
            ? transaction.cache_owner->commit_snapshot(transaction.reservation)
            : transaction.cache_owner->abort_snapshot(transaction.reservation);
    if (cache_error != PrefixCacheError::None) {
        transaction.state = PrefixCacheTransactionState::FailedRetained;
        return cache_failure(cache_error);
    }
    transaction.state = PrefixCacheTransactionState::Finished;
    return {};
}

} // namespace tatara::engine
