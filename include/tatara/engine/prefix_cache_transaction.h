#pragma once

#include "tatara/runtime/owner_protocol.h"
#include "tatara/runtime/prefix_state_transfer.h"
#include "tatara/service/prefix_cache.h"

#include <cstdint>
#include <optional>

namespace tatara::engine {

enum class PrefixCacheCommandObservation : std::uint8_t {
    Success,
    ObservedFailure,
    UnobservedFailure,
};

enum class PrefixCacheTerminalDisposition : std::uint8_t {
    SuccessfulStopToken,
    SuccessfulMaximumOutput,
    Cancelled,
    Deadline,
    RequestFailure,
    EngineFailure,
    AdministrativeDrain,
};

enum class PrefixCacheTransactionState : std::uint8_t {
    Encoded,
    PendingPublication,
    Finished,
    FailedRetained,
};

enum class PrefixCacheTransactionError : std::uint8_t {
    None,
    InvalidOwner,
    InvalidCommand,
    BindingMismatch,
    InvalidState,
    TransferFailure,
    CacheFailure,
    OwnershipRetained,
};

struct PrefixCacheTransactionStatus {
    PrefixCacheTransactionError error{PrefixCacheTransactionError::None};
    runtime::PrefixStateTransferError transfer_error{
        runtime::PrefixStateTransferError::None};
    service::PrefixCacheError cache_error{service::PrefixCacheError::None};

    explicit operator bool() const noexcept {
        return error == PrefixCacheTransactionError::None;
    }
};

struct PrefixRestoreTransaction {
    service::PrefixCache* cache_owner{nullptr};
    const runtime::DecodeStep* decode_owner{nullptr};
    runtime::DecodeStateSlot* state_owner{nullptr};
    const backend::metal::MetalBuffer* arena_owner{nullptr};
    runtime::RequestHandle request{};
    runtime::SlotHandle slot{};
    runtime::CommandTicket command{};
    service::PrefixCacheRestoreLease cache_lease{};
    runtime::PrefixStateTransferTicket transfer{};
    PrefixCacheTransactionState state{PrefixCacheTransactionState::Encoded};
};

struct PrefixSnapshotTransaction {
    service::PrefixCache* cache_owner{nullptr};
    const runtime::DecodeStep* decode_owner{nullptr};
    runtime::DecodeStateSlot* state_owner{nullptr};
    const backend::metal::MetalBuffer* arena_owner{nullptr};
    runtime::RequestHandle request{};
    runtime::SlotHandle slot{};
    runtime::CommandTicket command{};
    service::PrefixCacheReservation reservation{};
    runtime::PrefixStateTransferTicket transfer{};
    PrefixCacheTransactionState state{PrefixCacheTransactionState::Encoded};
};

struct PrefixRestoreTransactionResult {
    PrefixCacheTransactionStatus status{};
    std::optional<PrefixRestoreTransaction> transaction;

    explicit operator bool() const noexcept {
        return static_cast<bool>(status) && transaction.has_value();
    }
};

struct PrefixSnapshotTransactionResult {
    PrefixCacheTransactionStatus status{};
    std::optional<PrefixSnapshotTransaction> transaction;

    explicit operator bool() const noexcept {
        return static_cast<bool>(status) && transaction.has_value();
    }
};

PrefixRestoreTransactionResult make_prefix_restore_transaction(
    service::PrefixCache& cache, const runtime::DecodeStep& decode,
    runtime::DecodeStateSlot& state, const backend::metal::MetalBuffer& arena,
    runtime::RequestHandle request, runtime::SlotHandle slot,
    runtime::CommandTicket command, service::PrefixCacheRestoreLease cache_lease,
    runtime::PrefixStateTransferTicket transfer) noexcept;

PrefixSnapshotTransactionResult make_prefix_snapshot_transaction(
    service::PrefixCache& cache, const runtime::DecodeStep& decode,
    runtime::DecodeStateSlot& state, const backend::metal::MetalBuffer& arena,
    runtime::RequestHandle request, runtime::SlotHandle slot,
    runtime::CommandTicket command, service::PrefixCacheReservation reservation,
    runtime::PrefixStateTransferTicket transfer) noexcept;

PrefixCacheTransactionStatus observe_prefix_restore(
    PrefixRestoreTransaction& transaction,
    PrefixCacheCommandObservation observation) noexcept;

PrefixCacheTransactionStatus observe_prefix_snapshot(
    PrefixSnapshotTransaction& transaction,
    PrefixCacheCommandObservation observation) noexcept;

PrefixCacheTransactionStatus resolve_prefix_snapshot_terminal(
    PrefixSnapshotTransaction& transaction,
    PrefixCacheTerminalDisposition terminal) noexcept;

} // namespace tatara::engine
