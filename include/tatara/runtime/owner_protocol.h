#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace tatara::runtime {

// These records are caller-backed so their addresses and count are fixed
// before the owner protocol becomes reachable. OwnerProtocol is single-writer
// state-machine machinery: its operations allocate nothing, take no locks and
// perform no system calls.

enum class TopologyStatus : std::uint8_t {
    Ok,
    PhysicalSlotsZero,
    GeneralConnectionsZero,
    OperationalReserveZero,
    AcceptedOverflow,
    PrepublicationBelowGeneral,
    DeliveryBelowAccepted,
    OwnerOverflow,
    ConnectionOverflow,
    AcceptedEquationMismatch,
    OwnerEquationMismatch,
    ConnectionEquationMismatch,
};

struct OwnerTopology {
    std::uint64_t physical_slots{0};             // C
    std::uint64_t scheduler_queue_entries{0};    // Q
    std::uint64_t accepted_credits{0};           // A
    std::uint64_t prepublication_credits{0};     // U
    std::uint64_t delivery_credits{0};           // D
    std::uint64_t request_owners{0};             // O
    std::uint64_t general_connections{0};        // G
    std::uint64_t operational_reserve{0};        // R_op
    std::uint64_t connection_slots{0};           // K
};

struct TopologyResult {
    TopologyStatus status{TopologyStatus::Ok};
    OwnerTopology topology{};

    constexpr explicit operator bool() const noexcept {
        return status == TopologyStatus::Ok;
    }
};

TopologyResult make_owner_topology(std::uint64_t physical_slots,
                                   std::uint64_t scheduler_queue_entries,
                                   std::uint64_t prepublication_credits,
                                   std::uint64_t delivery_credits,
                                   std::uint64_t general_connections,
                                   std::uint64_t operational_reserve) noexcept;
TopologyStatus validate_owner_topology(const OwnerTopology& topology) noexcept;

struct RequestHandle {
    std::uint64_t owner_index{0};
    std::uint64_t owner_generation{0};
    friend constexpr bool operator==(const RequestHandle&, const RequestHandle&) = default;
};

struct SlotHandle {
    std::uint64_t slot_index{0};
    std::uint64_t slot_generation{0};
    friend constexpr bool operator==(const SlotHandle&, const SlotHandle&) = default;
};

struct CacheHandle {
    std::uint64_t entry_index{0};
    std::uint64_t entry_generation{0};
    friend constexpr bool operator==(const CacheHandle&, const CacheHandle&) = default;
};

enum class CommandKind : std::uint8_t {
    Prefill,
    Decode,
    Restore,
    Snapshot,
    Reset,
};

struct CommandTicket {
    std::uint64_t command_generation{0};
    CommandKind kind{CommandKind::Prefill};
    RequestHandle request{};
    SlotHandle slot{};
    friend constexpr bool operator==(const CommandTicket&, const CommandTicket&) = default;
};

struct PrepublicationHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const PrepublicationHandle&,
                                     const PrepublicationHandle&) = default;
};

struct AcceptedHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const AcceptedHandle&, const AcceptedHandle&) = default;
};

struct DeliveryHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const DeliveryHandle&, const DeliveryHandle&) = default;
};

struct ConnectionHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const ConnectionHandle&, const ConnectionHandle&) = default;
};

struct ReserveHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const ReserveHandle&, const ReserveHandle&) = default;
};

struct AdmissionCreditHandle {
    std::uint64_t credit_index{0};
    std::uint64_t credit_generation{0};
    RequestHandle request{};
    friend constexpr bool operator==(const AdmissionCreditHandle&,
                                     const AdmissionCreditHandle&) = default;
};

enum class PrimitiveStatus : std::uint8_t {
    Ok,
    NotInitialized,
    AlreadyInitialized,
    InvalidTopology,
    StorageMismatch,
    GenerationLimitZero,
    MinimumCapacityPolicyInvalid,
    CapacityUnavailable,
    SchedulerCellUnavailable,
    SubmitCellUnavailable,
    GateUnavailable,
    AdmissionClosed,
    PublicationProofInvalid,
    ZeroHandle,
    ForeignHandle,
    StaleHandle,
    RepeatedTransition,
    Exhausted,
    EvidenceInvalid,
    InvalidTransition,
    InvariantViolation,
    Count,
};

enum class OperationPhase : std::uint8_t {
    Topology,
    Initialization,
    Identity,
    Prepublication,
    Publication,
    Submit,
    Queue,
    Slot,
    Delivery,
    Reserve,
    Conservation,
    Exhaustion,
};

enum class OwnerDisposition : std::uint8_t {
    Unchanged,
    Acquired,
    Transferred,
    Released,
    FailedRetained,
    Exhausted,
    AdmissionClosed,
    EvidenceInvalid,
};

enum class NextAction : std::uint8_t {
    Continue,
    RetryWhenCapacityChanges,
    RejectDelayedOperation,
    CloseAdmission,
    RetainAndStop,
    RepairInvariant,
    RecreateProcess,
};

struct OperationResult {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    OperationPhase phase{OperationPhase::Identity};
    PrimitiveStatus internal_cause{PrimitiveStatus::Ok};
    OwnerDisposition disposition{OwnerDisposition::Unchanged};
    std::uint64_t expected{0};
    std::uint64_t actual{0};
    NextAction next_action{NextAction::Continue};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

OperationResult make_operation_result(
    PrimitiveStatus status, OperationPhase phase, std::uint64_t expected = 0,
    std::uint64_t actual = 0,
    OwnerDisposition success_disposition = OwnerDisposition::Unchanged) noexcept;

enum class PrepublicationState : std::uint8_t {
    Free,
    Reading,
    Preparing,
    Prepared,
    Publishing,
    Exhausted,
};

enum class AcceptedState : std::uint8_t {
    Free,
    SubmitPending,
    Queued,
    Running,
    TerminalWaitingEngineDetach,
    FailedRetained,
    Exhausted,
};

enum class DeliveryState : std::uint8_t {
    Free,
    TerminalMailbox,
    Emitting,
    Reclaiming,
    FailedRetained,
    Exhausted,
};

enum class RequestOwnerState : std::uint8_t {
    Free,
    Prepublication,
    Accepted,
    Delivery,
    FailedRetained,
    Exhausted,
};

enum class GeneralConnectionState : std::uint8_t {
    Free,
    Reading,
    Attached,
    Output,
    Closing,
    FailedRetained,
    Exhausted,
};

enum class ReserveConnectionState : std::uint8_t {
    Free,
    Unclassified,
    Operational,
    Refusing,
    Output,
    Closing,
    FailedRetained,
    Exhausted,
};

enum class AdmissionCreditState : std::uint8_t {
    Free,
    Publishing,
    Accepted,
    Released,
    Exhausted,
};

enum class SlotState : std::uint8_t {
    Free,
    RequestOwned,
    TransferPending,
    ResetPending,
    Poisoned,
    FailedRetained,
    Exhausted,
};

enum class SchedulerCellState : std::uint8_t {
    Free,
    Ready,
    Retiring,
    FailedRetained,
    Exhausted,
};

enum class SubmitCellState : std::uint8_t {
    Free,
    Owned,
    Published,
    FailedRetained,
    Exhausted,
};

enum class AdmissionGateState : std::uint8_t {
    Open,
    Publishing,
    ClosedExhaustion,
    ClosedInvariant,
};

enum class OwnerDomain : std::uint8_t {
    Prepublication,
    Accepted,
    Delivery,
    RequestOwner,
    GeneralConnection,
    OperationalReserve,
    SchedulerCell,
    Slot,
    AdmissionCredit,
    SubmitCell,
    AdmissionGate,
    HistoryCounter,
    Count,
};

struct SchedulerCellHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const SchedulerCellHandle&,
                                     const SchedulerCellHandle&) = default;
};

struct SubmitCellHandle {
    std::uint64_t index{0};
    std::uint64_t generation{0};
    friend constexpr bool operator==(const SubmitCellHandle&,
                                     const SubmitCellHandle&) = default;
};

struct AdmissionGateHandle {
    std::uint64_t generation{0};
    RequestHandle request{};
    friend constexpr bool operator==(const AdmissionGateHandle&,
                                     const AdmissionGateHandle&) = default;
};

struct PublicationAuthority {
    RequestHandle request{};
    std::uint64_t immutable_input_generation{0};
    std::uint64_t deadline_proof_generation{0};
    bool immutable_input_ready{false};
    bool deadline_valid{false};
};

struct PrepublicationRecord {
    std::uint64_t generation{0};
    PrepublicationState state{PrepublicationState::Free};
    RequestHandle request{};
};

struct AcceptedRecord {
    std::uint64_t generation{0};
    AcceptedState state{AcceptedState::Free};
    RequestHandle request{};
};

struct DeliveryRecord {
    std::uint64_t generation{0};
    DeliveryState state{DeliveryState::Free};
    RequestHandle request{};
};

struct RequestOwnerRecord {
    std::uint64_t generation{0};
    RequestOwnerState state{RequestOwnerState::Free};
    PrepublicationHandle prepublication{};
    AcceptedHandle accepted{};
    DeliveryHandle delivery{};
    ConnectionHandle connection{};
    AdmissionCreditHandle admission_credit{};
    SlotHandle slot{};
    SchedulerCellHandle scheduler_cell{};
    SubmitCellHandle submit_cell{};
};

struct GeneralConnectionRecord {
    std::uint64_t generation{0};
    GeneralConnectionState state{GeneralConnectionState::Free};
    RequestHandle request{};
};

struct ReserveConnectionRecord {
    std::uint64_t generation{0};
    ReserveConnectionState state{ReserveConnectionState::Free};
};

struct AdmissionCreditRecord {
    std::uint64_t generation{0};
    AdmissionCreditState state{AdmissionCreditState::Free};
    RequestHandle request{};
    AcceptedHandle accepted{};
};

struct SlotRecord {
    std::uint64_t generation{0};
    SlotState state{SlotState::Free};
    RequestHandle request{};
};

struct SchedulerCellRecord {
    std::uint64_t generation{0};
    SchedulerCellState state{SchedulerCellState::Free};
    RequestHandle request{};
};

struct SubmitCellRecord {
    std::uint64_t generation{0};
    SubmitCellState state{SubmitCellState::Free};
    RequestHandle request{};
    std::uint64_t immutable_input_generation{0};
    std::uint64_t deadline_proof_generation{0};
};

struct AdmissionGateRecord {
    std::uint64_t generation{0};
    AdmissionGateState state{AdmissionGateState::Open};
    RequestHandle request{};
    OwnerDomain close_domain{OwnerDomain::Count};
};

struct ExhaustionEvidence {
    std::uint64_t configured_capacity{0};
    std::uint64_t effective_capacity{0};
    std::uint64_t minimum_capacity{0};
    std::uint64_t allowed_retirements{0};
    bool evidence_valid{true};
    bool exhaustion_latched{false};
    bool exhaustion_without_owner{false};
    bool exhaustion_owner_retained{false};
    bool below_minimum{false};
};

struct OwnerHistories {
    std::uint64_t submit_published_total{0};
    std::uint64_t admission_rejected_total{0};
    std::uint64_t accepted_total{0};
    std::uint64_t terminal_selected_total{0};
    std::uint64_t terminal_published_total{0};
    std::uint64_t terminal_consumed_total{0};
    std::uint64_t delivery_consumed_total{0};
};

struct OwnerStorage {
    std::span<PrepublicationRecord> prepublication;
    std::span<AcceptedRecord> accepted;
    std::span<DeliveryRecord> delivery;
    std::span<RequestOwnerRecord> owners;
    std::span<GeneralConnectionRecord> general_connections;
    std::span<ReserveConnectionRecord> operational_reserve;
    std::span<AdmissionCreditRecord> admission_credits;
    std::span<SlotRecord> slots;
    std::span<SchedulerCellRecord> scheduler_cells;
    std::span<SubmitCellRecord> submit_cells;
    std::span<AdmissionGateRecord> admission_gate;
    std::span<ExhaustionEvidence> exhaustion;
    std::span<OwnerHistories> histories;
};

struct RequestLease {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    RequestHandle request{};
    PrepublicationHandle prepublication{};
    ConnectionHandle connection{};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

struct PublicationTicket {
    RequestHandle request{};
    PrepublicationHandle prepublication{};
    AcceptedHandle accepted{};
    AdmissionCreditHandle admission_credit{};
    SubmitCellHandle submit_cell{};
    AdmissionGateHandle gate{};
    PublicationAuthority authority{};
};

struct PublicationResult {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    PublicationTicket ticket{};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

struct SlotResult {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    SlotHandle slot{};
    SchedulerCellHandle retiring_cell{};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

struct DeliveryLease {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    RequestHandle request{};
    DeliveryHandle delivery{};
    AdmissionCreditHandle admission_credit{};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

struct ReserveResult {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    ReserveHandle handle{};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

struct OwnerCounts {
    std::uint64_t prepub_free{0};
    std::uint64_t reading{0};
    std::uint64_t preparing{0};
    std::uint64_t prepared{0};
    std::uint64_t publishing{0};
    std::uint64_t prepub_exhausted{0};

    std::uint64_t accepted_free{0};
    std::uint64_t submit_pending{0};
    std::uint64_t queued{0};
    std::uint64_t running{0};
    std::uint64_t terminal_waiting_engine_detach{0};
    std::uint64_t accepted_failed_retained{0};
    std::uint64_t accepted_exhausted{0};

    std::uint64_t delivery_free{0};
    std::uint64_t terminal_mailbox{0};
    std::uint64_t emitting{0};
    std::uint64_t reclaiming{0};
    std::uint64_t delivery_failed_retained{0};
    std::uint64_t delivery_exhausted{0};

    std::uint64_t owner_free{0};
    std::uint64_t owner_prepublication{0};
    std::uint64_t owner_accepted{0};
    std::uint64_t owner_delivery{0};
    std::uint64_t owner_failed_retained{0};
    std::uint64_t owner_exhausted{0};

    std::uint64_t general_free{0};
    std::uint64_t general_reading{0};
    std::uint64_t general_attached{0};
    std::uint64_t general_output{0};
    std::uint64_t general_closing{0};
    std::uint64_t general_failed_retained{0};
    std::uint64_t general_exhausted{0};

    std::uint64_t reserve_free{0};
    std::uint64_t reserve_unclassified{0};
    std::uint64_t reserve_operational{0};
    std::uint64_t reserve_refusing{0};
    std::uint64_t reserve_output{0};
    std::uint64_t reserve_closing{0};
    std::uint64_t reserve_failed_retained{0};
    std::uint64_t reserve_exhausted{0};

    std::uint64_t slot_free{0};
    std::uint64_t slot_request_owned{0};
    std::uint64_t slot_transfer_pending{0};
    std::uint64_t slot_reset_pending{0};
    std::uint64_t slot_poisoned{0};
    std::uint64_t slot_failed_retained{0};
    std::uint64_t slot_exhausted{0};

    std::uint64_t admission_free{0};
    std::uint64_t admission_publishing{0};
    std::uint64_t admission_accepted{0};
    std::uint64_t admission_released{0};
    std::uint64_t admission_exhausted{0};

    std::uint64_t scheduler_cell_free{0};
    std::uint64_t scheduler_cell_ready{0};
    std::uint64_t scheduler_cell_retiring{0};
    std::uint64_t scheduler_cell_failed_retained{0};
    std::uint64_t scheduler_cell_exhausted{0};

    std::uint64_t submit_cell_free{0};
    std::uint64_t submit_cell_owned{0};
    std::uint64_t submit_cell_published{0};
    std::uint64_t submit_cell_failed_retained{0};
    std::uint64_t submit_cell_exhausted{0};
};

enum class ConservationDomain : std::uint8_t {
    None,
    Topology,
    Storage,
    Prepublication,
    Accepted,
    Delivery,
    Owner,
    GeneralConnection,
    OperationalReserve,
    Slot,
    AdmissionCredit,
    SchedulerCell,
    SubmitCell,
    HistorySubmit,
    HistoryAccepted,
    HistoryTerminalSelected,
    HistoryTerminalPublished,
    Exhaustion,
    CrossReference,
};

struct ConservationResult {
    PrimitiveStatus status{PrimitiveStatus::Ok};
    ConservationDomain domain{ConservationDomain::None};
    std::uint64_t expected{0};
    std::uint64_t actual{0};
    OwnerCounts counts{};
    bool partitions_conserved{false};
    bool qualifying{false};

    constexpr explicit operator bool() const noexcept {
        return status == PrimitiveStatus::Ok;
    }
};

PrimitiveStatus validate_command_ticket(const CommandTicket& observed,
                                        const CommandTicket& expected) noexcept;

class OwnerProtocol {
  public:
    OwnerProtocol(const OwnerTopology& topology, OwnerStorage storage,
                  std::uint64_t generation_limit =
                      std::numeric_limits<std::uint64_t>::max()) noexcept;

    OwnerProtocol(const OwnerProtocol&) = delete;
    OwnerProtocol& operator=(const OwnerProtocol&) = delete;

    PrimitiveStatus initialize() noexcept;
    bool initialized() const noexcept;

    RequestLease acquire_prepublication() noexcept;
    PrimitiveStatus advance_prepublication(RequestHandle request,
                                           PrepublicationState target) noexcept;
    PublicationResult begin_publication(const PublicationAuthority& authority) noexcept;
    PrimitiveStatus commit_publication(const PublicationTicket& ticket) noexcept;
    PrimitiveStatus abort_publication(const PublicationTicket& ticket) noexcept;
    PrimitiveStatus reject_published(RequestHandle request) noexcept;

    PrimitiveStatus queue_accepted(RequestHandle request) noexcept;
    SlotResult bind_slot(RequestHandle request) noexcept;
    PrimitiveStatus retire_scheduler_cell(RequestHandle request) noexcept;
    PrimitiveStatus transition_slot(RequestHandle request, SlotHandle slot,
                                    SlotState target) noexcept;
    PrimitiveStatus release_slot(RequestHandle request, SlotHandle slot) noexcept;
    PrimitiveStatus mark_terminal_waiting(RequestHandle request) noexcept;

    DeliveryLease detach_to_delivery(RequestHandle request) noexcept;
    PrimitiveStatus recycle_admission_credit(AdmissionCreditHandle credit) noexcept;
    PrimitiveStatus advance_delivery(RequestHandle request, DeliveryState target) noexcept;
    PrimitiveStatus finish_delivery(RequestHandle request) noexcept;

    PrimitiveStatus retain_accepted_failure(RequestHandle request,
                                            SlotHandle slot) noexcept;
    PrimitiveStatus retain_delivery_failure(RequestHandle request) noexcept;

    ReserveResult acquire_reserve() noexcept;
    PrimitiveStatus transition_reserve(ReserveHandle handle,
                                       ReserveConnectionState target) noexcept;
    PrimitiveStatus retain_reserve_failure(ReserveHandle handle) noexcept;
    PrimitiveStatus release_reserve(ReserveHandle handle) noexcept;

    PrimitiveStatus validate_request_handle(RequestHandle request) const noexcept;
    PrimitiveStatus validate_slot_handle(SlotHandle slot) const noexcept;
    OperationResult detailed_result(PrimitiveStatus status, OperationPhase phase,
                                    std::uint64_t expected = 0,
                                    std::uint64_t actual = 0,
                                    OwnerDisposition success_disposition =
                                        OwnerDisposition::Unchanged) const noexcept;
    ConservationResult validate_conservation() const noexcept;

  private:
    void latch_exhaustion(OwnerDomain domain, bool owner_retained) noexcept;
    void increment_history(std::uint64_t& counter,
                           bool owner_retained) noexcept;
    bool admission_open() const noexcept;

    OwnerTopology topology_;
    OwnerStorage storage_;
    std::uint64_t generation_limit_;
    bool initialized_{false};
};

} // namespace tatara::runtime
