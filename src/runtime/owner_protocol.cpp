#include "tatara/runtime/owner_protocol.h"

#include <cstddef>
#include <limits>

namespace tatara::runtime {
namespace {

constexpr RequestHandle kNoRequest{};
constexpr PrepublicationHandle kNoPrepublication{};
constexpr AcceptedHandle kNoAccepted{};
constexpr DeliveryHandle kNoDelivery{};
constexpr ConnectionHandle kNoConnection{};
constexpr AdmissionCreditHandle kNoAdmission{};
constexpr SlotHandle kNoSlotHandle{};
constexpr SchedulerCellHandle kNoSchedulerCell{};
constexpr SubmitCellHandle kNoSubmitCell{};

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return false;
    }
    result = left + right;
    return true;
}

bool has_request(RequestHandle handle) noexcept {
    return handle.owner_generation != 0;
}

template <class Record>
bool all_uninitialized(std::span<Record> records) noexcept {
    for (const Record& record : records) {
        if (record.generation != 0) {
            return false;
        }
    }
    return true;
}

template <class Record, class State>
bool release_record(Record& record, State free_state, State exhausted_state,
                    std::uint64_t generation_limit) noexcept {
    record.request = kNoRequest;
    if (record.generation >= generation_limit) {
        record.state = exhausted_state;
        return true;
    } else {
        ++record.generation;
        record.state = free_state;
        return false;
    }
}

bool release_owner(RequestOwnerRecord& record, std::uint64_t generation_limit) noexcept {
    record.prepublication = kNoPrepublication;
    record.accepted = kNoAccepted;
    record.delivery = kNoDelivery;
    record.connection = kNoConnection;
    record.admission_credit = kNoAdmission;
    record.slot = kNoSlotHandle;
    record.scheduler_cell = kNoSchedulerCell;
    record.submit_cell = kNoSubmitCell;
    if (record.generation >= generation_limit) {
        record.state = RequestOwnerState::Exhausted;
        return true;
    } else {
        ++record.generation;
        record.state = RequestOwnerState::Free;
        return false;
    }
}

bool release_reserve_record(ReserveConnectionRecord& record,
                            std::uint64_t generation_limit) noexcept {
    if (record.generation >= generation_limit) {
        record.state = ReserveConnectionState::Exhausted;
        return true;
    } else {
        ++record.generation;
        record.state = ReserveConnectionState::Free;
        return false;
    }
}

bool release_admission(AdmissionCreditRecord& record,
                       std::uint64_t generation_limit) noexcept {
    record.request = kNoRequest;
    record.accepted = kNoAccepted;
    if (record.generation >= generation_limit) {
        record.state = AdmissionCreditState::Exhausted;
        return true;
    } else {
        ++record.generation;
        record.state = AdmissionCreditState::Free;
        return false;
    }
}

bool release_submit(SubmitCellRecord& record,
                    std::uint64_t generation_limit) noexcept {
    record.request = kNoRequest;
    record.immutable_input_generation = 0;
    record.deadline_proof_generation = 0;
    if (record.generation >= generation_limit) {
        record.state = SubmitCellState::Exhausted;
        return true;
    }
    ++record.generation;
    record.state = SubmitCellState::Free;
    return false;
}

template <class Record, class State>
PrimitiveStatus validate_generation(std::span<const Record> records, std::uint64_t index,
                                    std::uint64_t generation, State exhausted) noexcept {
    if (generation == 0) {
        return PrimitiveStatus::ZeroHandle;
    }
    if (index >= records.size()) {
        return PrimitiveStatus::ForeignHandle;
    }
    const Record& record = records[static_cast<std::size_t>(index)];
    if (record.generation != generation) {
        return PrimitiveStatus::StaleHandle;
    }
    if (record.state == exhausted) {
        return PrimitiveStatus::Exhausted;
    }
    return PrimitiveStatus::Ok;
}

template <class Record, class State>
std::size_t first_state(std::span<Record> records, State state) noexcept {
    for (std::size_t index = 0; index < records.size(); ++index) {
        if (records[index].state == state) {
            return index;
        }
    }
    return records.size();
}

template <class Record, class State>
PrimitiveStatus unavailable_status(std::span<Record> records, State exhausted) noexcept {
    for (const Record& record : records) {
        if (record.state != exhausted) {
            return PrimitiveStatus::CapacityUnavailable;
        }
    }
    return PrimitiveStatus::Exhausted;
}

bool zero_bindings(const RequestOwnerRecord& record) noexcept {
    return record.prepublication == kNoPrepublication && record.accepted == kNoAccepted &&
           record.delivery == kNoDelivery && record.connection == kNoConnection &&
           record.admission_credit == kNoAdmission &&
           record.slot == kNoSlotHandle &&
           record.scheduler_cell == kNoSchedulerCell &&
           record.submit_cell == kNoSubmitCell;
}

} // namespace

OperationResult make_operation_result(
    PrimitiveStatus status, OperationPhase phase, std::uint64_t expected,
    std::uint64_t actual, OwnerDisposition success_disposition) noexcept {
    OperationResult result{status, phase, status, OwnerDisposition::Unchanged,
                           expected, actual, NextAction::Continue};
    switch (status) {
    case PrimitiveStatus::Ok:
        result.internal_cause = PrimitiveStatus::Ok;
        result.disposition = success_disposition;
        result.next_action = NextAction::Continue;
        break;
    case PrimitiveStatus::CapacityUnavailable:
    case PrimitiveStatus::SchedulerCellUnavailable:
    case PrimitiveStatus::SubmitCellUnavailable:
    case PrimitiveStatus::GateUnavailable:
        result.next_action = NextAction::RetryWhenCapacityChanges;
        break;
    case PrimitiveStatus::ZeroHandle:
    case PrimitiveStatus::ForeignHandle:
    case PrimitiveStatus::StaleHandle:
    case PrimitiveStatus::RepeatedTransition:
        result.next_action = NextAction::RejectDelayedOperation;
        break;
    case PrimitiveStatus::AdmissionClosed:
    case PrimitiveStatus::Exhausted:
        result.disposition = OwnerDisposition::AdmissionClosed;
        result.next_action = NextAction::CloseAdmission;
        break;
    case PrimitiveStatus::EvidenceInvalid:
        result.disposition = OwnerDisposition::EvidenceInvalid;
        result.next_action = NextAction::CloseAdmission;
        break;
    case PrimitiveStatus::InvariantViolation:
        result.disposition = OwnerDisposition::FailedRetained;
        result.next_action = NextAction::RetainAndStop;
        break;
    case PrimitiveStatus::AlreadyInitialized:
        result.next_action = NextAction::RecreateProcess;
        break;
    case PrimitiveStatus::NotInitialized:
    case PrimitiveStatus::InvalidTopology:
    case PrimitiveStatus::StorageMismatch:
    case PrimitiveStatus::GenerationLimitZero:
    case PrimitiveStatus::MinimumCapacityPolicyInvalid:
    case PrimitiveStatus::PublicationProofInvalid:
    case PrimitiveStatus::InvalidTransition:
        result.next_action = NextAction::RepairInvariant;
        break;
    case PrimitiveStatus::Count:
        result.status = PrimitiveStatus::InvariantViolation;
        result.internal_cause = PrimitiveStatus::InvariantViolation;
        result.next_action = NextAction::RepairInvariant;
        break;
    }
    return result;
}

TopologyResult make_owner_topology(std::uint64_t physical_slots,
                                   std::uint64_t scheduler_queue_entries,
                                   std::uint64_t prepublication_credits,
                                   std::uint64_t delivery_credits,
                                   std::uint64_t general_connections,
                                   std::uint64_t operational_reserve) noexcept {
    TopologyResult result;
    result.topology.physical_slots = physical_slots;
    result.topology.scheduler_queue_entries = scheduler_queue_entries;
    result.topology.prepublication_credits = prepublication_credits;
    result.topology.delivery_credits = delivery_credits;
    result.topology.general_connections = general_connections;
    result.topology.operational_reserve = operational_reserve;

    if (physical_slots == 0) {
        result.status = TopologyStatus::PhysicalSlotsZero;
        return result;
    }
    if (general_connections == 0) {
        result.status = TopologyStatus::GeneralConnectionsZero;
        return result;
    }
    if (operational_reserve == 0) {
        result.status = TopologyStatus::OperationalReserveZero;
        return result;
    }
    if (!checked_add(physical_slots, scheduler_queue_entries,
                     result.topology.accepted_credits)) {
        result.status = TopologyStatus::AcceptedOverflow;
        return result;
    }
    if (prepublication_credits < general_connections) {
        result.status = TopologyStatus::PrepublicationBelowGeneral;
        return result;
    }
    if (delivery_credits < result.topology.accepted_credits) {
        result.status = TopologyStatus::DeliveryBelowAccepted;
        return result;
    }
    if (!checked_add(general_connections, operational_reserve,
                     result.topology.connection_slots)) {
        result.status = TopologyStatus::ConnectionOverflow;
        return result;
    }
    std::uint64_t accepted_plus_prepublication = 0;
    if (!checked_add(result.topology.accepted_credits, prepublication_credits,
                     accepted_plus_prepublication) ||
        !checked_add(accepted_plus_prepublication, delivery_credits,
                     result.topology.request_owners)) {
        result.status = TopologyStatus::OwnerOverflow;
        return result;
    }
    return result;
}

TopologyStatus validate_owner_topology(const OwnerTopology& topology) noexcept {
    const TopologyResult derived =
        make_owner_topology(topology.physical_slots, topology.scheduler_queue_entries,
                            topology.prepublication_credits, topology.delivery_credits,
                            topology.general_connections, topology.operational_reserve);
    if (!derived) {
        return derived.status;
    }
    if (topology.accepted_credits != derived.topology.accepted_credits) {
        return TopologyStatus::AcceptedEquationMismatch;
    }
    if (topology.request_owners != derived.topology.request_owners) {
        return TopologyStatus::OwnerEquationMismatch;
    }
    if (topology.connection_slots != derived.topology.connection_slots) {
        return TopologyStatus::ConnectionEquationMismatch;
    }
    return TopologyStatus::Ok;
}

PrimitiveStatus validate_command_ticket(const CommandTicket& observed,
                                        const CommandTicket& expected) noexcept {
    if (observed.command_generation == 0 || observed.request.owner_generation == 0 ||
        observed.slot.slot_generation == 0) {
        return PrimitiveStatus::ZeroHandle;
    }
    if (observed.command_generation != expected.command_generation ||
        observed.request.owner_generation != expected.request.owner_generation ||
        observed.slot.slot_generation != expected.slot.slot_generation) {
        return PrimitiveStatus::StaleHandle;
    }
    if (observed.kind != expected.kind ||
        observed.request.owner_index != expected.request.owner_index ||
        observed.slot.slot_index != expected.slot.slot_index) {
        return PrimitiveStatus::ForeignHandle;
    }
    return PrimitiveStatus::Ok;
}

OwnerProtocol::OwnerProtocol(const OwnerTopology& topology, OwnerStorage storage,
                             std::uint64_t generation_limit) noexcept
    : topology_(topology), storage_(storage), generation_limit_(generation_limit) {}

bool OwnerProtocol::initialized() const noexcept {
    return initialized_;
}

PrimitiveStatus OwnerProtocol::initialize() noexcept {
    if (initialized_) {
        return PrimitiveStatus::AlreadyInitialized;
    }
    if (generation_limit_ == 0) {
        return PrimitiveStatus::GenerationLimitZero;
    }
    if (validate_owner_topology(topology_) != TopologyStatus::Ok) {
        return PrimitiveStatus::InvalidTopology;
    }
    if (storage_.prepublication.size() != topology_.prepublication_credits ||
        storage_.accepted.size() != topology_.accepted_credits ||
        storage_.delivery.size() != topology_.delivery_credits ||
        storage_.owners.size() != topology_.request_owners ||
        storage_.general_connections.size() != topology_.general_connections ||
        storage_.operational_reserve.size() != topology_.operational_reserve ||
        storage_.admission_credits.size() != topology_.accepted_credits ||
        storage_.slots.size() != topology_.physical_slots ||
        storage_.scheduler_cells.size() != topology_.scheduler_queue_entries ||
        storage_.submit_cells.size() != topology_.accepted_credits ||
        storage_.admission_gate.size() != 1 ||
        storage_.exhaustion.size() !=
            static_cast<std::size_t>(OwnerDomain::Count) ||
        storage_.histories.size() != 1) {
        return PrimitiveStatus::StorageMismatch;
    }
    if (!all_uninitialized(storage_.prepublication) ||
        !all_uninitialized(storage_.accepted) || !all_uninitialized(storage_.delivery) ||
        !all_uninitialized(storage_.owners) ||
        !all_uninitialized(storage_.general_connections) ||
        !all_uninitialized(storage_.operational_reserve) ||
        !all_uninitialized(storage_.admission_credits) ||
        !all_uninitialized(storage_.slots) ||
        !all_uninitialized(storage_.scheduler_cells) ||
        !all_uninitialized(storage_.submit_cells) ||
        !all_uninitialized(storage_.admission_gate)) {
        return PrimitiveStatus::AlreadyInitialized;
    }

    const std::uint64_t configured[] = {
        topology_.prepublication_credits,
        topology_.accepted_credits,
        topology_.delivery_credits,
        topology_.request_owners,
        topology_.general_connections,
        topology_.operational_reserve,
        topology_.scheduler_queue_entries,
        topology_.physical_slots,
        topology_.accepted_credits,
        topology_.accepted_credits,
        1,
        std::numeric_limits<std::uint64_t>::max(),
    };
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(OwnerDomain::Count); ++index) {
        const ExhaustionEvidence& policy = storage_.exhaustion[index];
        if (policy.allowed_retirements > configured[index] ||
            policy.minimum_capacity >
                configured[index] - policy.allowed_retirements) {
            return PrimitiveStatus::MinimumCapacityPolicyInvalid;
        }
    }

    for (PrepublicationRecord& record : storage_.prepublication) {
        record = {1, PrepublicationState::Free, {}};
    }
    for (AcceptedRecord& record : storage_.accepted) {
        record = {1, AcceptedState::Free, {}};
    }
    for (DeliveryRecord& record : storage_.delivery) {
        record = {1, DeliveryState::Free, {}};
    }
    for (RequestOwnerRecord& record : storage_.owners) {
        record = {1, RequestOwnerState::Free, {}, {}, {}, {}, {}};
    }
    for (GeneralConnectionRecord& record : storage_.general_connections) {
        record = {1, GeneralConnectionState::Free, {}};
    }
    for (ReserveConnectionRecord& record : storage_.operational_reserve) {
        record = {1, ReserveConnectionState::Free};
    }
    for (AdmissionCreditRecord& record : storage_.admission_credits) {
        record = {1, AdmissionCreditState::Free, {}, {}};
    }
    for (SlotRecord& record : storage_.slots) {
        record = {1, SlotState::Free, {}};
    }
    for (SchedulerCellRecord& record : storage_.scheduler_cells) {
        record = {1, SchedulerCellState::Free, {}};
    }
    for (SubmitCellRecord& record : storage_.submit_cells) {
        record = {1, SubmitCellState::Free, {}, 0, 0};
    }
    storage_.admission_gate[0] =
        {1, AdmissionGateState::Open, {}, OwnerDomain::Count};
    storage_.histories[0] = {};
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(OwnerDomain::Count); ++index) {
        const std::uint64_t minimum = storage_.exhaustion[index].minimum_capacity;
        const std::uint64_t allowed =
            storage_.exhaustion[index].allowed_retirements;
        storage_.exhaustion[index] =
            {configured[index], configured[index], minimum, allowed,
             true, false, false, false, false};
    }
    initialized_ = true;
    return PrimitiveStatus::Ok;
}

bool OwnerProtocol::admission_open() const noexcept {
    return initialized_ && storage_.admission_gate.size() == 1 &&
           storage_.admission_gate[0].state == AdmissionGateState::Open;
}

void OwnerProtocol::latch_exhaustion(OwnerDomain domain,
                                     bool owner_retained) noexcept {
    const std::size_t index = static_cast<std::size_t>(domain);
    if (index >= storage_.exhaustion.size()) {
        if (storage_.admission_gate.size() == 1) {
            storage_.admission_gate[0].state =
                AdmissionGateState::ClosedInvariant;
            storage_.admission_gate[0].request = {};
            storage_.admission_gate[0].close_domain = OwnerDomain::Count;
        }
        return;
    }
    ExhaustionEvidence& evidence = storage_.exhaustion[index];
    const bool first_latch = !evidence.exhaustion_latched;
    evidence.evidence_valid = false;
    evidence.exhaustion_latched = true;
    if (first_latch) {
        evidence.exhaustion_owner_retained = owner_retained;
        evidence.exhaustion_without_owner = !owner_retained;
    }
    if (evidence.effective_capacity != 0) {
        --evidence.effective_capacity;
    }
    evidence.below_minimum =
        evidence.effective_capacity < evidence.minimum_capacity;
    AdmissionGateRecord& gate = storage_.admission_gate[0];
    gate.state = AdmissionGateState::ClosedExhaustion;
    gate.request = {};
    gate.close_domain = domain;
}

void OwnerProtocol::increment_history(std::uint64_t& counter,
                                      bool owner_retained) noexcept {
    if (counter == std::numeric_limits<std::uint64_t>::max()) {
        latch_exhaustion(OwnerDomain::HistoryCounter, owner_retained);
        return;
    }
    ++counter;
}

OperationResult OwnerProtocol::detailed_result(
    PrimitiveStatus status, OperationPhase phase, std::uint64_t expected,
    std::uint64_t actual, OwnerDisposition success_disposition) const noexcept {
    return make_operation_result(status, phase, expected, actual,
                                 success_disposition);
}

PrimitiveStatus OwnerProtocol::validate_request_handle(RequestHandle request) const noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status =
        validate_generation<RequestOwnerRecord, RequestOwnerState>(
        storage_.owners, request.owner_index, request.owner_generation,
        RequestOwnerState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    return storage_.owners[request.owner_index].state == RequestOwnerState::Free
               ? PrimitiveStatus::RepeatedTransition
               : PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::validate_slot_handle(SlotHandle slot) const noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status = validate_generation<SlotRecord, SlotState>(
        storage_.slots, slot.slot_index, slot.slot_generation, SlotState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    return storage_.slots[slot.slot_index].state == SlotState::Free
               ? PrimitiveStatus::RepeatedTransition
               : PrimitiveStatus::Ok;
}

RequestLease OwnerProtocol::acquire_prepublication() noexcept {
    if (!initialized_) {
        return {PrimitiveStatus::NotInitialized};
    }
    if (!admission_open()) {
        return {PrimitiveStatus::AdmissionClosed};
    }
    const std::size_t owner_index =
        first_state(storage_.owners, RequestOwnerState::Free);
    if (owner_index == storage_.owners.size()) {
        return {unavailable_status(storage_.owners, RequestOwnerState::Exhausted)};
    }
    const std::size_t prepublication_index =
        first_state(storage_.prepublication, PrepublicationState::Free);
    if (prepublication_index == storage_.prepublication.size()) {
        return {
            unavailable_status(storage_.prepublication, PrepublicationState::Exhausted)};
    }
    const std::size_t connection_index =
        first_state(storage_.general_connections, GeneralConnectionState::Free);
    if (connection_index == storage_.general_connections.size()) {
        return {unavailable_status(storage_.general_connections,
                                   GeneralConnectionState::Exhausted)};
    }

    RequestOwnerRecord& owner = storage_.owners[owner_index];
    PrepublicationRecord& prepublication = storage_.prepublication[prepublication_index];
    GeneralConnectionRecord& connection =
        storage_.general_connections[connection_index];
    const RequestHandle request{owner_index, owner.generation};
    const PrepublicationHandle prepublication_handle{prepublication_index,
                                                     prepublication.generation};
    const ConnectionHandle connection_handle{connection_index, connection.generation};

    owner.state = RequestOwnerState::Prepublication;
    owner.prepublication = prepublication_handle;
    owner.connection = connection_handle;
    prepublication.state = PrepublicationState::Reading;
    prepublication.request = request;
    connection.state = GeneralConnectionState::Reading;
    connection.request = request;
    return {PrimitiveStatus::Ok, request, prepublication_handle, connection_handle};
}

PrimitiveStatus OwnerProtocol::advance_prepublication(RequestHandle request,
                                                      PrepublicationState target) noexcept {
    const PrimitiveStatus handle_status = validate_request_handle(request);
    if (handle_status != PrimitiveStatus::Ok) {
        return handle_status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Prepublication) {
        return owner.state == RequestOwnerState::Accepted ||
                       owner.state == RequestOwnerState::Delivery
                   ? PrimitiveStatus::RepeatedTransition
                   : PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus prepublication_status =
        validate_generation<PrepublicationRecord, PrepublicationState>(
            storage_.prepublication, owner.prepublication.index,
            owner.prepublication.generation, PrepublicationState::Exhausted);
    if (prepublication_status != PrimitiveStatus::Ok) {
        return prepublication_status;
    }
    PrepublicationRecord& record = storage_.prepublication[owner.prepublication.index];
    if (record.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (record.state == target) {
        return PrimitiveStatus::RepeatedTransition;
    }
    const bool allowed =
        (record.state == PrepublicationState::Reading &&
         target == PrepublicationState::Preparing) ||
        (record.state == PrepublicationState::Preparing &&
         target == PrepublicationState::Prepared);
    if (!allowed) {
        return PrimitiveStatus::InvalidTransition;
    }
    record.state = target;
    return PrimitiveStatus::Ok;
}

PublicationResult
OwnerProtocol::begin_publication(const PublicationAuthority& authority) noexcept {
    const RequestHandle request = authority.request;
    if (authority.immutable_input_generation == 0 ||
        authority.deadline_proof_generation == 0 ||
        !authority.immutable_input_ready || !authority.deadline_valid) {
        return {PrimitiveStatus::PublicationProofInvalid};
    }
    const PrimitiveStatus handle_status = validate_request_handle(request);
    if (handle_status != PrimitiveStatus::Ok) {
        return {handle_status};
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Prepublication) {
        return {PrimitiveStatus::RepeatedTransition};
    }
    const PrimitiveStatus prepublication_status =
        validate_generation<PrepublicationRecord, PrepublicationState>(
            storage_.prepublication, owner.prepublication.index,
            owner.prepublication.generation, PrepublicationState::Exhausted);
    if (prepublication_status != PrimitiveStatus::Ok) {
        return {prepublication_status};
    }
    PrepublicationRecord& prepublication =
        storage_.prepublication[owner.prepublication.index];
    if (prepublication.request != request) {
        return {PrimitiveStatus::ForeignHandle};
    }
    if (prepublication.state == PrepublicationState::Publishing) {
        return {PrimitiveStatus::RepeatedTransition};
    }
    if (prepublication.state != PrepublicationState::Prepared) {
        return {PrimitiveStatus::InvalidTransition};
    }
    AdmissionGateRecord& gate = storage_.admission_gate[0];
    if (gate.state == AdmissionGateState::ClosedExhaustion ||
        gate.state == AdmissionGateState::ClosedInvariant) {
        return {PrimitiveStatus::AdmissionClosed};
    }
    if (gate.state != AdmissionGateState::Open) {
        return {PrimitiveStatus::GateUnavailable};
    }

    std::size_t index = storage_.accepted.size();
    for (std::size_t candidate = 0; candidate < storage_.accepted.size(); ++candidate) {
        if (storage_.accepted[candidate].state == AcceptedState::Free &&
            storage_.admission_credits[candidate].state == AdmissionCreditState::Free) {
            index = candidate;
            break;
        }
    }
    if (index == storage_.accepted.size()) {
        return {PrimitiveStatus::CapacityUnavailable};
    }
    const std::size_t submit_index =
        first_state(storage_.submit_cells, SubmitCellState::Free);
    if (submit_index == storage_.submit_cells.size()) {
        return {unavailable_status(storage_.submit_cells,
                                   SubmitCellState::Exhausted) ==
                        PrimitiveStatus::Exhausted
                    ? PrimitiveStatus::Exhausted
                    : PrimitiveStatus::SubmitCellUnavailable};
    }

    AcceptedRecord& accepted = storage_.accepted[index];
    AdmissionCreditRecord& credit = storage_.admission_credits[index];
    SubmitCellRecord& submit = storage_.submit_cells[submit_index];
    const AcceptedHandle accepted_handle{index, accepted.generation};
    const AdmissionCreditHandle credit_handle{index, credit.generation, request};
    const SubmitCellHandle submit_handle{submit_index, submit.generation};
    const AdmissionGateHandle gate_handle{gate.generation, request};
    prepublication.state = PrepublicationState::Publishing;
    credit.state = AdmissionCreditState::Publishing;
    credit.request = request;
    credit.accepted = accepted_handle;
    submit.state = SubmitCellState::Owned;
    submit.request = request;
    submit.immutable_input_generation = authority.immutable_input_generation;
    submit.deadline_proof_generation = authority.deadline_proof_generation;
    gate.state = AdmissionGateState::Publishing;
    gate.request = request;
    owner.admission_credit = credit_handle;
    owner.submit_cell = submit_handle;
    return {PrimitiveStatus::Ok,
            {request, owner.prepublication, accepted_handle, credit_handle,
             submit_handle, gate_handle, authority}};
}

PrimitiveStatus OwnerProtocol::commit_publication(const PublicationTicket& ticket) noexcept {
    const PrimitiveStatus request_status = validate_request_handle(ticket.request);
    if (request_status != PrimitiveStatus::Ok) {
        return request_status;
    }
    RequestOwnerRecord& owner = storage_.owners[ticket.request.owner_index];
    if (owner.state != RequestOwnerState::Prepublication) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (owner.prepublication != ticket.prepublication) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (owner.admission_credit != ticket.admission_credit ||
        owner.submit_cell != ticket.submit_cell) {
        return PrimitiveStatus::ForeignHandle;
    }
    const PrimitiveStatus prepublication_status =
        validate_generation<PrepublicationRecord, PrepublicationState>(
            storage_.prepublication, ticket.prepublication.index,
            ticket.prepublication.generation, PrepublicationState::Exhausted);
    if (prepublication_status != PrimitiveStatus::Ok) {
        return prepublication_status;
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, ticket.accepted.index, ticket.accepted.generation,
            AcceptedState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return accepted_status;
    }
    const PrimitiveStatus credit_status =
        validate_generation<AdmissionCreditRecord, AdmissionCreditState>(
            storage_.admission_credits, ticket.admission_credit.credit_index,
            ticket.admission_credit.credit_generation, AdmissionCreditState::Exhausted);
    if (credit_status != PrimitiveStatus::Ok) {
        return credit_status;
    }
    const PrimitiveStatus submit_status =
        validate_generation<SubmitCellRecord, SubmitCellState>(
            storage_.submit_cells, ticket.submit_cell.index,
            ticket.submit_cell.generation, SubmitCellState::Exhausted);
    if (submit_status != PrimitiveStatus::Ok) {
        return submit_status;
    }
    if (ticket.gate.generation == 0) {
        return PrimitiveStatus::ZeroHandle;
    }
    AdmissionGateRecord& gate = storage_.admission_gate[0];
    if (gate.generation != ticket.gate.generation) {
        return PrimitiveStatus::StaleHandle;
    }
    if (gate.state == AdmissionGateState::ClosedExhaustion ||
        gate.state == AdmissionGateState::ClosedInvariant) {
        return PrimitiveStatus::AdmissionClosed;
    }
    if (gate.state != AdmissionGateState::Publishing) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (ticket.admission_credit.request != ticket.request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (ticket.gate.request != ticket.request || gate.request != ticket.request ||
        ticket.authority.request != ticket.request ||
        ticket.authority.immutable_input_generation == 0 ||
        ticket.authority.deadline_proof_generation == 0 ||
        !ticket.authority.immutable_input_ready ||
        !ticket.authority.deadline_valid) {
        return PrimitiveStatus::PublicationProofInvalid;
    }

    PrepublicationRecord& prepublication =
        storage_.prepublication[ticket.prepublication.index];
    AcceptedRecord& accepted = storage_.accepted[ticket.accepted.index];
    AdmissionCreditRecord& credit =
        storage_.admission_credits[ticket.admission_credit.credit_index];
    SubmitCellRecord& submit = storage_.submit_cells[ticket.submit_cell.index];
    if (prepublication.state != PrepublicationState::Publishing ||
        accepted.state != AcceptedState::Free ||
        credit.state != AdmissionCreditState::Publishing ||
        submit.state != SubmitCellState::Owned) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (prepublication.request != ticket.request || credit.request != ticket.request ||
        credit.accepted != ticket.accepted || submit.request != ticket.request ||
        submit.immutable_input_generation !=
            ticket.authority.immutable_input_generation ||
        submit.deadline_proof_generation !=
            ticket.authority.deadline_proof_generation) {
        return PrimitiveStatus::ForeignHandle;
    }

    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    if (connection_status != PrimitiveStatus::Ok) {
        return connection_status;
    }
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    if (connection.request != ticket.request ||
        connection.state != GeneralConnectionState::Reading) {
        return PrimitiveStatus::InvariantViolation;
    }

    accepted.state = AcceptedState::SubmitPending;
    accepted.request = ticket.request;
    credit.state = AdmissionCreditState::Accepted;
    submit.state = SubmitCellState::Published;
    owner.state = RequestOwnerState::Accepted;
    owner.accepted = ticket.accepted;
    owner.admission_credit = ticket.admission_credit;
    owner.prepublication = kNoPrepublication;
    connection.state = GeneralConnectionState::Attached;
    increment_history(storage_.histories[0].submit_published_total, true);
    if (release_record(prepublication, PrepublicationState::Free,
                       PrepublicationState::Exhausted, generation_limit_)) {
        latch_exhaustion(OwnerDomain::Prepublication, false);
    }
    gate.request = {};
    if (gate.generation >= generation_limit_) {
        latch_exhaustion(OwnerDomain::AdmissionGate, false);
    } else {
        ++gate.generation;
        if (gate.state == AdmissionGateState::Publishing) {
            gate.state = AdmissionGateState::Open;
            gate.close_domain = OwnerDomain::Count;
        }
    }
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::abort_publication(const PublicationTicket& ticket) noexcept {
    const PrimitiveStatus request_status = validate_request_handle(ticket.request);
    if (request_status != PrimitiveStatus::Ok) {
        return request_status;
    }
    RequestOwnerRecord& owner = storage_.owners[ticket.request.owner_index];
    if (owner.state != RequestOwnerState::Prepublication ||
        owner.prepublication != ticket.prepublication) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (owner.admission_credit != ticket.admission_credit ||
        owner.submit_cell != ticket.submit_cell) {
        return PrimitiveStatus::ForeignHandle;
    }
    const PrimitiveStatus prepublication_status =
        validate_generation<PrepublicationRecord, PrepublicationState>(
            storage_.prepublication, ticket.prepublication.index,
            ticket.prepublication.generation, PrepublicationState::Exhausted);
    if (prepublication_status != PrimitiveStatus::Ok) {
        return prepublication_status;
    }
    const PrimitiveStatus credit_status =
        validate_generation<AdmissionCreditRecord, AdmissionCreditState>(
            storage_.admission_credits, ticket.admission_credit.credit_index,
            ticket.admission_credit.credit_generation, AdmissionCreditState::Exhausted);
    if (credit_status != PrimitiveStatus::Ok) {
        return credit_status;
    }
    const PrimitiveStatus submit_status =
        validate_generation<SubmitCellRecord, SubmitCellState>(
            storage_.submit_cells, ticket.submit_cell.index,
            ticket.submit_cell.generation, SubmitCellState::Exhausted);
    if (submit_status != PrimitiveStatus::Ok) {
        return submit_status;
    }
    if (ticket.gate.generation == 0) {
        return PrimitiveStatus::ZeroHandle;
    }
    AdmissionGateRecord& gate = storage_.admission_gate[0];
    if (gate.generation != ticket.gate.generation) {
        return PrimitiveStatus::StaleHandle;
    }
    if (gate.state != AdmissionGateState::Publishing) {
        return gate.state == AdmissionGateState::ClosedExhaustion ||
                       gate.state == AdmissionGateState::ClosedInvariant
                   ? PrimitiveStatus::AdmissionClosed
                   : PrimitiveStatus::RepeatedTransition;
    }
    PrepublicationRecord& prepublication =
        storage_.prepublication[ticket.prepublication.index];
    AdmissionCreditRecord& credit =
        storage_.admission_credits[ticket.admission_credit.credit_index];
    SubmitCellRecord& submit = storage_.submit_cells[ticket.submit_cell.index];
    if (prepublication.state != PrepublicationState::Publishing ||
        credit.state != AdmissionCreditState::Publishing ||
        submit.state != SubmitCellState::Owned) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (credit.request != ticket.request || credit.accepted != ticket.accepted ||
        ticket.admission_credit.request != ticket.request ||
        submit.request != ticket.request || gate.request != ticket.request) {
        return PrimitiveStatus::ForeignHandle;
    }
    prepublication.state = PrepublicationState::Prepared;
    if (release_admission(credit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::AdmissionCredit, false);
    }
    if (release_submit(submit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::SubmitCell, false);
    }
    owner.admission_credit = kNoAdmission;
    owner.submit_cell = kNoSubmitCell;
    gate.request = {};
    if (gate.generation >= generation_limit_) {
        latch_exhaustion(OwnerDomain::AdmissionGate, false);
    } else {
        ++gate.generation;
        if (gate.state == AdmissionGateState::Publishing) {
            gate.state = AdmissionGateState::Open;
            gate.close_domain = OwnerDomain::Count;
        }
    }
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::reject_published(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    const PrimitiveStatus credit_status =
        validate_generation<AdmissionCreditRecord, AdmissionCreditState>(
            storage_.admission_credits, owner.admission_credit.credit_index,
            owner.admission_credit.credit_generation,
            AdmissionCreditState::Exhausted);
    const PrimitiveStatus submit_status =
        validate_generation<SubmitCellRecord, SubmitCellState>(
            storage_.submit_cells, owner.submit_cell.index,
            owner.submit_cell.generation, SubmitCellState::Exhausted);
    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return accepted_status;
    }
    if (credit_status != PrimitiveStatus::Ok) {
        return credit_status;
    }
    if (submit_status != PrimitiveStatus::Ok) {
        return submit_status;
    }
    if (connection_status != PrimitiveStatus::Ok) {
        return connection_status;
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    AdmissionCreditRecord& credit =
        storage_.admission_credits[owner.admission_credit.credit_index];
    SubmitCellRecord& submit = storage_.submit_cells[owner.submit_cell.index];
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    if (accepted.state != AcceptedState::SubmitPending ||
        credit.state != AdmissionCreditState::Accepted ||
        submit.state != SubmitCellState::Published ||
        connection.state != GeneralConnectionState::Attached) {
        return PrimitiveStatus::InvalidTransition;
    }
    if (accepted.request != request || credit.request != request ||
        submit.request != request || connection.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }

    increment_history(storage_.histories[0].admission_rejected_total, true);
    if (release_submit(submit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::SubmitCell, false);
    }
    if (release_admission(credit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::AdmissionCredit, false);
    }
    if (release_record(accepted, AcceptedState::Free, AcceptedState::Exhausted,
                       generation_limit_)) {
        latch_exhaustion(OwnerDomain::Accepted, false);
    }
    if (release_record(connection, GeneralConnectionState::Free,
                       GeneralConnectionState::Exhausted, generation_limit_)) {
        latch_exhaustion(OwnerDomain::GeneralConnection, false);
    }
    if (release_owner(owner, generation_limit_)) {
        latch_exhaustion(OwnerDomain::RequestOwner, false);
    }
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::queue_accepted(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    const PrimitiveStatus submit_status =
        validate_generation<SubmitCellRecord, SubmitCellState>(
            storage_.submit_cells, owner.submit_cell.index,
            owner.submit_cell.generation, SubmitCellState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return accepted_status;
    }
    if (submit_status != PrimitiveStatus::Ok) {
        return submit_status;
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    SubmitCellRecord& submit = storage_.submit_cells[owner.submit_cell.index];
    if (accepted.request != request || submit.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (accepted.state == AcceptedState::Queued) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (accepted.state != AcceptedState::SubmitPending) {
        return PrimitiveStatus::InvalidTransition;
    }
    if (submit.state != SubmitCellState::Published) {
        return PrimitiveStatus::InvariantViolation;
    }
    const std::size_t scheduler_index =
        first_state(storage_.scheduler_cells, SchedulerCellState::Free);
    if (scheduler_index == storage_.scheduler_cells.size()) {
        return unavailable_status(storage_.scheduler_cells,
                                  SchedulerCellState::Exhausted) ==
                       PrimitiveStatus::Exhausted
                   ? PrimitiveStatus::Exhausted
                   : PrimitiveStatus::SchedulerCellUnavailable;
    }
    SchedulerCellRecord& scheduler = storage_.scheduler_cells[scheduler_index];
    const SchedulerCellHandle scheduler_handle{scheduler_index,
                                                scheduler.generation};
    scheduler.state = SchedulerCellState::Ready;
    scheduler.request = request;
    accepted.state = AcceptedState::Queued;
    owner.scheduler_cell = scheduler_handle;
    owner.submit_cell = kNoSubmitCell;
    increment_history(storage_.histories[0].accepted_total, true);
    if (release_submit(submit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::SubmitCell, false);
    }
    return PrimitiveStatus::Ok;
}

SlotResult OwnerProtocol::bind_slot(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return {status};
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return {PrimitiveStatus::InvalidTransition};
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return {accepted_status};
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    if (accepted.request != request) {
        return {PrimitiveStatus::ForeignHandle};
    }
    if (accepted.state == AcceptedState::Running) {
        return {PrimitiveStatus::RepeatedTransition};
    }
    if (accepted.state != AcceptedState::SubmitPending &&
        accepted.state != AcceptedState::Queued) {
        return {PrimitiveStatus::InvalidTransition};
    }
    SubmitCellRecord* submit = nullptr;
    SchedulerCellRecord* scheduler = nullptr;
    SchedulerCellHandle retiring{};
    if (accepted.state == AcceptedState::SubmitPending) {
        const PrimitiveStatus submit_status =
            validate_generation<SubmitCellRecord, SubmitCellState>(
                storage_.submit_cells, owner.submit_cell.index,
                owner.submit_cell.generation, SubmitCellState::Exhausted);
        if (submit_status != PrimitiveStatus::Ok) {
            return {submit_status};
        }
        submit = &storage_.submit_cells[owner.submit_cell.index];
        if (submit->state != SubmitCellState::Published ||
            submit->request != request) {
            return {PrimitiveStatus::InvariantViolation};
        }
    } else {
        const PrimitiveStatus scheduler_status =
            validate_generation<SchedulerCellRecord, SchedulerCellState>(
                storage_.scheduler_cells, owner.scheduler_cell.index,
                owner.scheduler_cell.generation,
                SchedulerCellState::Exhausted);
        if (scheduler_status != PrimitiveStatus::Ok) {
            return {scheduler_status};
        }
        scheduler = &storage_.scheduler_cells[owner.scheduler_cell.index];
        if (scheduler->state != SchedulerCellState::Ready ||
            scheduler->request != request) {
            return {PrimitiveStatus::InvariantViolation};
        }
        retiring = owner.scheduler_cell;
    }
    const std::size_t index = first_state(storage_.slots, SlotState::Free);
    if (index == storage_.slots.size()) {
        return {unavailable_status(storage_.slots, SlotState::Exhausted)};
    }
    SlotRecord& slot = storage_.slots[index];
    slot.state = SlotState::RequestOwned;
    slot.request = request;
    owner.slot = {index, slot.generation};
    accepted.state = AcceptedState::Running;
    if (submit != nullptr) {
        owner.submit_cell = kNoSubmitCell;
        increment_history(storage_.histories[0].accepted_total, true);
        if (release_submit(*submit, generation_limit_)) {
            latch_exhaustion(OwnerDomain::SubmitCell, false);
        }
    } else {
        scheduler->state = SchedulerCellState::Retiring;
    }
    return {PrimitiveStatus::Ok, {index, slot.generation}, retiring};
}

PrimitiveStatus
OwnerProtocol::retire_scheduler_cell(RequestHandle request) noexcept {
    const PrimitiveStatus request_status = validate_request_handle(request);
    if (request_status != PrimitiveStatus::Ok) {
        return request_status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus scheduler_status =
        validate_generation<SchedulerCellRecord, SchedulerCellState>(
            storage_.scheduler_cells, owner.scheduler_cell.index,
            owner.scheduler_cell.generation, SchedulerCellState::Exhausted);
    if (scheduler_status != PrimitiveStatus::Ok) {
        return scheduler_status;
    }
    SchedulerCellRecord& scheduler =
        storage_.scheduler_cells[owner.scheduler_cell.index];
    if (scheduler.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (scheduler.state != SchedulerCellState::Retiring) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (release_record(scheduler, SchedulerCellState::Free,
                       SchedulerCellState::Exhausted, generation_limit_)) {
        latch_exhaustion(OwnerDomain::SchedulerCell, false);
    }
    owner.scheduler_cell = kNoSchedulerCell;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::transition_slot(RequestHandle request, SlotHandle slot,
                                               SlotState target) noexcept {
    const PrimitiveStatus request_status = validate_request_handle(request);
    if (request_status != PrimitiveStatus::Ok) {
        return request_status;
    }
    const PrimitiveStatus slot_status = validate_slot_handle(slot);
    if (slot_status != PrimitiveStatus::Ok) {
        return slot_status;
    }
    SlotRecord& record = storage_.slots[slot.slot_index];
    if (record.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (record.state == target) {
        return PrimitiveStatus::RepeatedTransition;
    }
    const bool allowed =
        (record.state == SlotState::RequestOwned &&
         (target == SlotState::TransferPending || target == SlotState::ResetPending ||
          target == SlotState::Poisoned || target == SlotState::FailedRetained)) ||
        (record.state == SlotState::TransferPending &&
         (target == SlotState::RequestOwned || target == SlotState::Poisoned ||
          target == SlotState::FailedRetained)) ||
        (record.state == SlotState::Poisoned && target == SlotState::FailedRetained);
    if (!allowed) {
        return PrimitiveStatus::InvalidTransition;
    }
    record.state = target;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::release_slot(RequestHandle request, SlotHandle slot) noexcept {
    const PrimitiveStatus request_status = validate_request_handle(request);
    if (request_status != PrimitiveStatus::Ok) {
        return request_status;
    }
    const PrimitiveStatus slot_status = validate_slot_handle(slot);
    if (slot_status != PrimitiveStatus::Ok) {
        return slot_status;
    }
    SlotRecord& record = storage_.slots[slot.slot_index];
    if (record.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (record.state != SlotState::ResetPending) {
        return PrimitiveStatus::RepeatedTransition;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.slot != slot) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (release_record(record, SlotState::Free, SlotState::Exhausted,
                       generation_limit_)) {
        latch_exhaustion(OwnerDomain::Slot, false);
    }
    owner.slot = kNoSlotHandle;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::mark_terminal_waiting(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return accepted_status;
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    if (accepted.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (accepted.state == AcceptedState::TerminalWaitingEngineDetach) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (accepted.state != AcceptedState::Running ||
        owner.slot != kNoSlotHandle ||
        owner.scheduler_cell != kNoSchedulerCell ||
        owner.submit_cell != kNoSubmitCell) {
        return PrimitiveStatus::InvalidTransition;
    }
    for (const SlotRecord& slot : storage_.slots) {
        if (slot.request == request && slot.state != SlotState::Free &&
            slot.state != SlotState::Exhausted) {
            return PrimitiveStatus::InvariantViolation;
        }
    }
    accepted.state = AcceptedState::TerminalWaitingEngineDetach;
    increment_history(storage_.histories[0].terminal_selected_total, true);
    return PrimitiveStatus::Ok;
}

DeliveryLease OwnerProtocol::detach_to_delivery(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return {status};
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Accepted) {
        return {owner.state == RequestOwnerState::Delivery
                    ? PrimitiveStatus::RepeatedTransition
                    : PrimitiveStatus::InvalidTransition};
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    if (accepted_status != PrimitiveStatus::Ok) {
        return {accepted_status};
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    if (accepted.request != request ||
        accepted.state != AcceptedState::TerminalWaitingEngineDetach) {
        return {PrimitiveStatus::InvalidTransition};
    }
    const std::size_t delivery_index =
        first_state(storage_.delivery, DeliveryState::Free);
    if (delivery_index == storage_.delivery.size()) {
        return {unavailable_status(storage_.delivery, DeliveryState::Exhausted)};
    }
    const PrimitiveStatus credit_status =
        validate_generation<AdmissionCreditRecord, AdmissionCreditState>(
            storage_.admission_credits, owner.admission_credit.credit_index,
            owner.admission_credit.credit_generation, AdmissionCreditState::Exhausted);
    if (credit_status != PrimitiveStatus::Ok) {
        return {credit_status};
    }
    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    if (connection_status != PrimitiveStatus::Ok) {
        return {connection_status};
    }
    AdmissionCreditRecord& credit =
        storage_.admission_credits[owner.admission_credit.credit_index];
    if (credit.state != AdmissionCreditState::Accepted || credit.request != request) {
        return {PrimitiveStatus::InvariantViolation};
    }
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    if (connection.state != GeneralConnectionState::Attached ||
        connection.request != request) {
        return {PrimitiveStatus::InvariantViolation};
    }

    DeliveryRecord& delivery = storage_.delivery[delivery_index];
    const DeliveryHandle delivery_handle{delivery_index, delivery.generation};
    delivery.state = DeliveryState::TerminalMailbox;
    delivery.request = request;
    credit.state = AdmissionCreditState::Released;
    connection.state = GeneralConnectionState::Output;
    increment_history(storage_.histories[0].terminal_published_total, true);
    if (release_record(accepted, AcceptedState::Free, AcceptedState::Exhausted,
                       generation_limit_)) {
        latch_exhaustion(OwnerDomain::Accepted, false);
    }
    owner.state = RequestOwnerState::Delivery;
    owner.delivery = delivery_handle;
    owner.accepted = kNoAccepted;
    return {PrimitiveStatus::Ok, request, delivery_handle, owner.admission_credit};
}

PrimitiveStatus
OwnerProtocol::recycle_admission_credit(AdmissionCreditHandle credit_handle) noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status =
        validate_generation<AdmissionCreditRecord, AdmissionCreditState>(
            storage_.admission_credits, credit_handle.credit_index,
            credit_handle.credit_generation, AdmissionCreditState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    AdmissionCreditRecord& credit =
        storage_.admission_credits[credit_handle.credit_index];
    if (credit.request != credit_handle.request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (credit.state != AdmissionCreditState::Released) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (credit_handle.request.owner_index < storage_.owners.size()) {
        RequestOwnerRecord& owner =
            storage_.owners[credit_handle.request.owner_index];
        if (owner.generation == credit_handle.request.owner_generation &&
            owner.admission_credit == credit_handle) {
            owner.admission_credit = kNoAdmission;
        }
    }
    if (release_admission(credit, generation_limit_)) {
        latch_exhaustion(OwnerDomain::AdmissionCredit, false);
    }
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::advance_delivery(RequestHandle request,
                                                DeliveryState target) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Delivery) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus delivery_status =
        validate_generation<DeliveryRecord, DeliveryState>(
            storage_.delivery, owner.delivery.index, owner.delivery.generation,
            DeliveryState::Exhausted);
    if (delivery_status != PrimitiveStatus::Ok) {
        return delivery_status;
    }
    DeliveryRecord& delivery = storage_.delivery[owner.delivery.index];
    if (delivery.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (delivery.state == target) {
        return PrimitiveStatus::RepeatedTransition;
    }
    const bool allowed =
        (delivery.state == DeliveryState::TerminalMailbox &&
         target == DeliveryState::Emitting) ||
        (delivery.state == DeliveryState::Emitting &&
         target == DeliveryState::Reclaiming);
    if (!allowed) {
        return PrimitiveStatus::InvalidTransition;
    }
    if (target == DeliveryState::Reclaiming) {
        const PrimitiveStatus connection_status =
            validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
                storage_.general_connections, owner.connection.index,
                owner.connection.generation, GeneralConnectionState::Exhausted);
        if (connection_status != PrimitiveStatus::Ok) {
            return connection_status;
        }
        GeneralConnectionRecord& connection =
            storage_.general_connections[owner.connection.index];
        if (connection.request != request ||
            connection.state != GeneralConnectionState::Output) {
            return PrimitiveStatus::InvariantViolation;
        }
        connection.state = GeneralConnectionState::Closing;
    }
    delivery.state = target;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::finish_delivery(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state != RequestOwnerState::Delivery) {
        return PrimitiveStatus::RepeatedTransition;
    }
    const PrimitiveStatus delivery_status =
        validate_generation<DeliveryRecord, DeliveryState>(
            storage_.delivery, owner.delivery.index, owner.delivery.generation,
            DeliveryState::Exhausted);
    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    if (delivery_status != PrimitiveStatus::Ok) {
        return delivery_status;
    }
    if (connection_status != PrimitiveStatus::Ok) {
        return connection_status;
    }
    DeliveryRecord& delivery = storage_.delivery[owner.delivery.index];
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    if (delivery.request != request || connection.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (delivery.state != DeliveryState::Reclaiming ||
        connection.state != GeneralConnectionState::Closing) {
        return PrimitiveStatus::InvalidTransition;
    }
    increment_history(storage_.histories[0].terminal_consumed_total, true);
    increment_history(storage_.histories[0].delivery_consumed_total, true);
    if (release_record(delivery, DeliveryState::Free, DeliveryState::Exhausted,
                       generation_limit_)) {
        latch_exhaustion(OwnerDomain::Delivery, false);
    }
    if (release_record(connection, GeneralConnectionState::Free,
                       GeneralConnectionState::Exhausted, generation_limit_)) {
        latch_exhaustion(OwnerDomain::GeneralConnection, false);
    }
    if (release_owner(owner, generation_limit_)) {
        latch_exhaustion(OwnerDomain::RequestOwner, false);
    }
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::retain_accepted_failure(RequestHandle request,
                                                       SlotHandle slot) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state == RequestOwnerState::FailedRetained) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (owner.state != RequestOwnerState::Accepted) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus accepted_status =
        validate_generation<AcceptedRecord, AcceptedState>(
            storage_.accepted, owner.accepted.index, owner.accepted.generation,
            AcceptedState::Exhausted);
    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    const PrimitiveStatus slot_status = validate_slot_handle(slot);
    if (accepted_status != PrimitiveStatus::Ok) {
        return accepted_status;
    }
    if (connection_status != PrimitiveStatus::Ok) {
        return connection_status;
    }
    if (slot_status != PrimitiveStatus::Ok) {
        return slot_status;
    }
    AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    SlotRecord& slot_record = storage_.slots[slot.slot_index];
    if (accepted.request != request || connection.request != request ||
        slot_record.request != request || owner.slot != slot) {
        return PrimitiveStatus::ForeignHandle;
    }
    if (accepted.state != AcceptedState::Running ||
        connection.state != GeneralConnectionState::Attached ||
        owner.scheduler_cell != kNoSchedulerCell ||
        owner.submit_cell != kNoSubmitCell ||
        (slot_record.state != SlotState::RequestOwned &&
         slot_record.state != SlotState::TransferPending &&
         slot_record.state != SlotState::Poisoned)) {
        return PrimitiveStatus::InvalidTransition;
    }
    accepted.state = AcceptedState::FailedRetained;
    owner.state = RequestOwnerState::FailedRetained;
    connection.state = GeneralConnectionState::FailedRetained;
    slot_record.state = SlotState::FailedRetained;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::retain_delivery_failure(RequestHandle request) noexcept {
    const PrimitiveStatus status = validate_request_handle(request);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    RequestOwnerRecord& owner = storage_.owners[request.owner_index];
    if (owner.state == RequestOwnerState::FailedRetained) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (owner.state != RequestOwnerState::Delivery) {
        return PrimitiveStatus::InvalidTransition;
    }
    const PrimitiveStatus delivery_status =
        validate_generation<DeliveryRecord, DeliveryState>(
            storage_.delivery, owner.delivery.index, owner.delivery.generation,
            DeliveryState::Exhausted);
    const PrimitiveStatus connection_status =
        validate_generation<GeneralConnectionRecord, GeneralConnectionState>(
            storage_.general_connections, owner.connection.index,
            owner.connection.generation, GeneralConnectionState::Exhausted);
    if (delivery_status != PrimitiveStatus::Ok) {
        return delivery_status;
    }
    if (connection_status != PrimitiveStatus::Ok) {
        return connection_status;
    }
    DeliveryRecord& delivery = storage_.delivery[owner.delivery.index];
    GeneralConnectionRecord& connection =
        storage_.general_connections[owner.connection.index];
    if (delivery.request != request || connection.request != request) {
        return PrimitiveStatus::ForeignHandle;
    }
    delivery.state = DeliveryState::FailedRetained;
    owner.state = RequestOwnerState::FailedRetained;
    connection.state = GeneralConnectionState::FailedRetained;
    return PrimitiveStatus::Ok;
}

ReserveResult OwnerProtocol::acquire_reserve() noexcept {
    if (!initialized_) {
        return {PrimitiveStatus::NotInitialized};
    }
    const std::size_t index =
        first_state(storage_.operational_reserve, ReserveConnectionState::Free);
    if (index == storage_.operational_reserve.size()) {
        return {unavailable_status(storage_.operational_reserve,
                                   ReserveConnectionState::Exhausted)};
    }
    ReserveConnectionRecord& record = storage_.operational_reserve[index];
    record.state = ReserveConnectionState::Unclassified;
    return {PrimitiveStatus::Ok, {index, record.generation}};
}

PrimitiveStatus
OwnerProtocol::transition_reserve(ReserveHandle handle,
                                  ReserveConnectionState target) noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status =
        validate_generation<ReserveConnectionRecord, ReserveConnectionState>(
            storage_.operational_reserve, handle.index, handle.generation,
            ReserveConnectionState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    ReserveConnectionRecord& record = storage_.operational_reserve[handle.index];
    if (record.state == target) {
        return PrimitiveStatus::RepeatedTransition;
    }
    const bool allowed =
        (record.state == ReserveConnectionState::Unclassified &&
         (target == ReserveConnectionState::Operational ||
          target == ReserveConnectionState::Refusing ||
          target == ReserveConnectionState::Closing)) ||
        ((record.state == ReserveConnectionState::Operational ||
          record.state == ReserveConnectionState::Refusing) &&
         (target == ReserveConnectionState::Output ||
          target == ReserveConnectionState::Closing)) ||
        (record.state == ReserveConnectionState::Output &&
         target == ReserveConnectionState::Closing);
    if (!allowed) {
        return PrimitiveStatus::InvalidTransition;
    }
    record.state = target;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::retain_reserve_failure(ReserveHandle handle) noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status =
        validate_generation<ReserveConnectionRecord, ReserveConnectionState>(
            storage_.operational_reserve, handle.index, handle.generation,
            ReserveConnectionState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    ReserveConnectionRecord& record = storage_.operational_reserve[handle.index];
    if (record.state == ReserveConnectionState::FailedRetained) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (record.state == ReserveConnectionState::Free) {
        return PrimitiveStatus::InvalidTransition;
    }
    record.state = ReserveConnectionState::FailedRetained;
    return PrimitiveStatus::Ok;
}

PrimitiveStatus OwnerProtocol::release_reserve(ReserveHandle handle) noexcept {
    if (!initialized_) {
        return PrimitiveStatus::NotInitialized;
    }
    const PrimitiveStatus status =
        validate_generation<ReserveConnectionRecord, ReserveConnectionState>(
            storage_.operational_reserve, handle.index, handle.generation,
            ReserveConnectionState::Exhausted);
    if (status != PrimitiveStatus::Ok) {
        return status;
    }
    ReserveConnectionRecord& record = storage_.operational_reserve[handle.index];
    if (record.state != ReserveConnectionState::Closing) {
        return PrimitiveStatus::RepeatedTransition;
    }
    if (release_reserve_record(record, generation_limit_)) {
        latch_exhaustion(OwnerDomain::OperationalReserve, false);
    }
    return PrimitiveStatus::Ok;
}

ConservationResult OwnerProtocol::validate_conservation() const noexcept {
    ConservationResult result;
    if (!initialized_) {
        result.status = PrimitiveStatus::NotInitialized;
        return result;
    }
    if (validate_owner_topology(topology_) != TopologyStatus::Ok) {
        result.status = PrimitiveStatus::InvariantViolation;
        result.domain = ConservationDomain::Topology;
        return result;
    }
    if (storage_.prepublication.size() != topology_.prepublication_credits ||
        storage_.accepted.size() != topology_.accepted_credits ||
        storage_.delivery.size() != topology_.delivery_credits ||
        storage_.owners.size() != topology_.request_owners ||
        storage_.general_connections.size() != topology_.general_connections ||
        storage_.operational_reserve.size() != topology_.operational_reserve ||
        storage_.slots.size() != topology_.physical_slots ||
        storage_.admission_credits.size() != topology_.accepted_credits ||
        storage_.scheduler_cells.size() != topology_.scheduler_queue_entries ||
        storage_.submit_cells.size() != topology_.accepted_credits ||
        storage_.admission_gate.size() != 1 ||
        storage_.exhaustion.size() !=
            static_cast<std::size_t>(OwnerDomain::Count) ||
        storage_.histories.size() != 1) {
        result.status = PrimitiveStatus::InvariantViolation;
        result.domain = ConservationDomain::Storage;
        return result;
    }

#define TATARA_COUNT_STATE(value, enumerator, counter)                                      \
    case enumerator:                                                                        \
        ++result.counts.counter;                                                             \
        break

    for (const PrepublicationRecord& record : storage_.prepublication) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, PrepublicationState::Free, prepub_free);
            TATARA_COUNT_STATE(record.state, PrepublicationState::Reading, reading);
            TATARA_COUNT_STATE(record.state, PrepublicationState::Preparing, preparing);
            TATARA_COUNT_STATE(record.state, PrepublicationState::Prepared, prepared);
            TATARA_COUNT_STATE(record.state, PrepublicationState::Publishing, publishing);
            TATARA_COUNT_STATE(record.state, PrepublicationState::Exhausted,
                               prepub_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::Prepublication;
            return result;
        }
    }
    for (const AcceptedRecord& record : storage_.accepted) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, AcceptedState::Free, accepted_free);
            TATARA_COUNT_STATE(record.state, AcceptedState::SubmitPending, submit_pending);
            TATARA_COUNT_STATE(record.state, AcceptedState::Queued, queued);
            TATARA_COUNT_STATE(record.state, AcceptedState::Running, running);
            TATARA_COUNT_STATE(record.state, AcceptedState::TerminalWaitingEngineDetach,
                               terminal_waiting_engine_detach);
            TATARA_COUNT_STATE(record.state, AcceptedState::FailedRetained,
                               accepted_failed_retained);
            TATARA_COUNT_STATE(record.state, AcceptedState::Exhausted,
                               accepted_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::Accepted;
            return result;
        }
    }
    for (const DeliveryRecord& record : storage_.delivery) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, DeliveryState::Free, delivery_free);
            TATARA_COUNT_STATE(record.state, DeliveryState::TerminalMailbox,
                               terminal_mailbox);
            TATARA_COUNT_STATE(record.state, DeliveryState::Emitting, emitting);
            TATARA_COUNT_STATE(record.state, DeliveryState::Reclaiming, reclaiming);
            TATARA_COUNT_STATE(record.state, DeliveryState::FailedRetained,
                               delivery_failed_retained);
            TATARA_COUNT_STATE(record.state, DeliveryState::Exhausted,
                               delivery_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::Delivery;
            return result;
        }
    }
    for (const RequestOwnerRecord& record : storage_.owners) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, RequestOwnerState::Free, owner_free);
            TATARA_COUNT_STATE(record.state, RequestOwnerState::Prepublication,
                               owner_prepublication);
            TATARA_COUNT_STATE(record.state, RequestOwnerState::Accepted,
                               owner_accepted);
            TATARA_COUNT_STATE(record.state, RequestOwnerState::Delivery,
                               owner_delivery);
            TATARA_COUNT_STATE(record.state, RequestOwnerState::FailedRetained,
                               owner_failed_retained);
            TATARA_COUNT_STATE(record.state, RequestOwnerState::Exhausted,
                               owner_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::Owner;
            return result;
        }
    }
    for (const GeneralConnectionRecord& record : storage_.general_connections) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Free, general_free);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Reading,
                               general_reading);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Attached,
                               general_attached);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Output,
                               general_output);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Closing,
                               general_closing);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::FailedRetained,
                               general_failed_retained);
            TATARA_COUNT_STATE(record.state, GeneralConnectionState::Exhausted,
                               general_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::GeneralConnection;
            return result;
        }
    }
    for (const ReserveConnectionRecord& record : storage_.operational_reserve) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Free, reserve_free);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Unclassified,
                               reserve_unclassified);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Operational,
                               reserve_operational);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Refusing,
                               reserve_refusing);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Output,
                               reserve_output);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Closing,
                               reserve_closing);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::FailedRetained,
                               reserve_failed_retained);
            TATARA_COUNT_STATE(record.state, ReserveConnectionState::Exhausted,
                               reserve_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::OperationalReserve;
            return result;
        }
    }
    for (const SlotRecord& record : storage_.slots) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, SlotState::Free, slot_free);
            TATARA_COUNT_STATE(record.state, SlotState::RequestOwned, slot_request_owned);
            TATARA_COUNT_STATE(record.state, SlotState::TransferPending,
                               slot_transfer_pending);
            TATARA_COUNT_STATE(record.state, SlotState::ResetPending,
                               slot_reset_pending);
            TATARA_COUNT_STATE(record.state, SlotState::Poisoned, slot_poisoned);
            TATARA_COUNT_STATE(record.state, SlotState::FailedRetained,
                               slot_failed_retained);
            TATARA_COUNT_STATE(record.state, SlotState::Exhausted, slot_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::Slot;
            return result;
        }
    }
    for (const AdmissionCreditRecord& record : storage_.admission_credits) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, AdmissionCreditState::Free, admission_free);
            TATARA_COUNT_STATE(record.state, AdmissionCreditState::Publishing,
                               admission_publishing);
            TATARA_COUNT_STATE(record.state, AdmissionCreditState::Accepted,
                               admission_accepted);
            TATARA_COUNT_STATE(record.state, AdmissionCreditState::Released,
                               admission_released);
            TATARA_COUNT_STATE(record.state, AdmissionCreditState::Exhausted,
                               admission_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::AdmissionCredit;
            return result;
        }
    }
    for (const SchedulerCellRecord& record : storage_.scheduler_cells) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, SchedulerCellState::Free,
                               scheduler_cell_free);
            TATARA_COUNT_STATE(record.state, SchedulerCellState::Ready,
                               scheduler_cell_ready);
            TATARA_COUNT_STATE(record.state, SchedulerCellState::Retiring,
                               scheduler_cell_retiring);
            TATARA_COUNT_STATE(record.state, SchedulerCellState::FailedRetained,
                               scheduler_cell_failed_retained);
            TATARA_COUNT_STATE(record.state, SchedulerCellState::Exhausted,
                               scheduler_cell_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::SchedulerCell;
            return result;
        }
    }
    for (const SubmitCellRecord& record : storage_.submit_cells) {
        switch (record.state) {
            TATARA_COUNT_STATE(record.state, SubmitCellState::Free,
                               submit_cell_free);
            TATARA_COUNT_STATE(record.state, SubmitCellState::Owned,
                               submit_cell_owned);
            TATARA_COUNT_STATE(record.state, SubmitCellState::Published,
                               submit_cell_published);
            TATARA_COUNT_STATE(record.state, SubmitCellState::FailedRetained,
                               submit_cell_failed_retained);
            TATARA_COUNT_STATE(record.state, SubmitCellState::Exhausted,
                               submit_cell_exhausted);
        default:
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::SubmitCell;
            return result;
        }
    }
#undef TATARA_COUNT_STATE

    const auto fail_sum = [&result](ConservationDomain domain, std::uint64_t expected,
                                    std::uint64_t actual) {
        result.status = PrimitiveStatus::InvariantViolation;
        result.domain = domain;
        result.expected = expected;
        result.actual = actual;
    };
    const std::uint64_t prepublication_sum =
        result.counts.prepub_free + result.counts.reading + result.counts.preparing +
        result.counts.prepared + result.counts.publishing +
        result.counts.prepub_exhausted;
    if (prepublication_sum != topology_.prepublication_credits) {
        fail_sum(ConservationDomain::Prepublication, topology_.prepublication_credits,
                 prepublication_sum);
        return result;
    }
    const std::uint64_t accepted_sum =
        result.counts.accepted_free + result.counts.submit_pending + result.counts.queued +
        result.counts.running + result.counts.terminal_waiting_engine_detach +
        result.counts.accepted_failed_retained + result.counts.accepted_exhausted;
    if (accepted_sum != topology_.accepted_credits) {
        fail_sum(ConservationDomain::Accepted, topology_.accepted_credits, accepted_sum);
        return result;
    }
    const std::uint64_t delivery_sum =
        result.counts.delivery_free + result.counts.terminal_mailbox +
        result.counts.emitting + result.counts.reclaiming +
        result.counts.delivery_failed_retained + result.counts.delivery_exhausted;
    if (delivery_sum != topology_.delivery_credits) {
        fail_sum(ConservationDomain::Delivery, topology_.delivery_credits, delivery_sum);
        return result;
    }
    const std::uint64_t owner_sum =
        result.counts.owner_free + result.counts.owner_prepublication +
        result.counts.owner_accepted + result.counts.owner_delivery +
        result.counts.owner_failed_retained + result.counts.owner_exhausted;
    if (owner_sum != topology_.request_owners) {
        fail_sum(ConservationDomain::Owner, topology_.request_owners, owner_sum);
        return result;
    }
    const std::uint64_t general_sum =
        result.counts.general_free + result.counts.general_reading +
        result.counts.general_attached + result.counts.general_output +
        result.counts.general_closing + result.counts.general_failed_retained +
        result.counts.general_exhausted;
    if (general_sum != topology_.general_connections) {
        fail_sum(ConservationDomain::GeneralConnection,
                 topology_.general_connections, general_sum);
        return result;
    }
    const std::uint64_t reserve_sum =
        result.counts.reserve_free + result.counts.reserve_unclassified +
        result.counts.reserve_operational + result.counts.reserve_refusing +
        result.counts.reserve_output + result.counts.reserve_closing +
        result.counts.reserve_failed_retained + result.counts.reserve_exhausted;
    if (reserve_sum != topology_.operational_reserve) {
        fail_sum(ConservationDomain::OperationalReserve,
                 topology_.operational_reserve, reserve_sum);
        return result;
    }
    const std::uint64_t slot_sum =
        result.counts.slot_free + result.counts.slot_request_owned +
        result.counts.slot_transfer_pending + result.counts.slot_reset_pending +
        result.counts.slot_poisoned + result.counts.slot_failed_retained +
        result.counts.slot_exhausted;
    if (slot_sum != topology_.physical_slots) {
        fail_sum(ConservationDomain::Slot, topology_.physical_slots, slot_sum);
        return result;
    }
    const std::uint64_t admission_sum =
        result.counts.admission_free + result.counts.admission_publishing +
        result.counts.admission_accepted + result.counts.admission_released +
        result.counts.admission_exhausted;
    if (admission_sum != topology_.accepted_credits) {
        fail_sum(ConservationDomain::AdmissionCredit,
                 topology_.accepted_credits, admission_sum);
        return result;
    }
    const std::uint64_t scheduler_sum =
        result.counts.scheduler_cell_free + result.counts.scheduler_cell_ready +
        result.counts.scheduler_cell_retiring +
        result.counts.scheduler_cell_failed_retained +
        result.counts.scheduler_cell_exhausted;
    if (scheduler_sum != topology_.scheduler_queue_entries) {
        fail_sum(ConservationDomain::SchedulerCell,
                 topology_.scheduler_queue_entries, scheduler_sum);
        return result;
    }
    const std::uint64_t submit_cell_sum =
        result.counts.submit_cell_free + result.counts.submit_cell_owned +
        result.counts.submit_cell_published +
        result.counts.submit_cell_failed_retained +
        result.counts.submit_cell_exhausted;
    if (submit_cell_sum != topology_.accepted_credits) {
        fail_sum(ConservationDomain::SubmitCell, topology_.accepted_credits,
                 submit_cell_sum);
        return result;
    }

    for (std::size_t index = 0; index < storage_.prepublication.size(); ++index) {
        const PrepublicationRecord& record = storage_.prepublication[index];
        if (record.state == PrepublicationState::Free ||
            record.state == PrepublicationState::Exhausted) {
            if (has_request(record.request)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(record.request) ||
            record.request.owner_index >= storage_.owners.size()) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        const RequestOwnerRecord& owner =
            storage_.owners[record.request.owner_index];
        if (owner.generation != record.request.owner_generation ||
            owner.state != RequestOwnerState::Prepublication ||
            owner.prepublication !=
                PrepublicationHandle{index, record.generation}) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }
    for (std::size_t index = 0; index < storage_.accepted.size(); ++index) {
        const AcceptedRecord& record = storage_.accepted[index];
        if (record.state == AcceptedState::Free ||
            record.state == AcceptedState::Exhausted) {
            if (has_request(record.request)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(record.request) ||
            record.request.owner_index >= storage_.owners.size()) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        const RequestOwnerRecord& owner =
            storage_.owners[record.request.owner_index];
        const RequestOwnerState required =
            record.state == AcceptedState::FailedRetained
                ? RequestOwnerState::FailedRetained
                : RequestOwnerState::Accepted;
        if (owner.generation != record.request.owner_generation ||
            owner.state != required ||
            owner.accepted != AcceptedHandle{index, record.generation}) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }
    for (std::size_t index = 0; index < storage_.delivery.size(); ++index) {
        const DeliveryRecord& record = storage_.delivery[index];
        if (record.state == DeliveryState::Free ||
            record.state == DeliveryState::Exhausted) {
            if (has_request(record.request)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(record.request) ||
            record.request.owner_index >= storage_.owners.size()) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        const RequestOwnerRecord& owner =
            storage_.owners[record.request.owner_index];
        const RequestOwnerState required =
            record.state == DeliveryState::FailedRetained
                ? RequestOwnerState::FailedRetained
                : RequestOwnerState::Delivery;
        if (owner.generation != record.request.owner_generation ||
            owner.state != required ||
            owner.delivery != DeliveryHandle{index, record.generation}) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }

    // Validate both directions of every live request binding. Delayed
    // operations therefore cannot make a stale generation look current.
    for (std::size_t index = 0; index < storage_.owners.size(); ++index) {
        const RequestOwnerRecord& owner = storage_.owners[index];
        const RequestHandle request{index, owner.generation};
        if ((owner.state == RequestOwnerState::Free ||
             owner.state == RequestOwnerState::Exhausted) &&
            !zero_bindings(owner)) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        if (owner.state == RequestOwnerState::Prepublication) {
            if (owner.prepublication.index >= storage_.prepublication.size() ||
                owner.connection.index >= storage_.general_connections.size()) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            const PrepublicationRecord& prepublication =
                storage_.prepublication[owner.prepublication.index];
            const GeneralConnectionRecord& connection =
                storage_.general_connections[owner.connection.index];
            if (prepublication.generation != owner.prepublication.generation ||
                prepublication.request != request ||
                connection.generation != owner.connection.generation ||
                connection.request != request ||
                connection.state != GeneralConnectionState::Reading) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            if (prepublication.state == PrepublicationState::Publishing) {
                if (owner.admission_credit.credit_index >=
                        storage_.admission_credits.size() ||
                    owner.admission_credit.request != request) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
                const AdmissionCreditRecord& credit =
                    storage_.admission_credits[owner.admission_credit.credit_index];
                if (credit.generation !=
                        owner.admission_credit.credit_generation ||
                    credit.state != AdmissionCreditState::Publishing ||
                    credit.request != request) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
            } else if (owner.admission_credit != kNoAdmission) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
        } else if (owner.state == RequestOwnerState::Accepted) {
            if (owner.accepted.index >= storage_.accepted.size() ||
                owner.connection.index >= storage_.general_connections.size() ||
                owner.admission_credit.credit_index >=
                    storage_.admission_credits.size()) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            const AcceptedRecord& accepted = storage_.accepted[owner.accepted.index];
            const GeneralConnectionRecord& connection =
                storage_.general_connections[owner.connection.index];
            const AdmissionCreditRecord& credit =
                storage_.admission_credits[owner.admission_credit.credit_index];
            if (accepted.generation != owner.accepted.generation ||
                accepted.request != request ||
                connection.generation != owner.connection.generation ||
                connection.request != request ||
                connection.state != GeneralConnectionState::Attached ||
                credit.generation != owner.admission_credit.credit_generation ||
                credit.request != request ||
                credit.state != AdmissionCreditState::Accepted) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
        } else if (owner.state == RequestOwnerState::Delivery) {
            if (owner.delivery.index >= storage_.delivery.size() ||
                owner.connection.index >= storage_.general_connections.size()) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            const DeliveryRecord& delivery = storage_.delivery[owner.delivery.index];
            const GeneralConnectionRecord& connection =
                storage_.general_connections[owner.connection.index];
            if (delivery.generation != owner.delivery.generation ||
                delivery.request != request ||
                connection.generation != owner.connection.generation ||
                connection.request != request ||
                (connection.state != GeneralConnectionState::Output &&
                 connection.state != GeneralConnectionState::Closing)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
        } else if (owner.state == RequestOwnerState::FailedRetained) {
            if (owner.connection.index >= storage_.general_connections.size()) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            const GeneralConnectionRecord& connection =
                storage_.general_connections[owner.connection.index];
            if (connection.generation != owner.connection.generation ||
                connection.request != request ||
                connection.state != GeneralConnectionState::FailedRetained) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            if (owner.accepted.generation != 0) {
                if (owner.accepted.index >= storage_.accepted.size()) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
                const AcceptedRecord& accepted =
                    storage_.accepted[owner.accepted.index];
                if (accepted.generation != owner.accepted.generation ||
                    accepted.request != request ||
                    accepted.state != AcceptedState::FailedRetained) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
            } else if (owner.delivery.generation != 0) {
                if (owner.delivery.index >= storage_.delivery.size()) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
                const DeliveryRecord& delivery =
                    storage_.delivery[owner.delivery.index];
                if (delivery.generation != owner.delivery.generation ||
                    delivery.request != request ||
                    delivery.state != DeliveryState::FailedRetained) {
                    result.status = PrimitiveStatus::InvariantViolation;
                    result.domain = ConservationDomain::CrossReference;
                    return result;
                }
            } else {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
        }
    }
    for (const GeneralConnectionRecord& connection : storage_.general_connections) {
        if (connection.state == GeneralConnectionState::Free ||
            connection.state == GeneralConnectionState::Exhausted) {
            if (has_request(connection.request)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(connection.request) ||
            connection.request.owner_index >= storage_.owners.size()) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        const RequestOwnerRecord& owner =
            storage_.owners[connection.request.owner_index];
        if (owner.generation != connection.request.owner_generation ||
            owner.connection != ConnectionHandle{
                                    static_cast<std::uint64_t>(
                                        &connection -
                                        storage_.general_connections.data()),
                                    connection.generation}) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }
    for (const SlotRecord& slot : storage_.slots) {
        if (slot.state == SlotState::Free || slot.state == SlotState::Exhausted) {
            if (has_request(slot.request)) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(slot.request) ||
            validate_request_handle(slot.request) != PrimitiveStatus::Ok) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }
    for (const AdmissionCreditRecord& credit : storage_.admission_credits) {
        if (credit.state == AdmissionCreditState::Free ||
            credit.state == AdmissionCreditState::Exhausted) {
            if (has_request(credit.request) || credit.accepted.generation != 0) {
                result.status = PrimitiveStatus::InvariantViolation;
                result.domain = ConservationDomain::CrossReference;
                return result;
            }
            continue;
        }
        if (!has_request(credit.request) ||
            credit.accepted.index >= storage_.accepted.size()) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        const AcceptedRecord& accepted =
            storage_.accepted[credit.accepted.index];
        if (accepted.generation != credit.accepted.generation &&
            credit.state != AdmissionCreditState::Released) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        if (credit.state == AdmissionCreditState::Publishing &&
            accepted.state != AcceptedState::Free) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
        if (credit.state == AdmissionCreditState::Accepted &&
            (accepted.request != credit.request ||
             (accepted.state != AcceptedState::SubmitPending &&
              accepted.state != AcceptedState::Queued &&
              accepted.state != AcceptedState::Running &&
              accepted.state != AcceptedState::TerminalWaitingEngineDetach &&
              accepted.state != AcceptedState::FailedRetained))) {
            result.status = PrimitiveStatus::InvariantViolation;
            result.domain = ConservationDomain::CrossReference;
            return result;
        }
    }

    // Exhaustion evidence Booleans are normative partition terms: exactly one
    // of evidence-valid/latched, and a latched domain records exactly one
    // owner-retention Boolean.
    bool any_exhaustion_latched = false;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(OwnerDomain::Count); ++index) {
        const ExhaustionEvidence& evidence = storage_.exhaustion[index];
        const std::uint64_t validity_terms =
            static_cast<std::uint64_t>(evidence.evidence_valid) +
            static_cast<std::uint64_t>(evidence.exhaustion_latched);
        if (validity_terms != 1) {
            fail_sum(ConservationDomain::Exhaustion, 1, validity_terms);
            return result;
        }
        const std::uint64_t retention_terms =
            static_cast<std::uint64_t>(evidence.exhaustion_without_owner) +
            static_cast<std::uint64_t>(evidence.exhaustion_owner_retained);
        const std::uint64_t expected_retention =
            evidence.exhaustion_latched ? std::uint64_t{1} : std::uint64_t{0};
        if (retention_terms != expected_retention) {
            fail_sum(ConservationDomain::Exhaustion, expected_retention,
                     retention_terms);
            return result;
        }
        if (evidence.effective_capacity > evidence.configured_capacity ||
            evidence.below_minimum !=
                (evidence.effective_capacity < evidence.minimum_capacity)) {
            fail_sum(ConservationDomain::Exhaustion,
                     evidence.configured_capacity, evidence.effective_capacity);
            return result;
        }
        if (evidence.exhaustion_latched) {
            any_exhaustion_latched = true;
        }
    }
    const AdmissionGateState gate_state = storage_.admission_gate[0].state;
    if (any_exhaustion_latched &&
        gate_state != AdmissionGateState::ClosedExhaustion &&
        gate_state != AdmissionGateState::ClosedInvariant) {
        fail_sum(ConservationDomain::Exhaustion,
                 static_cast<std::uint64_t>(AdmissionGateState::ClosedExhaustion),
                 static_cast<std::uint64_t>(gate_state));
        return result;
    }
    result.partitions_conserved = true;

    // The first exhaustion latch makes every subsequent history equation
    // non-evidence: an unrepresentable event must never be fabricated into a
    // total, so the partition gauges above stay observable but the snapshot
    // can never qualify.
    if (any_exhaustion_latched) {
        return result;
    }
    const OwnerHistories& histories = storage_.histories[0];
    const std::uint64_t expected_submit_published =
        result.counts.submit_pending + histories.admission_rejected_total +
        histories.accepted_total;
    if (histories.submit_published_total != expected_submit_published) {
        fail_sum(ConservationDomain::HistorySubmit, expected_submit_published,
                 histories.submit_published_total);
        return result;
    }
    const std::uint64_t terminal_pending_or_delivery_now =
        result.counts.terminal_waiting_engine_detach +
        result.counts.terminal_mailbox + result.counts.emitting +
        result.counts.reclaiming + result.counts.delivery_failed_retained;
    const std::uint64_t expected_accepted_total =
        result.counts.queued + result.counts.running +
        result.counts.accepted_failed_retained +
        terminal_pending_or_delivery_now + histories.terminal_consumed_total;
    if (histories.accepted_total != expected_accepted_total) {
        fail_sum(ConservationDomain::HistoryAccepted, expected_accepted_total,
                 histories.accepted_total);
        return result;
    }
    const std::uint64_t expected_terminal_selected =
        terminal_pending_or_delivery_now + histories.terminal_consumed_total;
    if (histories.terminal_selected_total != expected_terminal_selected) {
        fail_sum(ConservationDomain::HistoryTerminalSelected,
                 expected_terminal_selected, histories.terminal_selected_total);
        return result;
    }
    const std::uint64_t expected_terminal_published =
        result.counts.terminal_mailbox + result.counts.emitting +
        result.counts.reclaiming + histories.delivery_consumed_total +
        result.counts.delivery_failed_retained;
    if (histories.terminal_published_total != expected_terminal_published) {
        fail_sum(ConservationDomain::HistoryTerminalPublished,
                 expected_terminal_published, histories.terminal_published_total);
        return result;
    }
    result.qualifying = true;
    return result;
}

} // namespace tatara::runtime
