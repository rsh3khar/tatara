#include "tatara/runtime/owner_protocol.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace {

std::size_t allocation_count = 0;
bool observe_allocations = false;

} // namespace

void* operator new(std::size_t size) {
    if (observe_allocations) {
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

using namespace tatara::runtime;

constexpr std::size_t kDomainCount =
    static_cast<std::size_t>(OwnerDomain::Count);
constexpr std::uint64_t kImmutableInputGeneration = 7;
constexpr std::uint64_t kDeadlineProofGeneration = 11;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

PublicationAuthority make_authority(RequestHandle request) {
    return {request, kImmutableInputGeneration, kDeadlineProofGeneration, true,
            true};
}

struct Fixture {
    std::array<PrepublicationRecord, 2> prepublication{};
    std::array<AcceptedRecord, 2> accepted{};
    std::array<DeliveryRecord, 2> delivery{};
    std::array<RequestOwnerRecord, 6> owners{};
    std::array<GeneralConnectionRecord, 2> general{};
    std::array<ReserveConnectionRecord, 2> reserve{};
    std::array<AdmissionCreditRecord, 2> admission{};
    std::array<SlotRecord, 1> slots{};
    std::array<SchedulerCellRecord, 1> scheduler_cells{};
    std::array<SubmitCellRecord, 2> submit_cells{};
    std::array<AdmissionGateRecord, 1> admission_gate{};
    std::array<ExhaustionEvidence, kDomainCount> exhaustion{};
    std::array<OwnerHistories, 1> histories{};
    TopologyResult topology = make_owner_topology(1, 1, 2, 2, 2, 2);
    OwnerProtocol protocol;

    explicit Fixture(
        std::uint64_t generation_limit = std::numeric_limits<std::uint64_t>::max())
        : protocol(topology.topology,
                   {prepublication, accepted, delivery, owners, general, reserve,
                    admission, slots, scheduler_cells, submit_cells,
                    admission_gate, exhaustion, histories},
                   generation_limit) {}
};

struct TinyFixture {
    std::array<PrepublicationRecord, 1> prepublication{};
    std::array<AcceptedRecord, 1> accepted{};
    std::array<DeliveryRecord, 1> delivery{};
    std::array<RequestOwnerRecord, 3> owners{};
    std::array<GeneralConnectionRecord, 1> general{};
    std::array<ReserveConnectionRecord, 1> reserve{};
    std::array<AdmissionCreditRecord, 1> admission{};
    std::array<SlotRecord, 1> slots{};
    std::array<SchedulerCellRecord, 0> scheduler_cells{};
    std::array<SubmitCellRecord, 1> submit_cells{};
    std::array<AdmissionGateRecord, 1> admission_gate{};
    std::array<ExhaustionEvidence, kDomainCount> exhaustion{};
    std::array<OwnerHistories, 1> histories{};
    TopologyResult topology = make_owner_topology(1, 0, 1, 1, 1, 1);
    OwnerProtocol protocol;

    explicit TinyFixture(
        std::uint64_t generation_limit = std::numeric_limits<std::uint64_t>::max())
        : protocol(topology.topology,
                   {prepublication, accepted, delivery, owners, general, reserve,
                    admission, slots, scheduler_cells, submit_cells,
                    admission_gate, exhaustion, histories},
                   generation_limit) {}
};

template <class AnyFixture>
PublicationResult prepare_and_begin(AnyFixture& fixture, RequestLease& lease) {
    lease = fixture.protocol.acquire_prepublication();
    check(static_cast<bool>(lease), "prepublication acquisition succeeds");
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Preparing) ==
              PrimitiveStatus::Ok,
          "reading advances to preparing");
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Prepared) ==
              PrimitiveStatus::Ok,
          "preparing advances to prepared");
    const PublicationResult publication =
        fixture.protocol.begin_publication(make_authority(lease.request));
    check(static_cast<bool>(publication), "prepared owner begins publication");
    return publication;
}

void topology_zero_max_and_overflow_are_typed() {
    check(make_owner_topology(0, 0, 1, 1, 1, 1).status ==
              TopologyStatus::PhysicalSlotsZero,
          "C zero is refused");
    check(make_owner_topology(1, 0, 1, 1, 0, 1).status ==
              TopologyStatus::GeneralConnectionsZero,
          "G zero is refused");
    check(make_owner_topology(1, 0, 1, 1, 1, 0).status ==
              TopologyStatus::OperationalReserveZero,
          "R_op zero is refused");
    check(make_owner_topology(2, 0, 1, 2, 2, 1).status ==
              TopologyStatus::PrepublicationBelowGeneral,
          "U below G is refused");
    check(make_owner_topology(2, 1, 2, 2, 1, 1).status ==
              TopologyStatus::DeliveryBelowAccepted,
          "D below A is refused");

    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    const TopologyResult maximal =
        make_owner_topology(maximum / 2, 0, 1, maximum / 2, 1, maximum - 1);
    check(maximal && maximal.topology.request_owners == maximum &&
              maximal.topology.connection_slots == maximum,
          "maximum representable O and K are accepted without narrowing");
    check(make_owner_topology(maximum, 1, maximum, maximum, 1, 1).status ==
              TopologyStatus::AcceptedOverflow,
          "A addition overflow is typed");
    check(make_owner_topology(1, 0, maximum, 1, maximum, 1).status ==
              TopologyStatus::ConnectionOverflow,
          "K addition overflow is typed");
    check(make_owner_topology(maximum / 2, 0, 2, maximum / 2, 1, 1).status ==
              TopologyStatus::OwnerOverflow,
          "O addition overflow is typed");

    OwnerTopology corrupted = make_owner_topology(1, 1, 2, 2, 2, 1).topology;
    ++corrupted.accepted_credits;
    check(validate_owner_topology(corrupted) ==
              TopologyStatus::AcceptedEquationMismatch,
          "A=C+Q is revalidated");
    corrupted = make_owner_topology(1, 1, 2, 2, 2, 1).topology;
    ++corrupted.request_owners;
    check(validate_owner_topology(corrupted) == TopologyStatus::OwnerEquationMismatch,
          "O=A+U+D is revalidated");
    corrupted = make_owner_topology(1, 1, 2, 2, 2, 1).topology;
    ++corrupted.connection_slots;
    check(validate_owner_topology(corrupted) ==
              TopologyStatus::ConnectionEquationMismatch,
          "K=G+R_op is revalidated");
}

void initialization_is_fixed_and_fail_closed() {
    Fixture fixture;
    check(fixture.topology && fixture.topology.topology.accepted_credits == 2 &&
              fixture.topology.topology.request_owners == 6 &&
              fixture.topology.topology.connection_slots == 4,
          "fixture equations are exact");
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok,
          "fresh fixed storage initializes");
    check(fixture.protocol.initialize() == PrimitiveStatus::AlreadyInitialized,
          "storage cannot be reinitialized");
    const ConservationResult initial = fixture.protocol.validate_conservation();
    check(initial && initial.counts.prepub_free == 2 &&
              initial.counts.accepted_free == 2 &&
              initial.counts.delivery_free == 2 && initial.counts.owner_free == 6 &&
              initial.counts.general_free == 2 && initial.counts.reserve_free == 2 &&
              initial.counts.slot_free == 1 && initial.counts.admission_free == 2 &&
              initial.counts.scheduler_cell_free == 1 &&
              initial.counts.submit_cell_free == 2,
          "every initial partition is conserved");
    check(initial.partitions_conserved && initial.qualifying,
          "an untouched machine is qualifying evidence");

    TinyFixture zero_generation_limit(0);
    check(zero_generation_limit.protocol.initialize() ==
              PrimitiveStatus::GenerationLimitZero,
          "zero generation limit is refused");

    TinyFixture invalid_policy;
    invalid_policy
        .exhaustion[static_cast<std::size_t>(OwnerDomain::Prepublication)]
        .minimum_capacity = 2;
    check(invalid_policy.protocol.initialize() ==
              PrimitiveStatus::MinimumCapacityPolicyInvalid,
          "a profile that cannot preserve minimum capacity is refused");

    std::array<PrepublicationRecord, 1> prepublication{};
    std::array<AcceptedRecord, 1> accepted{};
    std::array<DeliveryRecord, 1> delivery{};
    std::array<RequestOwnerRecord, 2> owners{};
    std::array<GeneralConnectionRecord, 1> general{};
    std::array<ReserveConnectionRecord, 1> reserve{};
    std::array<AdmissionCreditRecord, 1> admission{};
    std::array<SlotRecord, 1> slots{};
    std::array<SchedulerCellRecord, 0> scheduler_cells{};
    std::array<SubmitCellRecord, 1> submit_cells{};
    std::array<AdmissionGateRecord, 1> admission_gate{};
    std::array<ExhaustionEvidence, kDomainCount> exhaustion{};
    std::array<OwnerHistories, 1> histories{};
    const TopologyResult topology = make_owner_topology(1, 0, 1, 1, 1, 1);
    OwnerProtocol wrong_size(topology.topology,
                             {prepublication, accepted, delivery, owners, general,
                              reserve, admission, slots, scheduler_cells,
                              submit_cells, admission_gate, exhaustion,
                              histories});
    check(wrong_size.initialize() == PrimitiveStatus::StorageMismatch,
          "storage must exactly match O");
}

void operation_results_carry_frozen_payload() {
    const OperationResult ok = make_operation_result(
        PrimitiveStatus::Ok, OperationPhase::Publication, 3, 3,
        OwnerDisposition::Transferred);
    check(ok && ok.phase == OperationPhase::Publication &&
              ok.internal_cause == PrimitiveStatus::Ok &&
              ok.disposition == OwnerDisposition::Transferred &&
              ok.expected == 3 && ok.actual == 3 &&
              ok.next_action == NextAction::Continue,
          "success payload carries phase, disposition and equation terms");

    const OperationResult retry = make_operation_result(
        PrimitiveStatus::CapacityUnavailable, OperationPhase::Queue);
    check(!retry && retry.internal_cause == PrimitiveStatus::CapacityUnavailable &&
              retry.disposition == OwnerDisposition::Unchanged &&
              retry.next_action == NextAction::RetryWhenCapacityChanges,
          "capacity unavailability is a typed retry");

    const OperationResult stale = make_operation_result(
        PrimitiveStatus::StaleHandle, OperationPhase::Identity, 4, 9);
    check(!stale && stale.expected == 4 && stale.actual == 9 &&
              stale.next_action == NextAction::RejectDelayedOperation,
          "stale identity rejects the delayed operation");

    const OperationResult exhausted = make_operation_result(
        PrimitiveStatus::Exhausted, OperationPhase::Exhaustion);
    check(!exhausted && exhausted.disposition == OwnerDisposition::AdmissionClosed &&
              exhausted.next_action == NextAction::CloseAdmission,
          "exhaustion closes admission in the typed payload");

    const OperationResult invalid_evidence = make_operation_result(
        PrimitiveStatus::EvidenceInvalid, OperationPhase::Conservation);
    check(!invalid_evidence &&
              invalid_evidence.disposition == OwnerDisposition::EvidenceInvalid &&
              invalid_evidence.next_action == NextAction::CloseAdmission,
          "invalid evidence closes admission");

    const OperationResult violation = make_operation_result(
        PrimitiveStatus::InvariantViolation, OperationPhase::Conservation);
    check(!violation && violation.disposition == OwnerDisposition::FailedRetained &&
              violation.next_action == NextAction::RetainAndStop,
          "invariant violation retains and stops");

    const OperationResult repair = make_operation_result(
        PrimitiveStatus::StorageMismatch, OperationPhase::Initialization);
    check(!repair && repair.next_action == NextAction::RepairInvariant,
          "storage mismatch demands repair");
    check(make_operation_result(PrimitiveStatus::AlreadyInitialized,
                                OperationPhase::Initialization)
                  .next_action == NextAction::RecreateProcess,
          "double initialization demands process recreation");

    const OperationResult coerced =
        make_operation_result(PrimitiveStatus::Count, OperationPhase::Identity);
    check(coerced.status == PrimitiveStatus::InvariantViolation &&
              coerced.next_action == NextAction::RepairInvariant,
          "an unrepresentable status is coerced to a typed violation");

    Fixture fixture;
    const OperationResult detailed = fixture.protocol.detailed_result(
        PrimitiveStatus::ForeignHandle, OperationPhase::Identity, 1, 2);
    check(!detailed && detailed.phase == OperationPhase::Identity &&
              detailed.internal_cause == PrimitiveStatus::ForeignHandle &&
              detailed.next_action == NextAction::RejectDelayedOperation,
          "detailed_result reuses the frozen payload mapping");
}

void compound_publication_owns_gate_inputs_and_deadline() {
    Fixture fixture;
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok, "fixture initializes");
    const auto* owner_address = fixture.owners.data();
    const auto* accepted_address = fixture.accepted.data();

    RequestLease lease = fixture.protocol.acquire_prepublication();
    check(static_cast<bool>(lease), "first lease acquires");
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Preparing) ==
                  PrimitiveStatus::Ok &&
              fixture.protocol.advance_prepublication(
                  lease.request, PrepublicationState::Prepared) ==
                  PrimitiveStatus::Ok,
          "first lease reaches prepared");
    RequestLease second = fixture.protocol.acquire_prepublication();
    check(static_cast<bool>(second) &&
              fixture.protocol.advance_prepublication(
                  second.request, PrepublicationState::Preparing) ==
                  PrimitiveStatus::Ok &&
              fixture.protocol.advance_prepublication(
                  second.request, PrepublicationState::Prepared) ==
                  PrimitiveStatus::Ok,
          "second lease reaches prepared");

    PublicationAuthority missing_input = make_authority(lease.request);
    missing_input.immutable_input_generation = 0;
    check(fixture.protocol.begin_publication(missing_input).status ==
              PrimitiveStatus::PublicationProofInvalid,
          "publication without an immutable input generation is refused");
    PublicationAuthority missing_deadline = make_authority(lease.request);
    missing_deadline.deadline_proof_generation = 0;
    check(fixture.protocol.begin_publication(missing_deadline).status ==
              PrimitiveStatus::PublicationProofInvalid,
          "publication without a deadline proof is refused");
    PublicationAuthority unready_input = make_authority(lease.request);
    unready_input.immutable_input_ready = false;
    check(fixture.protocol.begin_publication(unready_input).status ==
              PrimitiveStatus::PublicationProofInvalid,
          "publication with unready immutable input is refused");
    PublicationAuthority expired_deadline = make_authority(lease.request);
    expired_deadline.deadline_valid = false;
    check(fixture.protocol.begin_publication(expired_deadline).status ==
              PrimitiveStatus::PublicationProofInvalid,
          "publication with an invalid deadline is refused");

    const PublicationResult publication =
        fixture.protocol.begin_publication(make_authority(lease.request));
    check(static_cast<bool>(publication), "valid authority begins publication");
    ConservationResult state = fixture.protocol.validate_conservation();
    check(state && state.counts.publishing == 1 &&
              state.counts.admission_publishing == 1 &&
              state.counts.submit_cell_owned == 1 &&
              state.counts.owner_prepublication == 2 &&
              state.counts.general_reading == 2 && state.qualifying,
          "publishing owns U, O, G, the submit cell and the admission credit");
    check(publication.ticket.admission_credit.request == lease.request &&
              publication.ticket.gate.request == lease.request,
          "credit and gate carry the exact request generation");
    check(fixture.protocol.begin_publication(make_authority(second.request))
                  .status == PrimitiveStatus::GateUnavailable,
          "the publication gate is owned by exactly one request");

    PublicationTicket stale_gate = publication.ticket;
    ++stale_gate.gate.generation;
    check(fixture.protocol.commit_publication(stale_gate) ==
              PrimitiveStatus::StaleHandle,
          "a stale gate generation cannot commit");
    PublicationTicket revoked_authority = publication.ticket;
    revoked_authority.authority.deadline_valid = false;
    check(fixture.protocol.commit_publication(revoked_authority) ==
              PrimitiveStatus::PublicationProofInvalid,
          "a revoked deadline proof cannot commit");
    PublicationTicket swapped_input = publication.ticket;
    swapped_input.authority.immutable_input_generation =
        kImmutableInputGeneration + 1;
    check(fixture.protocol.commit_publication(swapped_input) ==
              PrimitiveStatus::ForeignHandle,
          "the submit cell owns the immutable input generation");

    check(fixture.protocol.commit_publication(publication.ticket) ==
              PrimitiveStatus::Ok,
          "compound publication commits");
    check(fixture.protocol.commit_publication(publication.ticket) ==
              PrimitiveStatus::RepeatedTransition,
          "publication cannot commit twice");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.submit_pending == 1 &&
              state.counts.owner_accepted == 1 &&
              state.counts.general_attached == 1 &&
              state.counts.admission_accepted == 1 &&
              state.counts.submit_cell_published == 1 && state.qualifying,
          "publication atomically transfers U to A");
    const PublicationResult reopened =
        fixture.protocol.begin_publication(make_authority(second.request));
    check(static_cast<bool>(reopened),
          "the gate reopens with a fresh generation after commit");
    check(fixture.protocol.abort_publication(reopened.ticket) ==
              PrimitiveStatus::Ok,
          "the second publication aborts cleanly");

    check(fixture.protocol.queue_accepted(lease.request) == PrimitiveStatus::Ok,
          "submit-pending request queues");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.queued == 1 &&
              state.counts.scheduler_cell_ready == 1 &&
              state.counts.submit_cell_free == 2,
          "queued A owns an independent Q cell and returns the submit cell");
    const SlotResult slot = fixture.protocol.bind_slot(lease.request);
    check(static_cast<bool>(slot), "queued request binds the physical slot");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.running == 1 &&
              state.counts.scheduler_cell_retiring == 1,
          "binding retires the scheduler cell explicitly");
    check(fixture.protocol.retire_scheduler_cell(lease.request) ==
              PrimitiveStatus::Ok,
          "the retiring scheduler cell is retired");
    check(fixture.protocol.retire_scheduler_cell(lease.request) ==
              PrimitiveStatus::ZeroHandle,
          "a retired scheduler cell cannot be retired twice");
    check(fixture.protocol.transition_slot(lease.request, slot.slot,
                                           SlotState::TransferPending) ==
              PrimitiveStatus::Ok,
          "running request owns a generation-bound transfer slot");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.slot_transfer_pending == 1 &&
              state.counts.scheduler_cell_free == 1,
          "transfer-pending and recycled Q partitions are conserved");
    check(fixture.protocol.transition_slot(lease.request, slot.slot,
                                           SlotState::RequestOwned) ==
              PrimitiveStatus::Ok,
          "slot transfer completes to the same request generation");
    check(fixture.protocol.transition_slot(lease.request, slot.slot,
                                           SlotState::ResetPending) ==
              PrimitiveStatus::Ok,
          "slot enters reset-pending before release");
    check(fixture.protocol.release_slot(lease.request, slot.slot) ==
              PrimitiveStatus::Ok,
          "reset slot releases");
    check(fixture.protocol.validate_slot_handle(slot.slot) ==
              PrimitiveStatus::StaleHandle,
          "released slot handle is stale");

    check(fixture.protocol.mark_terminal_waiting(lease.request) ==
              PrimitiveStatus::Ok,
          "slot-safe request reaches terminal detach");
    const DeliveryLease delivery = fixture.protocol.detach_to_delivery(lease.request);
    check(static_cast<bool>(delivery), "accepted request detaches into delivery");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.accepted_free == 2 &&
              state.counts.terminal_mailbox == 1 &&
              state.counts.owner_delivery == 1 &&
              state.counts.general_output == 1 &&
              state.counts.admission_released == 1,
          "A transfers to D while credit becomes Released");
    check(fixture.protocol.recycle_admission_credit(delivery.admission_credit) ==
              PrimitiveStatus::Ok,
          "released admission credit recycles with generation advance");
    check(fixture.protocol.advance_delivery(lease.request, DeliveryState::Emitting) ==
              PrimitiveStatus::Ok,
          "mailbox delivery starts emitting");
    check(fixture.protocol.advance_delivery(lease.request, DeliveryState::Reclaiming) ==
              PrimitiveStatus::Ok,
          "delivery reaches reclaiming and connection closing");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.reclaiming == 1 &&
              state.counts.general_closing == 1,
          "reclaiming and closing partitions are explicit");
    check(fixture.protocol.finish_delivery(lease.request) == PrimitiveStatus::Ok,
          "delivery, owner and connection release together");
    check(fixture.protocol.validate_request_handle(lease.request) ==
              PrimitiveStatus::StaleHandle,
          "completed request generation is stale");
    state = fixture.protocol.validate_conservation();
    check(state && state.partitions_conserved && state.qualifying,
          "completed transfer conserves all pools and qualifies");
    check(fixture.histories[0].submit_published_total == 1 &&
              fixture.histories[0].accepted_total == 1 &&
              fixture.histories[0].terminal_selected_total == 1 &&
              fixture.histories[0].terminal_published_total == 1 &&
              fixture.histories[0].terminal_consumed_total == 1 &&
              fixture.histories[0].delivery_consumed_total == 1,
          "every normative history counts the single completed request");
    check(owner_address == fixture.owners.data() &&
              accepted_address == fixture.accepted.data(),
          "caller-backed records remain address-stable");
}

void typed_identities_are_exhaustively_classified() {
    Fixture fixture;
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok, "fixture initializes");
    check(fixture.protocol.validate_request_handle({0, 0}) ==
              PrimitiveStatus::ZeroHandle,
          "zero request handle is typed");
    check(fixture.protocol.validate_request_handle({999, 1}) ==
              PrimitiveStatus::ForeignHandle,
          "foreign request index is typed");

    RequestLease lease = fixture.protocol.acquire_prepublication();
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Reading) ==
              PrimitiveStatus::RepeatedTransition,
          "repeated state transition is typed");
    check(fixture.protocol.advance_prepublication(
              {lease.request.owner_index, lease.request.owner_generation + 1},
              PrepublicationState::Preparing) == PrimitiveStatus::StaleHandle,
          "wrong request generation is stale");
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Publishing) ==
              PrimitiveStatus::InvalidTransition,
          "unowned Publishing edge is refused");

    // Bounded exhaustive classification: every (index, generation) pair in the
    // grid must produce exactly the typed status the raw records demand.
    for (std::uint64_t index = 0; index < fixture.owners.size() + 3; ++index) {
        for (std::uint64_t generation = 0; generation < 4; ++generation) {
            PrimitiveStatus expected = PrimitiveStatus::Ok;
            if (generation == 0) {
                expected = PrimitiveStatus::ZeroHandle;
            } else if (index >= fixture.owners.size()) {
                expected = PrimitiveStatus::ForeignHandle;
            } else {
                const RequestOwnerRecord& record =
                    fixture.owners[static_cast<std::size_t>(index)];
                if (record.generation != generation) {
                    expected = PrimitiveStatus::StaleHandle;
                } else if (record.state == RequestOwnerState::Exhausted) {
                    expected = PrimitiveStatus::Exhausted;
                } else if (record.state == RequestOwnerState::Free) {
                    expected = PrimitiveStatus::RepeatedTransition;
                }
            }
            check(fixture.protocol.validate_request_handle({index, generation}) ==
                      expected,
                  "request identity grid classifies exactly");
        }
    }
    for (std::uint64_t index = 0; index < fixture.slots.size() + 3; ++index) {
        for (std::uint64_t generation = 0; generation < 4; ++generation) {
            PrimitiveStatus expected = PrimitiveStatus::Ok;
            if (generation == 0) {
                expected = PrimitiveStatus::ZeroHandle;
            } else if (index >= fixture.slots.size()) {
                expected = PrimitiveStatus::ForeignHandle;
            } else {
                const SlotRecord& record =
                    fixture.slots[static_cast<std::size_t>(index)];
                if (record.generation != generation) {
                    expected = PrimitiveStatus::StaleHandle;
                } else if (record.state == SlotState::Exhausted) {
                    expected = PrimitiveStatus::Exhausted;
                } else if (record.state == SlotState::Free) {
                    expected = PrimitiveStatus::RepeatedTransition;
                }
            }
            check(fixture.protocol.validate_slot_handle({index, generation}) ==
                      expected,
                  "slot identity grid classifies exactly");
        }
    }

    // Malformed subordinate handles must be rejected before any dereference.
    check(fixture.protocol.advance_prepublication(lease.request,
                                                  PrepublicationState::Preparing) ==
                  PrimitiveStatus::Ok &&
              fixture.protocol.advance_prepublication(
                  lease.request, PrepublicationState::Prepared) ==
                  PrimitiveStatus::Ok,
          "classification lease reaches prepared");
    const PublicationResult publication =
        fixture.protocol.begin_publication(make_authority(lease.request));
    check(static_cast<bool>(publication), "classification lease begins publication");
    PublicationTicket malformed = publication.ticket;
    malformed.prepublication.index = 1000000;
    check(fixture.protocol.commit_publication(malformed) ==
              PrimitiveStatus::ForeignHandle,
          "a malformed prepublication index is typed, not dereferenced");
    malformed = publication.ticket;
    malformed.accepted.index = 1000000;
    check(fixture.protocol.commit_publication(malformed) ==
              PrimitiveStatus::ForeignHandle,
          "a malformed accepted index is typed, not dereferenced");
    malformed = publication.ticket;
    malformed.admission_credit.credit_index = 1000000;
    check(fixture.protocol.commit_publication(malformed) ==
              PrimitiveStatus::ForeignHandle,
          "a malformed admission-credit index is typed, not dereferenced");
    malformed = publication.ticket;
    malformed.submit_cell.index = 1000000;
    check(fixture.protocol.abort_publication(malformed) ==
              PrimitiveStatus::ForeignHandle,
          "a malformed submit-cell index is typed, not dereferenced");
    check(fixture.protocol.recycle_admission_credit(
              {1000000, 1, lease.request}) == PrimitiveStatus::ForeignHandle,
          "a malformed credit handle is typed, not dereferenced");
    check(fixture.protocol.abort_publication(publication.ticket) ==
              PrimitiveStatus::Ok,
          "classification publication aborts cleanly");

    const CommandTicket expected{7, CommandKind::Decode, lease.request, {0, 3}};
    check(validate_command_ticket(expected, expected) == PrimitiveStatus::Ok,
          "exact command ticket validates");
    CommandTicket changed = expected;
    changed.command_generation = 6;
    check(validate_command_ticket(changed, expected) == PrimitiveStatus::StaleHandle,
          "stale command generation is typed");
    changed = expected;
    changed.kind = CommandKind::Reset;
    check(validate_command_ticket(changed, expected) == PrimitiveStatus::ForeignHandle,
          "foreign command kind is typed");
    changed = expected;
    changed.slot.slot_generation = 0;
    check(validate_command_ticket(changed, expected) == PrimitiveStatus::ZeroHandle,
          "zero nested command handle is typed");

    const CacheHandle cache{4, 9};
    check(cache == CacheHandle{4, 9}, "cache handles preserve index and generation");
}

void abort_reject_and_failed_retained_are_slot_coupled() {
    Fixture abort_fixture;
    check(abort_fixture.protocol.initialize() == PrimitiveStatus::Ok,
          "abort fixture initializes");
    RequestLease abort_lease;
    const PublicationResult abort_publication =
        prepare_and_begin(abort_fixture, abort_lease);
    PublicationTicket foreign = abort_publication.ticket;
    foreign.admission_credit.request.owner_index = 99;
    check(abort_fixture.protocol.commit_publication(foreign) ==
              PrimitiveStatus::ForeignHandle,
          "foreign admission-credit binding is refused");
    check(abort_fixture.protocol.abort_publication(abort_publication.ticket) ==
              PrimitiveStatus::Ok,
          "publication abort returns its credit");
    ConservationResult state = abort_fixture.protocol.validate_conservation();
    check(state && state.counts.prepared == 1 &&
              state.counts.admission_free == 2 &&
              state.counts.submit_cell_free == 2 && state.qualifying,
          "abort restores Prepared and returns credit and submit cell");
    const PublicationResult retry =
        abort_fixture.protocol.begin_publication(make_authority(abort_lease.request));
    check(static_cast<bool>(retry) &&
              abort_fixture.protocol.commit_publication(retry.ticket) ==
                  PrimitiveStatus::Ok,
          "aborted publication can be re-begun and committed");
    check(abort_fixture.protocol.reject_published(abort_lease.request) ==
              PrimitiveStatus::Ok,
          "a published request can be rejected before queueing");
    state = abort_fixture.protocol.validate_conservation();
    check(state && state.qualifying && state.counts.owner_free == 6 &&
              state.counts.general_free == 2 &&
              abort_fixture.histories[0].admission_rejected_total == 1 &&
              abort_fixture.histories[0].submit_published_total == 1,
          "rejection is a history term, not a hidden state");
    check(abort_fixture.protocol.validate_request_handle(abort_lease.request) ==
              PrimitiveStatus::StaleHandle,
          "rejected request generation is stale");

    Fixture accepted_failure;
    check(accepted_failure.protocol.initialize() == PrimitiveStatus::Ok,
          "accepted-failure fixture initializes");
    RequestLease accepted_lease;
    const PublicationResult accepted_publication =
        prepare_and_begin(accepted_failure, accepted_lease);
    check(accepted_failure.protocol.commit_publication(accepted_publication.ticket) ==
              PrimitiveStatus::Ok,
          "failure request publishes");
    const SlotResult failed_slot =
        accepted_failure.protocol.bind_slot(accepted_lease.request);
    check(failed_slot &&
              accepted_failure.protocol.transition_slot(
                  accepted_lease.request, failed_slot.slot, SlotState::Poisoned) ==
                  PrimitiveStatus::Ok,
          "bound slot becomes poisoned");
    check(accepted_failure.protocol.retain_accepted_failure(
              accepted_lease.request, {0, 0}) == PrimitiveStatus::ZeroHandle,
          "failure retention without the slot identity is typed");
    check(accepted_failure.protocol.retain_accepted_failure(
              accepted_lease.request,
              {failed_slot.slot.slot_index,
               failed_slot.slot.slot_generation + 1}) ==
              PrimitiveStatus::StaleHandle,
          "failure retention with a stale slot identity is typed");
    check(accepted_failure.protocol.retain_accepted_failure(
              accepted_lease.request, failed_slot.slot) == PrimitiveStatus::Ok,
          "accepted failure retains A/O/G with its exact slot");
    state = accepted_failure.protocol.validate_conservation();
    check(state && state.counts.accepted_failed_retained == 1 &&
              state.counts.owner_failed_retained == 1 &&
              state.counts.general_failed_retained == 1 &&
              state.counts.slot_failed_retained == 1,
          "accepted, owner, connection and slot failure terms remain current");
    check(state.partitions_conserved && state.qualifying,
          "failed-retained is a current state and still qualifying evidence");

    Fixture delivery_failure;
    check(delivery_failure.protocol.initialize() == PrimitiveStatus::Ok,
          "delivery-failure fixture initializes");
    RequestLease delivery_request;
    const PublicationResult delivery_publication =
        prepare_and_begin(delivery_failure, delivery_request);
    check(delivery_failure.protocol.commit_publication(delivery_publication.ticket) ==
                  PrimitiveStatus::Ok &&
              delivery_failure.protocol.bind_slot(delivery_request.request).status ==
                  PrimitiveStatus::Ok,
          "delivery-failure request runs");
    const SlotHandle delivery_slot =
        delivery_failure.owners[delivery_request.request.owner_index].slot;
    check(delivery_failure.protocol.transition_slot(delivery_request.request,
                                                    delivery_slot,
                                                    SlotState::ResetPending) ==
                  PrimitiveStatus::Ok &&
              delivery_failure.protocol.release_slot(delivery_request.request,
                                                     delivery_slot) ==
                  PrimitiveStatus::Ok &&
              delivery_failure.protocol.mark_terminal_waiting(
                  delivery_request.request) == PrimitiveStatus::Ok,
          "delivery-failure request reaches terminal");
    const DeliveryLease detached =
        delivery_failure.protocol.detach_to_delivery(delivery_request.request);
    check(static_cast<bool>(detached), "delivery-failure request detaches");
    check(delivery_failure.protocol.retain_delivery_failure(delivery_request.request) ==
              PrimitiveStatus::Ok,
          "delivery failure retains D/O/G");
    state = delivery_failure.protocol.validate_conservation();
    check(state && state.counts.delivery_failed_retained == 1 &&
              state.counts.owner_failed_retained == 1 &&
              state.counts.general_failed_retained == 1 && state.qualifying,
          "delivery failed-retained partitions conserve with exact histories");
}

void reserve_partitions_and_exhaustion_never_resurrect() {
    Fixture fixture;
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok, "fixture initializes");
    const ReserveResult operational = fixture.protocol.acquire_reserve();
    check(static_cast<bool>(operational), "reserve becomes unclassified");
    ConservationResult state = fixture.protocol.validate_conservation();
    check(state && state.counts.reserve_unclassified == 1,
          "reserve unclassified partition is visible");
    check(fixture.protocol.transition_reserve(
              operational.handle, ReserveConnectionState::Operational) ==
              PrimitiveStatus::Ok &&
              fixture.protocol.transition_reserve(operational.handle,
                                                  ReserveConnectionState::Output) ==
                  PrimitiveStatus::Ok &&
              fixture.protocol.transition_reserve(operational.handle,
                                                  ReserveConnectionState::Closing) ==
                  PrimitiveStatus::Ok &&
              fixture.protocol.release_reserve(operational.handle) ==
                  PrimitiveStatus::Ok,
          "operational reserve traverses output and closing");
    check(fixture.protocol.transition_reserve(
              operational.handle, ReserveConnectionState::Operational) ==
              PrimitiveStatus::StaleHandle,
          "released reserve generation is stale");

    const ReserveResult refusing = fixture.protocol.acquire_reserve();
    check(refusing &&
              fixture.protocol.transition_reserve(
                  refusing.handle, ReserveConnectionState::Refusing) ==
                  PrimitiveStatus::Ok,
          "reserve refusal partition is reachable");
    const ReserveResult retained = fixture.protocol.acquire_reserve();
    check(retained &&
              fixture.protocol.retain_reserve_failure(retained.handle) ==
                  PrimitiveStatus::Ok,
          "reserve failure is retained");
    state = fixture.protocol.validate_conservation();
    check(state && state.counts.reserve_refusing == 1 &&
              state.counts.reserve_failed_retained == 1,
          "refusing and failed-retained reserve terms conserve independently");

    TinyFixture exhausted(1);
    exhausted.exhaustion[static_cast<std::size_t>(OwnerDomain::RequestOwner)]
        .minimum_capacity = 3;
    check(exhausted.protocol.initialize() == PrimitiveStatus::Ok,
          "generation-max fixture initializes");
    RequestLease request;
    const PublicationResult publication = prepare_and_begin(exhausted, request);
    check(exhausted.protocol.commit_publication(publication.ticket) ==
              PrimitiveStatus::Ok,
          "generation-max request publishes");
    const SlotResult slot = exhausted.protocol.bind_slot(request.request);
    check(slot &&
              exhausted.protocol.transition_slot(request.request, slot.slot,
                                                 SlotState::ResetPending) ==
                  PrimitiveStatus::Ok &&
              exhausted.protocol.release_slot(request.request, slot.slot) ==
                  PrimitiveStatus::Ok,
          "generation-max slot retires");
    check(exhausted.protocol.validate_slot_handle(slot.slot) ==
              PrimitiveStatus::Exhausted,
          "slot at generation max exhausts without wrap");
    check(exhausted.protocol.mark_terminal_waiting(request.request) ==
              PrimitiveStatus::Ok,
          "generation-max request reaches terminal");
    const DeliveryLease delivery =
        exhausted.protocol.detach_to_delivery(request.request);
    check(static_cast<bool>(delivery), "generation-max request detaches");
    check(exhausted.protocol.recycle_admission_credit(delivery.admission_credit) ==
              PrimitiveStatus::Ok,
          "generation-max admission credit retires");
    check(exhausted.protocol.advance_delivery(request.request, DeliveryState::Emitting) ==
              PrimitiveStatus::Ok &&
              exhausted.protocol.advance_delivery(request.request,
                                                  DeliveryState::Reclaiming) ==
                  PrimitiveStatus::Ok &&
              exhausted.protocol.finish_delivery(request.request) ==
                  PrimitiveStatus::Ok,
          "generation-max request completes");
    state = exhausted.protocol.validate_conservation();
    check(state && state.counts.prepub_exhausted == 1 &&
              state.counts.accepted_exhausted == 1 &&
              state.counts.delivery_exhausted == 1 &&
              state.counts.owner_exhausted == 1 &&
              state.counts.general_exhausted == 1 &&
              state.counts.slot_exhausted == 1 &&
              state.counts.admission_exhausted == 1,
          "every retired generation-max domain remains exhausted");
    check(state.partitions_conserved && !state.qualifying,
          "exhaustion latches can never remain qualifying evidence");
    const ExhaustionEvidence& owner_evidence =
        exhausted.exhaustion[static_cast<std::size_t>(OwnerDomain::RequestOwner)];
    check(owner_evidence.exhaustion_latched && !owner_evidence.evidence_valid &&
              owner_evidence.below_minimum,
          "the owner domain latch records invalid evidence below minimum");
    check(exhausted.admission_gate[0].state ==
              AdmissionGateState::ClosedExhaustion,
          "the first latch closes admission");
    check(exhausted.protocol.validate_request_handle(request.request) ==
              PrimitiveStatus::Exhausted,
          "exhausted request is distinct from stale");
    check(exhausted.protocol.acquire_prepublication().status ==
              PrimitiveStatus::AdmissionClosed,
          "closed admission refuses new prepublication leases");

    const ReserveResult reserve = exhausted.protocol.acquire_reserve();
    check(reserve &&
              exhausted.protocol.transition_reserve(
                  reserve.handle, ReserveConnectionState::Closing) ==
                  PrimitiveStatus::Ok &&
              exhausted.protocol.release_reserve(reserve.handle) ==
                  PrimitiveStatus::Ok,
          "generation-max reserve retires");
    check(exhausted.protocol.transition_reserve(
              reserve.handle, ReserveConnectionState::Operational) ==
              PrimitiveStatus::Exhausted,
          "exhausted reserve cannot transition");
}

void conservation_corruption_is_detected() {
    Fixture state_corruption;
    check(state_corruption.protocol.initialize() == PrimitiveStatus::Ok,
          "corruption fixture initializes");
    state_corruption.accepted[0].state = static_cast<AcceptedState>(255);
    const ConservationResult invalid_state =
        state_corruption.protocol.validate_conservation();
    check(!invalid_state &&
              invalid_state.domain == ConservationDomain::Accepted,
          "unknown partition state fails validation");

    Fixture binding_corruption;
    check(binding_corruption.protocol.initialize() == PrimitiveStatus::Ok,
          "binding-corruption fixture initializes");
    const RequestLease lease = binding_corruption.protocol.acquire_prepublication();
    binding_corruption.prepublication[lease.prepublication.index]
        .request.owner_generation++;
    const ConservationResult invalid_binding =
        binding_corruption.protocol.validate_conservation();
    check(!invalid_binding &&
              invalid_binding.domain == ConservationDomain::CrossReference,
          "cross-generation binding corruption fails validation");

    Fixture history_corruption;
    check(history_corruption.protocol.initialize() == PrimitiveStatus::Ok,
          "history-corruption fixture initializes");
    history_corruption.histories[0].submit_published_total = 1;
    ConservationResult invalid_history =
        history_corruption.protocol.validate_conservation();
    check(!invalid_history &&
              invalid_history.domain == ConservationDomain::HistorySubmit &&
              invalid_history.expected == 0 && invalid_history.actual == 1,
          "a fabricated submit history is a typed violation");
    history_corruption.histories[0] = {};
    history_corruption.histories[0].submit_published_total = 1;
    history_corruption.histories[0].accepted_total = 1;
    invalid_history = history_corruption.protocol.validate_conservation();
    check(!invalid_history &&
              invalid_history.domain == ConservationDomain::HistoryAccepted,
          "a fabricated accepted history is a typed violation");
    history_corruption.histories[0] = {};
    history_corruption.histories[0].terminal_selected_total = 1;
    invalid_history = history_corruption.protocol.validate_conservation();
    check(!invalid_history &&
              invalid_history.domain ==
                  ConservationDomain::HistoryTerminalSelected,
          "a fabricated terminal-selected history is a typed violation");
    history_corruption.histories[0] = {};
    history_corruption.histories[0].terminal_published_total = 1;
    invalid_history = history_corruption.protocol.validate_conservation();
    check(!invalid_history &&
              invalid_history.domain ==
                  ConservationDomain::HistoryTerminalPublished,
          "a fabricated terminal-published history is a typed violation");
    history_corruption.histories[0] = {};
    check(static_cast<bool>(history_corruption.protocol.validate_conservation()),
          "restored histories validate again");

    Fixture evidence_corruption;
    check(evidence_corruption.protocol.initialize() == PrimitiveStatus::Ok,
          "evidence-corruption fixture initializes");
    evidence_corruption.exhaustion[0].evidence_valid = false;
    ConservationResult invalid_evidence =
        evidence_corruption.protocol.validate_conservation();
    check(!invalid_evidence &&
              invalid_evidence.domain == ConservationDomain::Exhaustion,
          "evidence-valid and latched must partition exactly");
    evidence_corruption.exhaustion[0].evidence_valid = true;
    evidence_corruption.exhaustion[0].below_minimum = true;
    invalid_evidence = evidence_corruption.protocol.validate_conservation();
    check(!invalid_evidence &&
              invalid_evidence.domain == ConservationDomain::Exhaustion,
          "a fabricated below-minimum flag is a typed violation");
    evidence_corruption.exhaustion[0].below_minimum = false;
    evidence_corruption.exhaustion[0].evidence_valid = false;
    evidence_corruption.exhaustion[0].exhaustion_latched = true;
    evidence_corruption.exhaustion[0].exhaustion_without_owner = true;
    invalid_evidence = evidence_corruption.protocol.validate_conservation();
    check(!invalid_evidence &&
              invalid_evidence.domain == ConservationDomain::Exhaustion,
          "a latch that left admission open is a typed violation");
}

// ---------------------------------------------------------------------------
// Bounded exhaustive stage/operation property: at every canonical lifecycle
// stage, every operation in the alphabet either succeeds or returns a typed
// status while leaving the entire owner state bit-for-bit unchanged, and
// conservation (including histories) holds after every attempt.
// ---------------------------------------------------------------------------

struct Snapshot {
    std::array<PrepublicationRecord, 2> prepublication;
    std::array<AcceptedRecord, 2> accepted;
    std::array<DeliveryRecord, 2> delivery;
    std::array<RequestOwnerRecord, 6> owners;
    std::array<GeneralConnectionRecord, 2> general;
    std::array<ReserveConnectionRecord, 2> reserve;
    std::array<AdmissionCreditRecord, 2> admission;
    std::array<SlotRecord, 1> slots;
    std::array<SchedulerCellRecord, 1> scheduler_cells;
    std::array<SubmitCellRecord, 2> submit_cells;
    std::array<AdmissionGateRecord, 1> admission_gate;
    std::array<ExhaustionEvidence, kDomainCount> exhaustion;
    std::array<OwnerHistories, 1> histories;
};

Snapshot save(const Fixture& fixture) {
    return {fixture.prepublication, fixture.accepted, fixture.delivery,
            fixture.owners, fixture.general, fixture.reserve, fixture.admission,
            fixture.slots, fixture.scheduler_cells, fixture.submit_cells,
            fixture.admission_gate, fixture.exhaustion, fixture.histories};
}

void restore(Fixture& fixture, const Snapshot& snapshot) {
    fixture.prepublication = snapshot.prepublication;
    fixture.accepted = snapshot.accepted;
    fixture.delivery = snapshot.delivery;
    fixture.owners = snapshot.owners;
    fixture.general = snapshot.general;
    fixture.reserve = snapshot.reserve;
    fixture.admission = snapshot.admission;
    fixture.slots = snapshot.slots;
    fixture.scheduler_cells = snapshot.scheduler_cells;
    fixture.submit_cells = snapshot.submit_cells;
    fixture.admission_gate = snapshot.admission_gate;
    fixture.exhaustion = snapshot.exhaustion;
    fixture.histories = snapshot.histories;
}

struct StateDigest {
    std::uint64_t value = 1469598103934665603ull;

    void feed(std::uint64_t term) noexcept {
        value = (value ^ term) * 1099511628211ull;
    }
    void feed_request(RequestHandle handle) noexcept {
        feed(handle.owner_index);
        feed(handle.owner_generation);
    }
};

std::uint64_t digest(const Fixture& fixture) {
    StateDigest d;
    for (const PrepublicationRecord& record : fixture.prepublication) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const AcceptedRecord& record : fixture.accepted) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const DeliveryRecord& record : fixture.delivery) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const RequestOwnerRecord& record : fixture.owners) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed(record.prepublication.index);
        d.feed(record.prepublication.generation);
        d.feed(record.accepted.index);
        d.feed(record.accepted.generation);
        d.feed(record.delivery.index);
        d.feed(record.delivery.generation);
        d.feed(record.connection.index);
        d.feed(record.connection.generation);
        d.feed(record.admission_credit.credit_index);
        d.feed(record.admission_credit.credit_generation);
        d.feed_request(record.admission_credit.request);
        d.feed(record.slot.slot_index);
        d.feed(record.slot.slot_generation);
        d.feed(record.scheduler_cell.index);
        d.feed(record.scheduler_cell.generation);
        d.feed(record.submit_cell.index);
        d.feed(record.submit_cell.generation);
    }
    for (const GeneralConnectionRecord& record : fixture.general) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const ReserveConnectionRecord& record : fixture.reserve) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
    }
    for (const AdmissionCreditRecord& record : fixture.admission) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
        d.feed(record.accepted.index);
        d.feed(record.accepted.generation);
    }
    for (const SlotRecord& record : fixture.slots) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const SchedulerCellRecord& record : fixture.scheduler_cells) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
    }
    for (const SubmitCellRecord& record : fixture.submit_cells) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
        d.feed(record.immutable_input_generation);
        d.feed(record.deadline_proof_generation);
    }
    for (const AdmissionGateRecord& record : fixture.admission_gate) {
        d.feed(record.generation);
        d.feed(static_cast<std::uint64_t>(record.state));
        d.feed_request(record.request);
        d.feed(static_cast<std::uint64_t>(record.close_domain));
    }
    for (const ExhaustionEvidence& record : fixture.exhaustion) {
        d.feed(record.configured_capacity);
        d.feed(record.effective_capacity);
        d.feed(record.minimum_capacity);
        d.feed(record.allowed_retirements);
        d.feed(static_cast<std::uint64_t>(record.evidence_valid));
        d.feed(static_cast<std::uint64_t>(record.exhaustion_latched));
        d.feed(static_cast<std::uint64_t>(record.exhaustion_without_owner));
        d.feed(static_cast<std::uint64_t>(record.exhaustion_owner_retained));
        d.feed(static_cast<std::uint64_t>(record.below_minimum));
    }
    const OwnerHistories& histories = fixture.histories[0];
    d.feed(histories.submit_published_total);
    d.feed(histories.admission_rejected_total);
    d.feed(histories.accepted_total);
    d.feed(histories.terminal_selected_total);
    d.feed(histories.terminal_published_total);
    d.feed(histories.terminal_consumed_total);
    d.feed(histories.delivery_consumed_total);
    return d.value;
}

struct ProbeContext {
    RequestLease lease{};
    PublicationResult publication{};
    SlotResult slot{};
    DeliveryLease delivery{};
};

constexpr std::size_t kProbeOpCount = 23;

PrimitiveStatus run_probe_op(Fixture& fixture, ProbeContext& ctx, std::size_t op) {
    switch (op) {
    case 0:
        return fixture.protocol.initialize();
    case 1:
        ctx.lease = fixture.protocol.acquire_prepublication();
        return ctx.lease.status;
    case 2:
        return fixture.protocol.advance_prepublication(
            ctx.lease.request, PrepublicationState::Preparing);
    case 3:
        return fixture.protocol.advance_prepublication(
            ctx.lease.request, PrepublicationState::Prepared);
    case 4:
        ctx.publication =
            fixture.protocol.begin_publication(make_authority(ctx.lease.request));
        return ctx.publication.status;
    case 5:
        return fixture.protocol.commit_publication(ctx.publication.ticket);
    case 6:
        return fixture.protocol.abort_publication(ctx.publication.ticket);
    case 7:
        return fixture.protocol.reject_published(ctx.lease.request);
    case 8:
        return fixture.protocol.queue_accepted(ctx.lease.request);
    case 9:
        ctx.slot = fixture.protocol.bind_slot(ctx.lease.request);
        return ctx.slot.status;
    case 10:
        return fixture.protocol.retire_scheduler_cell(ctx.lease.request);
    case 11:
        return fixture.protocol.transition_slot(ctx.lease.request, ctx.slot.slot,
                                                SlotState::TransferPending);
    case 12:
        return fixture.protocol.transition_slot(ctx.lease.request, ctx.slot.slot,
                                                SlotState::RequestOwned);
    case 13:
        return fixture.protocol.transition_slot(ctx.lease.request, ctx.slot.slot,
                                                SlotState::ResetPending);
    case 14:
        return fixture.protocol.release_slot(ctx.lease.request, ctx.slot.slot);
    case 15:
        return fixture.protocol.mark_terminal_waiting(ctx.lease.request);
    case 16:
        ctx.delivery = fixture.protocol.detach_to_delivery(ctx.lease.request);
        return ctx.delivery.status;
    case 17:
        return fixture.protocol.recycle_admission_credit(
            ctx.delivery.admission_credit);
    case 18:
        return fixture.protocol.advance_delivery(ctx.lease.request,
                                                 DeliveryState::Emitting);
    case 19:
        return fixture.protocol.advance_delivery(ctx.lease.request,
                                                 DeliveryState::Reclaiming);
    case 20:
        return fixture.protocol.finish_delivery(ctx.lease.request);
    case 21:
        return fixture.protocol.retain_accepted_failure(ctx.lease.request,
                                                        ctx.slot.slot);
    case 22:
        return fixture.protocol.retain_delivery_failure(ctx.lease.request);
    default:
        return PrimitiveStatus::Count;
    }
}

void bounded_exhaustive_stage_probe() {
    Fixture fixture;
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok,
          "probe fixture initializes");
    ProbeContext ctx;
    constexpr std::array<std::size_t, 18> canonical{
        1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    for (std::size_t stage = 0; stage <= canonical.size(); ++stage) {
        for (std::size_t op = 0; op < kProbeOpCount; ++op) {
            const Snapshot snapshot = save(fixture);
            const std::uint64_t before = digest(fixture);
            ProbeContext probe_ctx = ctx;
            const PrimitiveStatus status = run_probe_op(fixture, probe_ctx, op);
            const ConservationResult conserved =
                fixture.protocol.validate_conservation();
            if (!conserved || !conserved.qualifying) {
                std::fprintf(stderr,
                             "FAIL: conservation after stage %zu op %zu\n",
                             stage, op);
                ++failures;
            }
            if (status != PrimitiveStatus::Ok && digest(fixture) != before) {
                std::fprintf(stderr,
                             "FAIL: rejected op mutated state, stage %zu op %zu\n",
                             stage, op);
                ++failures;
            }
            restore(fixture, snapshot);
        }
        if (stage < canonical.size() &&
            run_probe_op(fixture, ctx, canonical[stage]) != PrimitiveStatus::Ok) {
            std::fprintf(stderr, "FAIL: canonical stage %zu did not advance\n",
                         stage);
            ++failures;
        }
    }
}

void operations_allocate_nothing() {
    Fixture fixture;
    check(fixture.protocol.initialize() == PrimitiveStatus::Ok, "fixture initializes");
    const std::size_t before = allocation_count;
    observe_allocations = true;

    RequestLease request;
    const PublicationResult publication = prepare_and_begin(fixture, request);
    fixture.protocol.commit_publication(publication.ticket);
    fixture.protocol.queue_accepted(request.request);
    const SlotResult slot = fixture.protocol.bind_slot(request.request);
    fixture.protocol.retire_scheduler_cell(request.request);
    fixture.protocol.transition_slot(request.request, slot.slot, SlotState::ResetPending);
    fixture.protocol.release_slot(request.request, slot.slot);
    fixture.protocol.mark_terminal_waiting(request.request);
    const DeliveryLease delivery = fixture.protocol.detach_to_delivery(request.request);
    fixture.protocol.recycle_admission_credit(delivery.admission_credit);
    fixture.protocol.advance_delivery(request.request, DeliveryState::Emitting);
    fixture.protocol.advance_delivery(request.request, DeliveryState::Reclaiming);
    fixture.protocol.validate_conservation();
    fixture.protocol.finish_delivery(request.request);
    const ReserveResult reserve = fixture.protocol.acquire_reserve();
    fixture.protocol.transition_reserve(reserve.handle, ReserveConnectionState::Closing);
    fixture.protocol.release_reserve(reserve.handle);

    observe_allocations = false;
    check(allocation_count == before,
          "all owner operations and validation are allocation-free");
}

} // namespace

int main() {
    topology_zero_max_and_overflow_are_typed();
    initialization_is_fixed_and_fail_closed();
    operation_results_carry_frozen_payload();
    compound_publication_owns_gate_inputs_and_deadline();
    typed_identities_are_exhaustively_classified();
    abort_reject_and_failed_retained_are_slot_coupled();
    reserve_partitions_and_exhaustion_never_resurrect();
    conservation_corruption_is_detected();
    bounded_exhaustive_stage_probe();
    operations_allocate_nothing();
    if (failures == 0) {
        std::printf("owner_protocol: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
