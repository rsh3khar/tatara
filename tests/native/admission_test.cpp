#include "tatara/service/admission.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

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

ServiceConfiguration configuration(std::uint32_t slots = 2, std::uint32_t queue = 3,
                                   std::uint32_t deadline = 100) {
    ServiceConfiguration value;
    value.max_context_tokens = 1000;
    value.max_concurrent_requests = slots;
    value.queue_depth = queue;
    value.request_deadline_milliseconds = deadline;
    return value;
}

Request request(RequestId id, std::uint32_t prompt = 10, std::uint32_t output = 10,
                Milliseconds deadline = 0) {
    return Request{id, prompt, output, deadline};
}

bool is_event(const SchedulerEvent& event, RequestId id, SchedulerEventKind kind,
              std::uint32_t slot = kNoSlot) {
    return event.id == id && event.kind == kind && event.slot == slot;
}

void readiness_identity_and_context_are_typed() {
    Scheduler unready(configuration(), false);
    const auto not_ready = unready.admit(request(1), 0);
    check(!not_ready.accepted && not_ready.rejection == Rejection::NotReady,
          "unready service refuses with typed result");

    Scheduler scheduler(configuration(), true);
    check(scheduler.admit(request(0), 0).rejection == Rejection::InvalidRequest,
          "zero request id is invalid");
    check(scheduler.admit(request(1, 900, 200), 0).rejection == Rejection::ContextExceeded,
          "whole conversation must fit");

    const auto first = scheduler.admit(request(1), 0);
    check(first.accepted && first.started && first.slot == 0, "first request starts in slot zero");
    check(scheduler.admit(request(1), 0).rejection == Rejection::DuplicateRequest,
          "duplicate running id is refused");
}

void idle_slots_start_without_a_queue() {
    Scheduler scheduler(configuration(2, 0), true);
    const auto first = scheduler.admit(request(1), 0);
    const auto second = scheduler.admit(request(2), 0);
    const auto full = scheduler.admit(request(3), 0);

    check(first.started && first.slot == 0, "zero-depth queue starts first idle slot");
    check(second.started && second.slot == 1, "zero-depth queue starts second idle slot");
    check(!full.accepted && full.rejection == Rejection::QueueFull,
          "zero-depth queue refuses only when slots are occupied");
    check(scheduler.saturation() == 0.0, "zero-depth queue has no waiting saturation");
}

void fifo_queue_and_slots_are_bounded() {
    Scheduler scheduler(configuration(), true);
    check(scheduler.admit(request(1), 0).started, "one starts");
    check(scheduler.admit(request(2), 0).started, "two starts");
    check(!scheduler.admit(request(3), 0).started, "three queues");
    check(!scheduler.admit(request(4), 0).started, "four queues");
    check(!scheduler.admit(request(5), 0).started, "five queues");
    check(scheduler.admit(request(6), 0).rejection == Rejection::QueueFull,
          "queue refuses above fixed depth");
    check(scheduler.running() == 2 && scheduler.queued() == 3 && scheduler.slots() == 2,
          "running and queue bounds hold");
    check(scheduler.saturation() == 1.0, "full queue reports full saturation");

    check(scheduler.retire(1), "first slot retires");
    const auto first_start = scheduler.start_ready(0);
    check(first_start.size() == 1 &&
              is_event(first_start.front(), 3, SchedulerEventKind::Started, 0),
          "FIFO head starts in first free slot");
    check(scheduler.retire(2), "second slot retires");
    const auto second_start = scheduler.start_ready(0);
    check(second_start.size() == 1 &&
              is_event(second_start.front(), 4, SchedulerEventKind::Started, 1),
          "FIFO order survives a second retirement");
}

void deadlines_are_disabled_checked_and_do_not_release_running_slots() {
    Scheduler disabled(configuration(1, 1, 0), true);
    check(disabled.admit(request(1), 5).started, "disabled-deadline request starts");
    check(disabled.expire(std::numeric_limits<Milliseconds>::max()).empty(),
          "zero configured deadline remains disabled");

    Scheduler overflow(configuration(), true);
    const auto too_late =
        overflow.admit(request(1, 10, 10, 50), std::numeric_limits<Milliseconds>::max() - 40);
    check(too_late.rejection == Rejection::DeadlineOverflow,
          "deadline addition overflow is a typed refusal");

    Scheduler scheduler(configuration(1, 2), true);
    scheduler.admit(request(1, 10, 10, 50), 0);
    scheduler.admit(request(2, 10, 10, 500), 0);
    scheduler.admit(request(3, 10, 10, 25), 0);
    const auto expired = scheduler.expire(50);
    check(expired.size() == 2, "running and queued deadline each produce one event");
    check(is_event(expired[0], 3, SchedulerEventKind::QueuedDeadlineExceeded),
          "queued expiry is removed");
    check(is_event(expired[1], 1, SchedulerEventKind::RunningDeadlineRequested, 0),
          "running expiry requests a stop");
    check(scheduler.running() == 1 && scheduler.queued() == 1,
          "running expiry retains slot while queued expiry leaves");
    check(scheduler.stop_reason(1) == StopReason::Deadline, "running deadline reason is retained");
    check(scheduler.expire(60).empty(), "running deadline event is not repeated");
    check(scheduler.start_ready(60).empty(), "waiting work cannot reuse an unsafe slot");
    check(scheduler.retire(1), "execution owner safely retires expired slot");
    const auto started = scheduler.start_ready(60);
    check(started.size() == 1 && is_event(started[0], 2, SchedulerEventKind::Started, 0),
          "waiting work starts only after safe retirement");
}

void cancellation_is_idempotent_and_preserves_running_ownership() {
    Scheduler scheduler(configuration(1, 2), true);
    scheduler.admit(request(1), 0);
    scheduler.admit(request(2), 0);
    const auto queued = scheduler.cancel(2);
    check(queued.size() == 1 && is_event(queued[0], 2, SchedulerEventKind::QueuedCancelled),
          "queued cancellation removes immediately");

    const auto running = scheduler.cancel(1);
    check(running.size() == 1 &&
              is_event(running[0], 1, SchedulerEventKind::RunningCancellationRequested, 0),
          "running cancellation requests stop in its slot");
    check(scheduler.running() == 1 && scheduler.stop_reason(1) == StopReason::Cancelled,
          "cancelled running request retains ownership");
    check(scheduler.cancel(1).empty(), "repeated cancellation is idempotent");
    check(!scheduler.retire(99), "unknown retirement is false");
    check(scheduler.retire(1) && scheduler.running() == 0, "safe retirement releases slot");
}

void drain_refuses_new_work_but_finishes_accepted_work() {
    Scheduler scheduler(configuration(1, 1), true);
    scheduler.admit(request(1), 0);
    scheduler.admit(request(2), 0);
    scheduler.begin_drain();
    check(scheduler.draining(), "drain state is observable");
    check(scheduler.admit(request(3), 0).rejection == Rejection::Draining,
          "drain refuses new admission");
    check(scheduler.retire(1), "running accepted work can retire during drain");
    const auto started = scheduler.start_ready(0);
    check(started.size() == 1 && is_event(started[0], 2, SchedulerEventKind::Started, 0),
          "queued accepted work can start during drain");
}

void every_operation_after_construction_allocates_nothing() {
    Scheduler scheduler(configuration(2, 3), true);
    const std::size_t before = allocation_count;
    track_allocations = true;

    scheduler.admit(request(1), 0);
    scheduler.admit(request(2), 0);
    scheduler.admit(request(3), 0);
    scheduler.admit(request(4, 10, 10, 5), 0);
    scheduler.cancel(3);
    scheduler.expire(5);
    scheduler.retire(1);
    scheduler.start_ready(5);
    scheduler.begin_drain();
    scheduler.retire(2);

    track_allocations = false;
    check(allocation_count == before, "admit/start/expire/cancel/retire/drain allocate nothing");
}

void rejection_codes_and_statuses_are_stable() {
    check(rejection_http_status(Rejection::QueueFull) == 429, "queue full is 429");
    check(rejection_http_status(Rejection::NotReady) == 503, "not ready is 503");
    check(rejection_http_status(Rejection::DeadlineExceeded) == 504, "elapsed deadline is 504");
    check(rejection_http_status(Rejection::DeadlineOverflow) == 400, "overflow is client error");
    check(rejection_code(Rejection::DuplicateRequest) == "tatara.duplicate_request",
          "duplicate identity has stable code");
    check(rejection_code(Rejection::Draining) == "tatara.draining", "draining has stable code");
}

} // namespace

int main() {
    readiness_identity_and_context_are_typed();
    idle_slots_start_without_a_queue();
    fifo_queue_and_slots_are_bounded();
    deadlines_are_disabled_checked_and_do_not_release_running_slots();
    cancellation_is_idempotent_and_preserves_running_ownership();
    drain_refuses_new_work_but_finishes_accepted_work();
    every_operation_after_construction_allocates_nothing();
    rejection_codes_and_statuses_are_stable();
    if (failures == 0) {
        std::printf("admission: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
