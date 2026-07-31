#include "tatara/service/admission.h"

#include <limits>

namespace tatara::service {

std::string_view rejection_code(Rejection rejection) {
    switch (rejection) {
    case Rejection::None:
        return "";
    case Rejection::InvalidRequest:
        return "tatara.invalid_request";
    case Rejection::DuplicateRequest:
        return "tatara.duplicate_request";
    case Rejection::ContextExceeded:
        return "tatara.context_exceeded";
    case Rejection::QueueFull:
        return "tatara.queue_full";
    case Rejection::NotReady:
        return "tatara.not_ready";
    case Rejection::Draining:
        return "tatara.draining";
    case Rejection::DeadlineExceeded:
        return "tatara.deadline_exceeded";
    case Rejection::DeadlineOverflow:
        return "tatara.deadline_overflow";
    }
    return "tatara.internal";
}

int rejection_http_status(Rejection rejection) {
    switch (rejection) {
    case Rejection::None:
        return 200;
    case Rejection::InvalidRequest:
    case Rejection::DuplicateRequest:
    case Rejection::ContextExceeded:
    case Rejection::DeadlineOverflow:
        return 400;
    // 429 and 503 are distinct because the operator's response differs: shed
    // load versus wait for readiness.
    case Rejection::QueueFull:
        return 429;
    case Rejection::NotReady:
    case Rejection::Draining:
        return 503;
    case Rejection::DeadlineExceeded:
        return 504;
    }
    return 500;
}

Scheduler::Scheduler(const ServiceConfiguration& configuration, bool ready)
    : configuration_(configuration), ready_(ready), queue_(configuration_.queue_depth),
      running_(configuration_.max_concurrent_requests),
      events_(static_cast<std::size_t>(configuration_.queue_depth) +
              configuration_.max_concurrent_requests) {}

void Scheduler::set_ready(bool ready) {
    ready_ = ready;
}

bool Scheduler::ready() const {
    return ready_;
}

void Scheduler::begin_drain() {
    draining_ = true;
}

bool Scheduler::draining() const {
    return draining_;
}

AdmissionOutcome Scheduler::admit(const Request& request, Milliseconds now_ms) {
    if (!ready_) {
        return {false, false, kNoSlot, Rejection::NotReady};
    }
    if (draining_) {
        return {false, false, kNoSlot, Rejection::Draining};
    }
    if (request.id == 0) {
        return {false, false, kNoSlot, Rejection::InvalidRequest};
    }
    if (contains(request.id)) {
        return {false, false, kNoSlot, Rejection::DuplicateRequest};
    }
    // The whole conversation must fit, not just the prompt: admitting a request
    // that cannot finish only defers the failure to a worse moment.
    const std::uint64_t total =
        static_cast<std::uint64_t>(request.prompt_tokens) + request.max_output_tokens;
    if (total > configuration_.max_context_tokens) {
        return {false, false, kNoSlot, Rejection::ContextExceeded};
    }

    const Milliseconds duration = request.deadline_ms != 0
                                      ? request.deadline_ms
                                      : configuration_.request_deadline_milliseconds;
    Milliseconds expires_at = std::numeric_limits<Milliseconds>::max();
    if (duration != 0) {
        if (now_ms > std::numeric_limits<Milliseconds>::max() - duration) {
            return {false, false, kNoSlot, Rejection::DeadlineOverflow};
        }
        expires_at = now_ms + duration;
    }

    const Admitted admitted{request.id, now_ms, expires_at};
    const std::uint32_t slot = free_slot();
    if (queue_size_ == 0 && slot != kNoSlot) {
        start_in_slot(admitted, slot);
        return {true, true, slot, Rejection::None};
    }
    if (queue_size_ >= queue_.size()) {
        return {false, false, kNoSlot, Rejection::QueueFull};
    }
    push_queue(admitted);
    return {true, false, kNoSlot, Rejection::None};
}

std::span<const SchedulerEvent> Scheduler::expire(Milliseconds now_ms) {
    reset_events();
    std::size_t offset = 0;
    while (offset < queue_size_) {
        const Admitted entry = queue_at(offset);
        if (expired(entry, now_ms)) {
            add_event(entry.id, SchedulerEventKind::QueuedDeadlineExceeded, kNoSlot);
            remove_queue_at(offset);
        } else {
            ++offset;
        }
    }
    for (std::uint32_t slot = 0; slot < running_.size(); ++slot) {
        Running& entry = running_[slot];
        if (entry.occupied && entry.stop_reason == StopReason::None &&
            expired(entry.request, now_ms)) {
            entry.stop_reason = StopReason::Deadline;
            add_event(entry.request.id, SchedulerEventKind::RunningDeadlineRequested, slot);
        }
    }
    return events();
}

std::span<const SchedulerEvent> Scheduler::start_ready(Milliseconds now_ms) {
    reset_events();
    std::size_t offset = 0;
    while (offset < queue_size_) {
        const Admitted entry = queue_at(offset);
        if (expired(entry, now_ms)) {
            add_event(entry.id, SchedulerEventKind::QueuedDeadlineExceeded, kNoSlot);
            remove_queue_at(offset);
        } else {
            ++offset;
        }
    }
    while (queue_size_ != 0) {
        const std::uint32_t slot = free_slot();
        if (slot == kNoSlot) {
            break;
        }
        const Admitted entry = pop_queue();
        start_in_slot(entry, slot);
        add_event(entry.id, SchedulerEventKind::Started, slot);
    }
    return events();
}

std::span<const SchedulerEvent> Scheduler::cancel(RequestId id) {
    reset_events();
    for (std::size_t offset = 0; offset < queue_size_; ++offset) {
        if (queue_at(offset).id == id) {
            remove_queue_at(offset);
            add_event(id, SchedulerEventKind::QueuedCancelled, kNoSlot);
            return events();
        }
    }
    for (std::uint32_t slot = 0; slot < running_.size(); ++slot) {
        Running& entry = running_[slot];
        if (entry.occupied && entry.request.id == id) {
            if (entry.stop_reason == StopReason::None) {
                entry.stop_reason = StopReason::Cancelled;
                add_event(id, SchedulerEventKind::RunningCancellationRequested, slot);
            }
            return events();
        }
    }
    return events();
}

bool Scheduler::retire(RequestId id) {
    reset_events();
    for (Running& entry : running_) {
        if (entry.occupied && entry.request.id == id) {
            entry = Running{};
            --running_count_;
            return true;
        }
    }
    return false;
}

std::size_t Scheduler::running() const {
    return running_count_;
}

std::size_t Scheduler::queued() const {
    return queue_size_;
}

std::size_t Scheduler::slots() const {
    return running_.size();
}

StopReason Scheduler::stop_reason(RequestId id) const {
    for (const Running& entry : running_) {
        if (entry.occupied && entry.request.id == id) {
            return entry.stop_reason;
        }
    }
    return StopReason::None;
}

double Scheduler::saturation() const {
    if (configuration_.queue_depth == 0) {
        return 0.0;
    }
    return static_cast<double>(queue_size_) / static_cast<double>(configuration_.queue_depth);
}

void Scheduler::reset_events() {
    event_count_ = 0;
}

void Scheduler::add_event(RequestId id, SchedulerEventKind kind, std::uint32_t slot) {
    if (event_count_ < events_.size()) {
        events_[event_count_++] = {id, kind, slot};
    }
}

std::span<const SchedulerEvent> Scheduler::events() const {
    return {events_.data(), event_count_};
}

bool Scheduler::contains(RequestId id) const {
    for (std::size_t offset = 0; offset < queue_size_; ++offset) {
        if (queue_at(offset).id == id) {
            return true;
        }
    }
    for (const Running& entry : running_) {
        if (entry.occupied && entry.request.id == id) {
            return true;
        }
    }
    return false;
}

bool Scheduler::expired(const Admitted& request, Milliseconds now_ms) const {
    return request.expires_at_ms != std::numeric_limits<Milliseconds>::max() &&
           request.expires_at_ms <= now_ms;
}

std::uint32_t Scheduler::free_slot() const {
    for (std::uint32_t slot = 0; slot < running_.size(); ++slot) {
        if (!running_[slot].occupied) {
            return slot;
        }
    }
    return kNoSlot;
}

void Scheduler::start_in_slot(const Admitted& request, std::uint32_t slot) {
    running_[slot] = {true, request, StopReason::None};
    ++running_count_;
}

void Scheduler::push_queue(const Admitted& request) {
    const std::size_t tail = (queue_head_ + queue_size_) % queue_.size();
    queue_[tail] = request;
    ++queue_size_;
}

Admitted Scheduler::pop_queue() {
    const Admitted entry = queue_[queue_head_];
    queue_head_ = (queue_head_ + 1) % queue_.size();
    --queue_size_;
    if (queue_size_ == 0) {
        queue_head_ = 0;
    }
    return entry;
}

const Admitted& Scheduler::queue_at(std::size_t offset) const {
    return queue_[(queue_head_ + offset) % queue_.size()];
}

void Scheduler::remove_queue_at(std::size_t offset) {
    for (std::size_t current = offset; current + 1 < queue_size_; ++current) {
        const std::size_t destination = (queue_head_ + current) % queue_.size();
        const std::size_t source = (queue_head_ + current + 1) % queue_.size();
        queue_[destination] = queue_[source];
    }
    --queue_size_;
    if (queue_size_ == 0) {
        queue_head_ = 0;
    }
}

} // namespace tatara::service
