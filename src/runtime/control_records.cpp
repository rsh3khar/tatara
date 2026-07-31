#include "tatara/runtime/control_records.h"

#include <limits>

namespace tatara::runtime {
namespace {

constexpr ControlEpoch kMaximumEpoch = std::numeric_limits<ControlEpoch>::max();

ControlStatus compare_expected(ControlEpoch expected, ControlEpoch current) noexcept {
    if (expected < current) {
        return ControlStatus::Stale;
    }
    if (expected > current) {
        return ControlStatus::Conflict;
    }
    return ControlStatus::Applied;
}

bool valid_fatal_reason(FatalReason reason) noexcept {
    switch (reason) {
    case FatalReason::WorkerFailure:
    case FatalReason::OwnershipInvariant:
    case FatalReason::TimeInvalid:
    case FatalReason::CounterExhausted:
        return true;
    case FatalReason::None:
        return false;
    }
    return false;
}

} // namespace

WakeRecord::WakeRecord(ControlParty publisher, ControlParty consumer,
                       ControlEpoch initial_epoch) noexcept
    : publisher_(publisher), consumer_(consumer), initial_epoch_(initial_epoch),
      published_(initial_epoch), acknowledged_(initial_epoch) {}

ControlStatus WakeRecord::publish(ControlParty writer, ControlEpoch expected_epoch) noexcept {
    if (writer != publisher_) {
        return ControlStatus::WrongWriter;
    }
    if (failed_retained_.load(std::memory_order_acquire)) {
        return ControlStatus::FailedRetained;
    }
    const ControlEpoch published = published_.load(std::memory_order_relaxed);
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_acquire);
    if (acknowledged > published) {
        return ControlStatus::Corrupt;
    }
    const ControlStatus comparison = compare_expected(expected_epoch, published);
    if (comparison != ControlStatus::Applied) {
        return comparison;
    }
    if (published == kMaximumEpoch) {
        exhausted_.store(true, std::memory_order_release);
        return ControlStatus::Exhausted;
    }
    published_.store(published + 1, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus WakeRecord::observe(ControlParty reader, ControlEpoch last_seen,
                                  ControlEpoch& observed) const noexcept {
    if (reader != consumer_) {
        return ControlStatus::WrongWriter;
    }
    if (failed_retained_.load(std::memory_order_acquire)) {
        observed = published_.load(std::memory_order_acquire);
        return ControlStatus::FailedRetained;
    }
    const ControlEpoch published = published_.load(std::memory_order_acquire);
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_acquire);
    if (acknowledged > published) {
        return ControlStatus::Corrupt;
    }
    observed = published;
    if (last_seen > published) {
        return ControlStatus::Conflict;
    }
    if (last_seen == published) {
        return exhausted_.load(std::memory_order_acquire) ? ControlStatus::Exhausted
                                                          : ControlStatus::Idle;
    }
    return ControlStatus::Observed;
}

ControlStatus WakeRecord::acknowledge(ControlParty writer, ControlEpoch observed_epoch) noexcept {
    if (writer != consumer_) {
        return ControlStatus::WrongWriter;
    }
    if (failed_retained_.load(std::memory_order_acquire)) {
        return ControlStatus::FailedRetained;
    }
    const ControlEpoch published = published_.load(std::memory_order_acquire);
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_relaxed);
    if (acknowledged > published) {
        return ControlStatus::Corrupt;
    }
    if (observed_epoch < acknowledged) {
        return ControlStatus::Stale;
    }
    if (observed_epoch == acknowledged) {
        return ControlStatus::Repeated;
    }
    if (observed_epoch > published) {
        return ControlStatus::Conflict;
    }
    acknowledged_.store(observed_epoch, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus WakeRecord::retain_failed(ControlParty writer, ControlEpoch observed_epoch) noexcept {
    if (writer != consumer_) {
        return ControlStatus::WrongWriter;
    }
    if (exhausted_.load(std::memory_order_acquire)) {
        return ControlStatus::Exhausted;
    }
    if (failed_retained_.load(std::memory_order_acquire)) {
        const ControlEpoch retained = failed_retained_epoch_.load(std::memory_order_relaxed);
        return retained == observed_epoch ? ControlStatus::Repeated
                                          : compare_expected(observed_epoch, retained);
    }
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_relaxed);
    const ControlEpoch published = published_.load(std::memory_order_acquire);
    if (acknowledged > published) {
        return ControlStatus::Corrupt;
    }
    if (observed_epoch < published) {
        return ControlStatus::Stale;
    }
    if (observed_epoch > published || observed_epoch == acknowledged) {
        return ControlStatus::Conflict;
    }
    failed_retained_epoch_.store(observed_epoch, std::memory_order_relaxed);
    failed_retained_.store(true, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus WakeRecord::state() const noexcept {
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_acquire);
    const ControlEpoch published = published_.load(std::memory_order_acquire);
    if (acknowledged > published) {
        return ControlStatus::Corrupt;
    }
    if (failed_retained_.load(std::memory_order_acquire)) {
        return ControlStatus::FailedRetained;
    }
    if (published != acknowledged) {
        return ControlStatus::Pending;
    }
    if (exhausted_.load(std::memory_order_acquire)) {
        return ControlStatus::Exhausted;
    }
    return published == initial_epoch_ ? ControlStatus::Idle : ControlStatus::Acknowledged;
}

WakeEvidenceSnapshot WakeRecord::evidence() const noexcept {
    WakeEvidenceSnapshot result;
    const ControlEpoch acknowledged = acknowledged_.load(std::memory_order_acquire);
    const ControlEpoch published = published_.load(std::memory_order_acquire);
    const bool failed_retained = failed_retained_.load(std::memory_order_acquire);
    const bool exhausted = exhausted_.load(std::memory_order_acquire);
    result.published_epoch = published;
    result.acknowledged_epoch = acknowledged;
    result.exhaustion_latched = exhausted;
    result.exhaustion_without_owner = exhausted;
    if (acknowledged > published || acknowledged < initial_epoch_ || published < initial_epoch_) {
        return result;
    }
    result.published_total = published - initial_epoch_;
    result.acknowledged_total = acknowledged - initial_epoch_;
    const std::uint64_t unacknowledged = published - acknowledged;
    if (failed_retained) {
        result.failed_retained_now = unacknowledged;
        result.failed_retained_total = unacknowledged;
        result.wake_failed_retained = 1;
    } else if (exhausted) {
        result.pending_count = unacknowledged;
        result.wake_exhausted = 1;
    } else if (unacknowledged != 0) {
        result.pending_count = unacknowledged;
        result.wake_pending = 1;
    } else if (result.published_total == 0) {
        result.wake_idle = 1;
    } else {
        result.wake_acknowledged = 1;
    }
    const std::uint16_t state_sum = static_cast<std::uint16_t>(result.wake_idle) +
                                    static_cast<std::uint16_t>(result.wake_pending) +
                                    static_cast<std::uint16_t>(result.wake_acknowledged) +
                                    static_cast<std::uint16_t>(result.wake_failed_retained) +
                                    static_cast<std::uint16_t>(result.wake_exhausted);
    result.conservation_holds =
        state_sum == 1 && result.published_total == result.pending_count +
                                                        result.acknowledged_total +
                                                        result.failed_retained_now;
    result.evidence_valid = result.conservation_holds && !exhausted;
    return result;
}

ControlEpoch WakeRecord::published_epoch() const noexcept {
    return published_.load(std::memory_order_acquire);
}

ControlEpoch WakeRecord::acknowledged_epoch() const noexcept {
    return acknowledged_.load(std::memory_order_acquire);
}

TimeRecord::TimeRecord(ControlParty publisher, ControlParty consumer, ControlEpoch epoch_id,
                       std::uint64_t initial_sequence) noexcept
    : publisher_(publisher), consumer_(consumer), epoch_id_(epoch_id), sequence_(initial_sequence) {
    if ((initial_sequence & 1U) != 0) {
        publisher_state_.store(PublisherState::Corrupt, std::memory_order_relaxed);
    }
}

ControlStatus TimeRecord::publish(ControlParty writer, std::uint64_t monotonic_units) noexcept {
    if (writer != publisher_) {
        return ControlStatus::WrongWriter;
    }
    if (epoch_id_ == 0) {
        return ControlStatus::InvalidEpoch;
    }
    const PublisherState state = publisher_state_.load(std::memory_order_acquire);
    if (state == PublisherState::Stopped) {
        return ControlStatus::Stopped;
    }
    if (state == PublisherState::Exhausted) {
        return ControlStatus::Exhausted;
    }
    if (state == PublisherState::Corrupt) {
        return ControlStatus::Corrupt;
    }
    if (published_.load(std::memory_order_acquire)) {
        const std::uint64_t previous = monotonic_units_.load(std::memory_order_relaxed);
        if (monotonic_units < previous) {
            return ControlStatus::Stale;
        }
        if (monotonic_units == previous) {
            return ControlStatus::Repeated;
        }
    }

    const std::uint64_t sequence = sequence_.load(std::memory_order_relaxed);
    if ((sequence & 1U) != 0) {
        publisher_state_.store(PublisherState::Corrupt, std::memory_order_release);
        return ControlStatus::Corrupt;
    }
    if (sequence > std::numeric_limits<std::uint64_t>::max() - 2) {
        publisher_state_.store(PublisherState::Exhausted, std::memory_order_release);
        return ControlStatus::Exhausted;
    }
    sequence_.store(sequence + 1, std::memory_order_seq_cst);
    monotonic_units_.store(monotonic_units, std::memory_order_relaxed);
    published_.store(true, std::memory_order_release);
    sequence_.store(sequence + 2, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus TimeRecord::read(ControlParty reader, ControlEpoch expected_epoch,
                               std::uint64_t observer_units, std::uint64_t maximum_age,
                               TimeSnapshot& snapshot) const noexcept {
    if (reader != consumer_) {
        return ControlStatus::WrongWriter;
    }
    const PublisherState state = publisher_state_.load(std::memory_order_acquire);
    if (state == PublisherState::Stopped) {
        return ControlStatus::Stopped;
    }
    if (state == PublisherState::Exhausted) {
        return ControlStatus::Exhausted;
    }
    if (state == PublisherState::Corrupt) {
        return ControlStatus::Corrupt;
    }
    if (expected_epoch == 0 || expected_epoch != epoch_id_) {
        return ControlStatus::ForeignEpoch;
    }
    if (!published_.load(std::memory_order_acquire)) {
        return ControlStatus::Idle;
    }

    for (int attempt = 0; attempt != 3; ++attempt) {
        const std::uint64_t before = sequence_.load(std::memory_order_acquire);
        if ((before & 1U) != 0) {
            continue;
        }
        const std::uint64_t units = monotonic_units_.load(std::memory_order_relaxed);
        const std::uint64_t after = sequence_.load(std::memory_order_acquire);
        if (before == after && (after & 1U) == 0) {
            snapshot = {epoch_id_, units, after};
            if (observer_units < units || observer_units - units > maximum_age) {
                return ControlStatus::Stale;
            }
            return ControlStatus::Observed;
        }
    }
    return ControlStatus::Busy;
}

ControlStatus TimeRecord::stop(ControlParty writer) noexcept {
    if (writer != publisher_) {
        return ControlStatus::WrongWriter;
    }
    const PublisherState state = publisher_state_.load(std::memory_order_acquire);
    if (state == PublisherState::Stopped) {
        return ControlStatus::Repeated;
    }
    if (state == PublisherState::Exhausted) {
        return ControlStatus::Exhausted;
    }
    if (state == PublisherState::Corrupt) {
        return ControlStatus::Corrupt;
    }
    publisher_state_.store(PublisherState::Stopped, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus TimeRecord::state() const noexcept {
    switch (publisher_state_.load(std::memory_order_acquire)) {
    case PublisherState::Running:
        return published_.load(std::memory_order_acquire) ? ControlStatus::Observed
                                                          : ControlStatus::Idle;
    case PublisherState::Stopped:
        return ControlStatus::Stopped;
    case PublisherState::Exhausted:
        return ControlStatus::Exhausted;
    case PublisherState::Corrupt:
        return ControlStatus::Corrupt;
    }
    return ControlStatus::Corrupt;
}

DrainRecord::DrainRecord(ControlParty publisher, ControlParty first_consumer,
                         ControlParty second_consumer, ControlEpoch initial_epoch) noexcept
    : publisher_(publisher), first_consumer_(first_consumer), second_consumer_(second_consumer),
      epoch_(initial_epoch), first_ack_(initial_epoch), second_ack_(initial_epoch) {}

ControlStatus DrainRecord::request(ControlParty writer, ControlEpoch expected_epoch) noexcept {
    if (writer != publisher_) {
        return ControlStatus::WrongWriter;
    }
    const ControlEpoch epoch = epoch_.load(std::memory_order_relaxed);
    const ControlEpoch first = first_ack_.load(std::memory_order_acquire);
    const ControlEpoch second = second_ack_.load(std::memory_order_acquire);
    if (first > epoch || second > epoch) {
        return ControlStatus::Corrupt;
    }
    const ControlStatus comparison = compare_expected(expected_epoch, epoch);
    if (comparison != ControlStatus::Applied) {
        return comparison;
    }
    if (first != epoch || second != epoch) {
        return ControlStatus::Repeated;
    }
    if (epoch == kMaximumEpoch) {
        exhausted_.store(true, std::memory_order_release);
        return ControlStatus::Exhausted;
    }
    epoch_.store(epoch + 1, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus DrainRecord::observe(ControlParty reader, ControlEpoch last_seen,
                                   ControlEpoch& observed) const noexcept {
    if (reader != first_consumer_ && reader != second_consumer_) {
        return ControlStatus::WrongWriter;
    }
    const ControlEpoch epoch = epoch_.load(std::memory_order_acquire);
    observed = epoch;
    if (last_seen > epoch) {
        return ControlStatus::Conflict;
    }
    if (last_seen == epoch) {
        return exhausted_.load(std::memory_order_acquire) ? ControlStatus::Exhausted
                                                          : ControlStatus::Idle;
    }
    return ControlStatus::Observed;
}

ControlStatus DrainRecord::acknowledge(ControlParty writer, ControlEpoch drain_epoch) noexcept {
    std::atomic<ControlEpoch>* acknowledgment = nullptr;
    if (writer == first_consumer_) {
        acknowledgment = &first_ack_;
    } else if (writer == second_consumer_) {
        acknowledgment = &second_ack_;
    } else {
        return ControlStatus::WrongWriter;
    }

    const ControlEpoch epoch = epoch_.load(std::memory_order_acquire);
    const ControlEpoch current = acknowledgment->load(std::memory_order_relaxed);
    if (current > epoch) {
        return ControlStatus::Corrupt;
    }
    if (drain_epoch < current) {
        return ControlStatus::Stale;
    }
    if (drain_epoch == current) {
        return ControlStatus::Repeated;
    }
    if (drain_epoch > epoch) {
        return ControlStatus::Conflict;
    }
    acknowledgment->store(drain_epoch, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus DrainRecord::state() const noexcept {
    const ControlEpoch epoch = epoch_.load(std::memory_order_acquire);
    const ControlEpoch first = first_ack_.load(std::memory_order_acquire);
    const ControlEpoch second = second_ack_.load(std::memory_order_acquire);
    if (first > epoch || second > epoch) {
        return ControlStatus::Corrupt;
    }
    if (first != epoch || second != epoch) {
        return ControlStatus::Pending;
    }
    if (exhausted_.load(std::memory_order_acquire)) {
        return ControlStatus::Exhausted;
    }
    return epoch == 0 ? ControlStatus::Idle : ControlStatus::Acknowledged;
}

ControlEpoch DrainRecord::epoch() const noexcept {
    return epoch_.load(std::memory_order_acquire);
}

FatalRecord::FatalRecord(ControlParty publisher, ControlParty consumer,
                         ControlEpoch initial_epoch) noexcept
    : publisher_(publisher), consumer_(consumer), initial_epoch_(initial_epoch) {}

ControlStatus FatalRecord::publish(ControlParty writer, ControlEpoch expected_epoch,
                                   FatalReason reason) noexcept {
    if (writer != publisher_) {
        return ControlStatus::WrongWriter;
    }
    if (!valid_fatal_reason(reason)) {
        return ControlStatus::InvalidValue;
    }
    const PublicationState state = publication_state_.load(std::memory_order_acquire);
    if (state == PublicationState::Published) {
        const ControlEpoch epoch = published_epoch_.load(std::memory_order_relaxed);
        const FatalReason current_reason = reason_.load(std::memory_order_relaxed);
        const ControlStatus comparison = compare_expected(expected_epoch, epoch);
        if (comparison != ControlStatus::Applied) {
            return comparison;
        }
        return current_reason == reason ? ControlStatus::Repeated : ControlStatus::Conflict;
    }
    if (state == PublicationState::Exhausted) {
        return ControlStatus::Exhausted;
    }
    const ControlStatus comparison = compare_expected(expected_epoch, initial_epoch_);
    if (comparison != ControlStatus::Applied) {
        return comparison;
    }
    if (initial_epoch_ == kMaximumEpoch) {
        exhausted_attempt_reason_.store(reason, std::memory_order_relaxed);
        publication_state_.store(PublicationState::Exhausted, std::memory_order_release);
        return ControlStatus::Exhausted;
    }
    reason_.store(reason, std::memory_order_relaxed);
    published_epoch_.store(initial_epoch_ + 1, std::memory_order_relaxed);
    publication_state_.store(PublicationState::Published, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus FatalRecord::observe(ControlParty reader, FatalSnapshot& snapshot) const noexcept {
    if (reader != consumer_) {
        return ControlStatus::WrongWriter;
    }
    const PublicationState state = publication_state_.load(std::memory_order_acquire);
    if (state == PublicationState::Idle) {
        snapshot = {};
        snapshot.epoch = initial_epoch_;
        return ControlStatus::Idle;
    }
    if (state == PublicationState::Exhausted) {
        snapshot = {};
        snapshot.epoch = initial_epoch_;
        snapshot.reason = exhausted_attempt_reason_.load(std::memory_order_relaxed);
        snapshot.evidence_valid = false;
        snapshot.exhaustion_latched = true;
        snapshot.exhausted_after_epoch = initial_epoch_;
        snapshot.exhausted_attempt_reason = snapshot.reason;
        snapshot.exhaustion_owner_retained = true;
        return ControlStatus::Exhausted;
    }
    snapshot = {};
    snapshot.epoch = published_epoch_.load(std::memory_order_relaxed);
    snapshot.reason = reason_.load(std::memory_order_relaxed);
    return ControlStatus::Observed;
}

ControlStatus FatalRecord::state() const noexcept {
    FatalSnapshot snapshot{};
    return observe(consumer_, snapshot);
}

StopRecord::StopRecord(ControlParty controller, ControlParty worker,
                       ControlEpoch initial_epoch) noexcept
    : controller_(controller), worker_(worker), requested_(initial_epoch), stopped_(initial_epoch),
      joined_(initial_epoch) {}

ControlStatus StopRecord::request(ControlParty writer, ControlEpoch expected_epoch) noexcept {
    if (writer != controller_) {
        return ControlStatus::WrongWriter;
    }
    const ControlEpoch requested = requested_.load(std::memory_order_relaxed);
    const ControlEpoch stopped = stopped_.load(std::memory_order_acquire);
    const ControlEpoch joined = joined_.load(std::memory_order_acquire);
    if (joined > stopped || stopped > requested) {
        return ControlStatus::Corrupt;
    }
    const ControlStatus comparison = compare_expected(expected_epoch, requested);
    if (comparison != ControlStatus::Applied) {
        return comparison;
    }
    if (requested != joined) {
        return ControlStatus::Repeated;
    }
    if (requested == kMaximumEpoch) {
        exhausted_.store(true, std::memory_order_release);
        return ControlStatus::Exhausted;
    }
    requested_.store(requested + 1, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus StopRecord::publish_stopped(ControlParty writer, ControlEpoch stop_epoch) noexcept {
    if (writer != worker_) {
        return ControlStatus::WrongWriter;
    }
    const ControlEpoch requested = requested_.load(std::memory_order_acquire);
    const ControlEpoch stopped = stopped_.load(std::memory_order_relaxed);
    if (stopped > requested) {
        return ControlStatus::Corrupt;
    }
    if (stop_epoch < stopped) {
        return ControlStatus::Stale;
    }
    if (stop_epoch == stopped) {
        return ControlStatus::Repeated;
    }
    if (stop_epoch != requested) {
        return ControlStatus::Conflict;
    }
    stopped_.store(stop_epoch, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus StopRecord::acknowledge_join(ControlParty writer, ControlEpoch stop_epoch) noexcept {
    if (writer != controller_) {
        return ControlStatus::WrongWriter;
    }
    const ControlEpoch stopped = stopped_.load(std::memory_order_acquire);
    const ControlEpoch joined = joined_.load(std::memory_order_relaxed);
    if (joined > stopped) {
        return ControlStatus::Corrupt;
    }
    if (stop_epoch < joined) {
        return ControlStatus::Stale;
    }
    if (stop_epoch == joined) {
        return ControlStatus::Repeated;
    }
    if (stop_epoch != stopped) {
        return ControlStatus::Conflict;
    }
    joined_.store(stop_epoch, std::memory_order_release);
    return ControlStatus::Applied;
}

ControlStatus StopRecord::state() const noexcept {
    const ControlEpoch requested = requested_.load(std::memory_order_acquire);
    const ControlEpoch stopped = stopped_.load(std::memory_order_acquire);
    const ControlEpoch joined = joined_.load(std::memory_order_acquire);
    if (joined > stopped || stopped > requested) {
        return ControlStatus::Corrupt;
    }
    if (joined == requested) {
        if (exhausted_.load(std::memory_order_acquire)) {
            return ControlStatus::Exhausted;
        }
        return requested == 0 ? ControlStatus::Idle : ControlStatus::Joined;
    }
    if (stopped == requested) {
        return ControlStatus::Stopped;
    }
    return ControlStatus::Pending;
}

ControlEpoch StopRecord::requested_epoch() const noexcept {
    return requested_.load(std::memory_order_acquire);
}

ControlEpoch StopRecord::stopped_epoch() const noexcept {
    return stopped_.load(std::memory_order_acquire);
}

ControlEpoch StopRecord::joined_epoch() const noexcept {
    return joined_.load(std::memory_order_acquire);
}

RuntimeControlRecords::RuntimeControlRecords(ControlEpoch time_epoch) noexcept
    : t_to_p(ControlParty::Transport, ControlParty::Preparation),
      p_to_t(ControlParty::Preparation, ControlParty::Transport),
      t_to_e(ControlParty::Transport, ControlParty::Engine),
      e_to_t(ControlParty::Engine, ControlParty::Transport),
      time(ControlParty::Transport, ControlParty::Engine, time_epoch),
      drain(ControlParty::Transport, ControlParty::Preparation, ControlParty::Engine),
      p_fatal(ControlParty::Preparation, ControlParty::Transport),
      e_fatal(ControlParty::Engine, ControlParty::Transport),
      p_stop(ControlParty::Supervisor, ControlParty::Preparation),
      e_stop(ControlParty::Supervisor, ControlParty::Engine),
      t_stop(ControlParty::Supervisor, ControlParty::Transport) {}

} // namespace tatara::runtime
