#pragma once

#include <atomic>
#include <cstdint>

namespace tatara::runtime {

using ControlEpoch = std::uint64_t;

enum class ControlParty : std::uint8_t {
    Supervisor,
    Transport,
    Preparation,
    Engine,
};

enum class ControlStatus : std::uint8_t {
    Applied,
    Observed,
    Idle,
    Pending,
    Acknowledged,
    Stopped,
    Joined,
    Repeated,
    Stale,
    ForeignEpoch,
    Conflict,
    Exhausted,
    FailedRetained,
    Corrupt,
    WrongWriter,
    InvalidEpoch,
    InvalidValue,
    Busy,
};

struct WakeEvidenceSnapshot {
    ControlEpoch published_epoch{0};
    ControlEpoch acknowledged_epoch{0};
    std::uint64_t published_total{0};
    std::uint64_t pending_count{0};
    std::uint64_t acknowledged_total{0};
    std::uint64_t failed_retained_now{0};
    std::uint64_t failed_retained_total{0};
    std::uint8_t wake_idle{0};
    std::uint8_t wake_pending{0};
    std::uint8_t wake_acknowledged{0};
    std::uint8_t wake_failed_retained{0};
    std::uint8_t wake_exhausted{0};
    bool evidence_valid{false};
    bool exhaustion_latched{false};
    bool exhaustion_without_owner{false};
    bool exhaustion_owner_retained{false};
    bool conservation_holds{false};
};

class WakeRecord {
  public:
    WakeRecord(ControlParty publisher, ControlParty consumer,
               ControlEpoch initial_epoch = 0) noexcept;

    WakeRecord(const WakeRecord&) = delete;
    WakeRecord& operator=(const WakeRecord&) = delete;

    ControlStatus publish(ControlParty writer, ControlEpoch expected_epoch) noexcept;
    ControlStatus observe(ControlParty reader, ControlEpoch last_seen,
                          ControlEpoch& observed) const noexcept;
    ControlStatus acknowledge(ControlParty writer, ControlEpoch observed_epoch) noexcept;
    ControlStatus retain_failed(ControlParty writer, ControlEpoch observed_epoch) noexcept;
    ControlStatus state() const noexcept;
    WakeEvidenceSnapshot evidence() const noexcept;

    ControlEpoch published_epoch() const noexcept;
    ControlEpoch acknowledged_epoch() const noexcept;

  private:
    ControlParty publisher_;
    ControlParty consumer_;
    ControlEpoch initial_epoch_;
    std::atomic<ControlEpoch> published_;
    std::atomic<ControlEpoch> acknowledged_;
    std::atomic<bool> failed_retained_{false};
    std::atomic<ControlEpoch> failed_retained_epoch_{0};
    std::atomic<bool> exhausted_{false};
};

struct TimeSnapshot {
    ControlEpoch epoch_id{0};
    std::uint64_t monotonic_units{0};
    std::uint64_t publication_sequence{0};
};

class TimeRecord {
  public:
    TimeRecord(ControlParty publisher, ControlParty consumer, ControlEpoch epoch_id,
               std::uint64_t initial_sequence = 0) noexcept;

    TimeRecord(const TimeRecord&) = delete;
    TimeRecord& operator=(const TimeRecord&) = delete;

    ControlStatus publish(ControlParty writer, std::uint64_t monotonic_units) noexcept;
    ControlStatus read(ControlParty reader, ControlEpoch expected_epoch,
                       std::uint64_t observer_units, std::uint64_t maximum_age,
                       TimeSnapshot& snapshot) const noexcept;
    ControlStatus stop(ControlParty writer) noexcept;
    ControlStatus state() const noexcept;

  private:
    enum class PublisherState : std::uint8_t {
        Running,
        Stopped,
        Exhausted,
        Corrupt,
    };

    ControlParty publisher_;
    ControlParty consumer_;
    ControlEpoch epoch_id_;
    std::atomic<std::uint64_t> sequence_;
    std::atomic<std::uint64_t> monotonic_units_{0};
    std::atomic<bool> published_{false};
    std::atomic<PublisherState> publisher_state_{PublisherState::Running};
};

class DrainRecord {
  public:
    DrainRecord(ControlParty publisher, ControlParty first_consumer, ControlParty second_consumer,
                ControlEpoch initial_epoch = 0) noexcept;

    DrainRecord(const DrainRecord&) = delete;
    DrainRecord& operator=(const DrainRecord&) = delete;

    ControlStatus request(ControlParty writer, ControlEpoch expected_epoch) noexcept;
    ControlStatus observe(ControlParty reader, ControlEpoch last_seen,
                          ControlEpoch& observed) const noexcept;
    ControlStatus acknowledge(ControlParty writer, ControlEpoch drain_epoch) noexcept;
    ControlStatus state() const noexcept;
    ControlEpoch epoch() const noexcept;

  private:
    ControlParty publisher_;
    ControlParty first_consumer_;
    ControlParty second_consumer_;
    std::atomic<ControlEpoch> epoch_;
    std::atomic<ControlEpoch> first_ack_;
    std::atomic<ControlEpoch> second_ack_;
    std::atomic<bool> exhausted_{false};
};

enum class FatalReason : std::uint8_t {
    None,
    WorkerFailure,
    OwnershipInvariant,
    TimeInvalid,
    CounterExhausted,
};

struct FatalSnapshot {
    ControlEpoch epoch{0};
    FatalReason reason{FatalReason::None};
    bool evidence_valid{true};
    bool exhaustion_latched{false};
    ControlEpoch exhausted_after_epoch{0};
    FatalReason exhausted_attempt_reason{FatalReason::None};
    bool exhaustion_without_owner{false};
    bool exhaustion_owner_retained{false};
};

class FatalRecord {
  public:
    FatalRecord(ControlParty publisher, ControlParty consumer,
                ControlEpoch initial_epoch = 0) noexcept;

    FatalRecord(const FatalRecord&) = delete;
    FatalRecord& operator=(const FatalRecord&) = delete;

    ControlStatus publish(ControlParty writer, ControlEpoch expected_epoch,
                          FatalReason reason) noexcept;
    ControlStatus observe(ControlParty reader, FatalSnapshot& snapshot) const noexcept;
    ControlStatus state() const noexcept;

  private:
    enum class PublicationState : std::uint8_t {
        Idle,
        Published,
        Exhausted,
    };

    ControlParty publisher_;
    ControlParty consumer_;
    ControlEpoch initial_epoch_;
    std::atomic<ControlEpoch> published_epoch_{0};
    std::atomic<FatalReason> reason_{FatalReason::None};
    std::atomic<FatalReason> exhausted_attempt_reason_{FatalReason::None};
    std::atomic<PublicationState> publication_state_{PublicationState::Idle};
};

class StopRecord {
  public:
    StopRecord(ControlParty controller, ControlParty worker,
               ControlEpoch initial_epoch = 0) noexcept;

    StopRecord(const StopRecord&) = delete;
    StopRecord& operator=(const StopRecord&) = delete;

    ControlStatus request(ControlParty writer, ControlEpoch expected_epoch) noexcept;
    ControlStatus publish_stopped(ControlParty writer, ControlEpoch stop_epoch) noexcept;
    ControlStatus acknowledge_join(ControlParty writer, ControlEpoch stop_epoch) noexcept;
    ControlStatus state() const noexcept;

    ControlEpoch requested_epoch() const noexcept;
    ControlEpoch stopped_epoch() const noexcept;
    ControlEpoch joined_epoch() const noexcept;

  private:
    ControlParty controller_;
    ControlParty worker_;
    std::atomic<ControlEpoch> requested_;
    std::atomic<ControlEpoch> stopped_;
    std::atomic<ControlEpoch> joined_;
    std::atomic<bool> exhausted_{false};
};

class RuntimeControlRecords {
  public:
    explicit RuntimeControlRecords(ControlEpoch time_epoch) noexcept;

    RuntimeControlRecords(const RuntimeControlRecords&) = delete;
    RuntimeControlRecords& operator=(const RuntimeControlRecords&) = delete;

    WakeRecord t_to_p;
    WakeRecord p_to_t;
    WakeRecord t_to_e;
    WakeRecord e_to_t;
    TimeRecord time;
    DrainRecord drain;
    FatalRecord p_fatal;
    FatalRecord e_fatal;
    StopRecord p_stop;
    StopRecord e_stop;
    StopRecord t_stop;
};

} // namespace tatara::runtime
