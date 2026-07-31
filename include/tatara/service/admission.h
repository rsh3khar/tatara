#pragma once

#include "tatara/service/configuration.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace tatara::service {

// Pure admission, queueing and deadline logic. No sockets, no threads, no
// clock: time is passed in. This is what makes overload behaviour testable
// without a running service, and overload behaviour is the part that is
// hardest to observe once a service is live.

using Milliseconds = std::uint64_t;
using RequestId = std::uint64_t;

enum class Rejection : std::uint8_t {
    None,
    InvalidRequest,
    DuplicateRequest,
    ContextExceeded,
    QueueFull,
    NotReady,
    Draining,
    DeadlineExceeded,
    DeadlineOverflow,
};

std::string_view rejection_code(Rejection rejection);
int rejection_http_status(Rejection rejection);

struct Request {
    RequestId id{0};
    std::uint32_t prompt_tokens{0};
    std::uint32_t max_output_tokens{0};
    Milliseconds deadline_ms{0};
};

struct Admitted {
    RequestId id{0};
    Milliseconds queued_at_ms{0};
    Milliseconds expires_at_ms{std::numeric_limits<Milliseconds>::max()};
};

inline constexpr std::uint32_t kNoSlot = std::numeric_limits<std::uint32_t>::max();

struct AdmissionOutcome {
    bool accepted{false};
    bool started{false};
    std::uint32_t slot{kNoSlot};
    Rejection rejection{Rejection::None};
};

enum class StopReason : std::uint8_t {
    None,
    Cancelled,
    Deadline,
};

enum class SchedulerEventKind : std::uint8_t {
    Started,
    QueuedCancelled,
    QueuedDeadlineExceeded,
    RunningCancellationRequested,
    RunningDeadlineRequested,
};

struct SchedulerEvent {
    RequestId id{0};
    SchedulerEventKind kind{SchedulerEventKind::Started};
    std::uint32_t slot{kNoSlot};
};

// Pure bounded request scheduler. All storage is allocated by the constructor;
// every later operation is allocation-free and returns a view into owned event
// storage. A running cancellation or deadline never releases a physical state
// slot: only retire(), called after submitted GPU work is safe, can do that.
class Scheduler {
  public:
    Scheduler(const ServiceConfiguration& configuration, bool ready);

    void set_ready(bool ready);
    bool ready() const;
    void begin_drain();
    bool draining() const;

    AdmissionOutcome admit(const Request& request, Milliseconds now_ms);

    // Views remain valid until the next non-const scheduler operation.
    std::span<const SchedulerEvent> start_ready(Milliseconds now_ms);
    std::span<const SchedulerEvent> expire(Milliseconds now_ms);
    std::span<const SchedulerEvent> cancel(RequestId id);

    // Releases a running slot only after its execution owner has established
    // that every submitted command which can address it is complete.
    bool retire(RequestId id);

    std::size_t running() const;
    std::size_t queued() const;
    std::size_t slots() const;
    StopReason stop_reason(RequestId id) const;

    // Backpressure signal for a load balancer: queue occupancy in [0, 1].
    double saturation() const;

  private:
    struct Running {
        bool occupied{false};
        Admitted request;
        StopReason stop_reason{StopReason::None};
    };

    void reset_events();
    void add_event(RequestId id, SchedulerEventKind kind, std::uint32_t slot);
    std::span<const SchedulerEvent> events() const;
    bool contains(RequestId id) const;
    bool expired(const Admitted& request, Milliseconds now_ms) const;
    std::uint32_t free_slot() const;
    void start_in_slot(const Admitted& request, std::uint32_t slot);
    void push_queue(const Admitted& request);
    Admitted pop_queue();
    const Admitted& queue_at(std::size_t offset) const;
    void remove_queue_at(std::size_t offset);

    ServiceConfiguration configuration_;
    bool ready_{false};
    bool draining_{false};
    std::vector<Admitted> queue_;
    std::size_t queue_head_{0};
    std::size_t queue_size_{0};
    std::vector<Running> running_;
    std::size_t running_count_{0};
    std::vector<SchedulerEvent> events_;
    std::size_t event_count_{0};
};

} // namespace tatara::service
