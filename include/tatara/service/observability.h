#pragma once

#include <cstdint>
#include <string>

namespace tatara::service {

// Liveness and readiness are separate signals on purpose. A supervisor that
// restarts on a failed liveness check while the model is still loading would
// loop forever, so liveness must never depend on the model.
enum class Health : std::uint8_t {
    Live,
    Ready,
    NotReady,
};

// Why the service is not ready. Reported so an operator does not have to guess
// whether to wait or to intervene.
enum class Readiness : std::uint8_t {
    Ready,
    ModelLoading,
    GatesUnmet,
    Draining,
    Failed,
};

std::string_view readiness_name(Readiness readiness);

struct ServiceCounters {
    std::uint64_t requests_admitted{0};
    std::uint64_t requests_rejected_context{0};
    std::uint64_t requests_rejected_queue_full{0};
    std::uint64_t requests_rejected_not_ready{0};
    std::uint64_t requests_expired{0};
    std::uint64_t requests_completed{0};
    std::uint64_t requests_cancelled{0};
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    std::uint64_t prefix_cache_hits{0};
    std::uint64_t prefix_cache_misses{0};
    std::uint64_t prefix_cache_publications{0};
    std::uint64_t prefix_cache_reservation_skips{0};
    std::uint64_t prefix_cache_lookup_failures{0};
    std::uint64_t prefix_cache_restore_failures{0};
    std::uint64_t prefix_cache_snapshot_failures{0};
};

struct ServiceGauges {
    std::uint64_t running{0};
    std::uint64_t queued{0};
    std::uint64_t queue_depth{0};
    double saturation{0.0};
    std::uint64_t uptime_seconds{0};
};

struct ServiceSnapshot {
    Readiness readiness{Readiness::ModelLoading};
    ServiceCounters counters;
    ServiceGauges gauges;
    std::string engine_version;
    std::string model_package_id;
    std::string record_digest;
};

bool is_ready(const ServiceSnapshot& snapshot);

// Liveness answers "is the process healthy enough to keep", not "can it serve".
// A draining or loading service is live.
bool is_live(const ServiceSnapshot& snapshot);

std::string render_health_json(const ServiceSnapshot& snapshot);

// Prometheus text exposition. Counters end in _total and carry HELP and TYPE,
// because a metric an operator has to guess at is not observability.
std::string render_prometheus(const ServiceSnapshot& snapshot);

} // namespace tatara::service
