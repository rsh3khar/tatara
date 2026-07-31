#include "tatara/service/observability.h"

#include <sstream>

namespace tatara::service {
namespace {

void counter(std::ostringstream& out, const char* name, const char* help, std::uint64_t value) {
    out << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " counter\n"
        << name << ' ' << value << '\n';
}

void gauge(std::ostringstream& out, const char* name, const char* help, double value) {
    out << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " gauge\n"
        << name << ' ' << value << '\n';
}

} // namespace

std::string_view readiness_name(Readiness readiness) {
    switch (readiness) {
    case Readiness::Ready:
        return "ready";
    case Readiness::ModelLoading:
        return "model-loading";
    case Readiness::GatesUnmet:
        return "gates-unmet";
    case Readiness::Draining:
        return "draining";
    case Readiness::Failed:
        return "failed";
    }
    return "unknown";
}

bool is_ready(const ServiceSnapshot& snapshot) {
    return snapshot.readiness == Readiness::Ready;
}

bool is_live(const ServiceSnapshot& snapshot) {
    // Only an unrecoverable failure makes the process worth replacing. Loading
    // and draining are transient states a restart would interrupt, not fix.
    return snapshot.readiness != Readiness::Failed;
}

std::string render_health_json(const ServiceSnapshot& snapshot) {
    std::ostringstream out;
    out << "{\"live\":" << (is_live(snapshot) ? "true" : "false")
        << ",\"ready\":" << (is_ready(snapshot) ? "true" : "false") << ",\"state\":\""
        << readiness_name(snapshot.readiness) << "\",\"running\":" << snapshot.gauges.running
        << ",\"queued\":" << snapshot.gauges.queued
        << ",\"uptime_seconds\":" << snapshot.gauges.uptime_seconds;
    // Identity travels with health so a supervisor can tell which build and
    // which model answered, per VISION §6 on reproducible evidence.
    if (!snapshot.engine_version.empty()) {
        out << ",\"engine\":\"" << snapshot.engine_version << "\"";
    }
    if (!snapshot.model_package_id.empty()) {
        out << ",\"model\":\"" << snapshot.model_package_id << "\"";
    }
    if (!snapshot.record_digest.empty()) {
        out << ",\"record\":\"" << snapshot.record_digest << "\"";
    }
    out << "}";
    return out.str();
}

std::string render_prometheus(const ServiceSnapshot& snapshot) {
    std::ostringstream out;
    const auto& counters = snapshot.counters;
    const auto& gauges = snapshot.gauges;

    counter(out, "tatara_requests_admitted_total", "Requests accepted into the queue.",
            counters.requests_admitted);
    counter(out, "tatara_requests_rejected_context_total",
            "Requests refused because prompt plus output exceeds the context.",
            counters.requests_rejected_context);
    counter(out, "tatara_requests_rejected_queue_full_total",
            "Requests shed because the queue was at its bound.",
            counters.requests_rejected_queue_full);
    counter(out, "tatara_requests_rejected_not_ready_total",
            "Requests refused because the service was not ready.",
            counters.requests_rejected_not_ready);
    counter(out, "tatara_requests_expired_total", "Requests dropped after their deadline elapsed.",
            counters.requests_expired);
    counter(out, "tatara_requests_completed_total", "Requests that produced a full response.",
            counters.requests_completed);
    counter(out, "tatara_requests_cancelled_total", "Requests cancelled by the client.",
            counters.requests_cancelled);
    counter(out, "tatara_prompt_tokens_total", "Prompt tokens ingested.", counters.prompt_tokens);
    counter(out, "tatara_generated_tokens_total", "Tokens generated.", counters.generated_tokens);
    counter(out, "tatara_prefix_cache_hits_total",
            "Exact prefix-cache matches restored into an execution slot.",
            counters.prefix_cache_hits);
    counter(out, "tatara_prefix_cache_misses_total",
            "Prefix-cache lookups with no reusable exact prefix.",
            counters.prefix_cache_misses);
    counter(out, "tatara_prefix_cache_publications_total",
            "Prefix snapshots published after a successful request terminal.",
            counters.prefix_cache_publications);
    counter(out, "tatara_prefix_cache_reservation_skips_total",
            "Snapshots skipped because the configured cache could not admit them.",
            counters.prefix_cache_reservation_skips);
    counter(out, "tatara_prefix_cache_lookup_failures_total",
            "Prefix-cache lookups rejected by an invariant or identity failure.",
            counters.prefix_cache_lookup_failures);
    counter(out, "tatara_prefix_cache_restore_failures_total",
            "Prefix restores that did not complete with observable success.",
            counters.prefix_cache_restore_failures);
    counter(out, "tatara_prefix_cache_snapshot_failures_total",
            "Prefix snapshots that did not complete with observable success.",
            counters.prefix_cache_snapshot_failures);

    gauge(out, "tatara_requests_running", "Requests currently executing.",
          static_cast<double>(gauges.running));
    gauge(out, "tatara_requests_queued", "Requests waiting for a slot.",
          static_cast<double>(gauges.queued));
    gauge(out, "tatara_queue_depth", "Configured queue bound.",
          static_cast<double>(gauges.queue_depth));
    gauge(out, "tatara_queue_saturation", "Queue occupancy in [0, 1].", gauges.saturation);
    gauge(out, "tatara_uptime_seconds", "Seconds since the process became live.",
          static_cast<double>(gauges.uptime_seconds));
    gauge(out, "tatara_ready", "1 when the service is admitting traffic.",
          is_ready(snapshot) ? 1.0 : 0.0);
    return out.str();
}

} // namespace tatara::service
