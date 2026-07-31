#include "tatara/service/observability.h"

#include <cstdio>
#include <string>

namespace {

using namespace tatara::service;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

ServiceSnapshot snapshot(Readiness readiness) {
    ServiceSnapshot value;
    value.readiness = readiness;
    value.engine_version = "0.0.1";
    value.model_package_id = "qwen36-35b-a3b";
    value.record_digest = "09ae40f4";
    value.gauges.queue_depth = 32;
    return value;
}

// The distinction the whole split exists for: a supervisor restarting on a
// failed liveness check while the model loads would loop forever.
void loading_is_live_but_not_ready() {
    const auto value = snapshot(Readiness::ModelLoading);

    check(is_live(value), "a loading service is live");
    check(!is_ready(value), "a loading service is not ready");
}

void draining_is_live_but_not_ready() {
    const auto value = snapshot(Readiness::Draining);

    check(is_live(value), "a draining service is live; restarting would interrupt the drain");
    check(!is_ready(value), "a draining service stops admitting");
}

void only_failure_stops_liveness() {
    check(!is_live(snapshot(Readiness::Failed)), "an unrecoverable failure is not live");
    check(is_live(snapshot(Readiness::GatesUnmet)), "unmet gates are still live");
}

void ready_is_ready() {
    const auto value = snapshot(Readiness::Ready);
    check(is_live(value) && is_ready(value), "ready is both");
}

void health_carries_identity() {
    const auto text = render_health_json(snapshot(Readiness::Ready));

    check(text.find("\"ready\":true") != std::string::npos, "readiness is reported");
    check(text.find("\"state\":\"ready\"") != std::string::npos, "the state is named");
    // A supervisor must be able to say which build and model answered.
    check(text.find("qwen36-35b-a3b") != std::string::npos, "model identity travels with health");
    check(text.find("09ae40f4") != std::string::npos, "record digest travels with health");
}

void readiness_states_are_named_not_boolean() {
    check(readiness_name(Readiness::ModelLoading) == "model-loading", "loading is named");
    check(readiness_name(Readiness::GatesUnmet) == "gates-unmet", "gates-unmet is named");
    check(readiness_name(Readiness::Draining) == "draining", "draining is named");
}

void prometheus_counters_are_typed_and_documented() {
    auto value = snapshot(Readiness::Ready);
    value.counters.requests_admitted = 7;
    value.counters.requests_rejected_queue_full = 3;
    value.counters.prefix_cache_hits = 5;
    value.counters.prefix_cache_misses = 2;
    value.counters.prefix_cache_publications = 4;
    value.counters.prefix_cache_reservation_skips = 1;
    value.counters.prefix_cache_lookup_failures = 6;
    value.counters.prefix_cache_restore_failures = 8;
    value.counters.prefix_cache_snapshot_failures = 9;
    value.gauges.queued = 2;
    value.gauges.saturation = 0.0625;

    const auto text = render_prometheus(value);

    check(text.find("# TYPE tatara_requests_admitted_total counter") != std::string::npos,
          "counters are typed");
    check(text.find("# HELP tatara_requests_admitted_total") != std::string::npos,
          "counters carry help; a metric an operator must guess at is not observability");
    check(text.find("tatara_requests_admitted_total 7") != std::string::npos, "value is exported");
    check(text.find("tatara_requests_rejected_queue_full_total 3") != std::string::npos,
          "shed load is separately countable from other refusals");
    check(text.find("tatara_prefix_cache_hits_total 5") != std::string::npos,
          "prefix-cache hits are exported");
    check(text.find("tatara_prefix_cache_misses_total 2") != std::string::npos,
          "prefix-cache misses are exported");
    check(text.find("tatara_prefix_cache_publications_total 4") != std::string::npos,
          "successful prefix publications are exported");
    check(text.find("tatara_prefix_cache_reservation_skips_total 1") != std::string::npos,
          "nonfatal cache-capacity skips are exported");
    check(text.find("tatara_prefix_cache_lookup_failures_total 6") != std::string::npos,
          "lookup invariant failures are exported");
    check(text.find("tatara_prefix_cache_restore_failures_total 8") != std::string::npos,
          "restore failures are exported");
    check(text.find("tatara_prefix_cache_snapshot_failures_total 9") != std::string::npos,
          "snapshot failures are exported");
    check(text.find("# TYPE tatara_queue_saturation gauge") != std::string::npos,
          "gauges are typed");
    check(text.find("tatara_ready 1") != std::string::npos, "readiness is scrapeable");
}

void not_ready_is_visible_in_metrics() {
    const auto text = render_prometheus(snapshot(Readiness::ModelLoading));
    check(text.find("tatara_ready 0") != std::string::npos, "an unready service reports zero");
}

} // namespace

int main() {
    loading_is_live_but_not_ready();
    draining_is_live_but_not_ready();
    only_failure_stops_liveness();
    ready_is_ready();
    health_carries_identity();
    readiness_states_are_named_not_boolean();
    prometheus_counters_are_typed_and_documented();
    not_ready_is_visible_in_metrics();
    if (failures == 0) {
        std::printf("observability: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
