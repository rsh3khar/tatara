#include "tatara/service/router.h"

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
    value.model_package_id = "qwen36-35b-a3b";
    value.record_digest = "09ae40f4";
    return value;
}

void the_openai_surface_resolves() {
    check(resolve_route("POST", "/v1/chat/completions") == Route::ChatCompletions, "chat");
    check(resolve_route("POST", "/v1/completions") == Route::Completions, "completions");
    check(resolve_route("GET", "/v1/models") == Route::Models, "models");
}

void the_operational_surface_resolves() {
    check(resolve_route("GET", "/health/live") == Route::HealthLive, "liveness");
    check(resolve_route("GET", "/health/ready") == Route::HealthReady, "readiness");
    check(resolve_route("GET", "/metrics") == Route::Metrics, "metrics");
}

void a_query_string_does_not_change_the_route() {
    check(resolve_route("GET", "/metrics?format=text") == Route::Metrics, "query ignored");
}

void the_wrong_method_is_405_not_404() {
    check(resolve_route("GET", "/v1/chat/completions") == Route::MethodNotAllowed,
          "GET on a POST endpoint is 405");
    check(resolve_route("POST", "/metrics") == Route::MethodNotAllowed, "POST on metrics is 405");
    check(route_operational(Route::MethodNotAllowed, snapshot(Readiness::Ready)).status == 405,
          "405 status");
}

void an_unknown_path_is_404() {
    check(resolve_route("GET", "/v1/embeddings") == Route::NotFound,
          "an unimplemented surface is not found, not a silent success");
    check(route_operational(Route::NotFound, snapshot(Readiness::Ready)).status == 404, "404");
}

// Liveness must not depend on the model, or a supervisor restarts a loading
// process forever.
void liveness_is_green_while_loading_but_readiness_is_not() {
    const auto loading = snapshot(Readiness::ModelLoading);

    check(route_operational(Route::HealthLive, loading).status == 200, "live while loading");
    check(route_operational(Route::HealthReady, loading).status == 503, "not ready while loading");
}

void a_failed_service_is_not_live() {
    check(route_operational(Route::HealthLive, snapshot(Readiness::Failed)).status == 503,
          "an unrecoverable failure is not live");
}

void models_reports_identity() {
    const auto response = route_operational(Route::Models, snapshot(Readiness::Ready));

    check(response.status == 200, "models is 200");
    check(response.body.find("qwen36-35b-a3b") != std::string::npos, "names the model");
    check(response.body.find("09ae40f4") != std::string::npos,
          "carries the record digest so a client can reproduce the decision");
    check(response.body.find("\"object\":\"list\"") != std::string::npos, "OpenAI list shape");
}

void metrics_is_prometheus_text_not_json() {
    const auto response = route_operational(Route::Metrics, snapshot(Readiness::Ready));
    check(response.content_type.find("text/plain") == 0, "Prometheus exposition is text");
}

// Shed load and not-ready must stay distinguishable all the way to the client.
void rejections_carry_distinct_statuses_and_codes() {
    const auto full = reject(Rejection::QueueFull);
    const auto unready = reject(Rejection::NotReady);
    const auto context = reject(Rejection::ContextExceeded);
    const auto deadline = reject(Rejection::DeadlineExceeded);

    check(full.status == 429 && full.body.find("tatara.queue_full") != std::string::npos,
          "429 with a stable code");
    check(unready.status == 503 && unready.body.find("tatara.not_ready") != std::string::npos,
          "503 with a stable code");
    check(context.status == 400 &&
              context.body.find("tatara.context_exceeded") != std::string::npos,
          "400 with a stable code");
    check(deadline.status == 504 &&
              deadline.body.find("tatara.deadline_exceeded") != std::string::npos,
          "504 with a stable code");
}

void errors_use_the_openai_envelope() {
    const auto body = render_error("something \"quoted\" broke", "server_error", "tatara.internal");

    check(body.find("\"error\":{") != std::string::npos, "OpenAI envelope so clients parse it");
    check(body.find("\"type\":\"server_error\"") != std::string::npos, "type is present");
    check(body.find("\"code\":\"tatara.internal\"") != std::string::npos, "stable Tatara code");
    check(body.find("\\\"quoted\\\"") != std::string::npos, "quotes are escaped");
}

} // namespace

int main() {
    the_openai_surface_resolves();
    the_operational_surface_resolves();
    a_query_string_does_not_change_the_route();
    the_wrong_method_is_405_not_404();
    an_unknown_path_is_404();
    liveness_is_green_while_loading_but_readiness_is_not();
    a_failed_service_is_not_live();
    models_reports_identity();
    metrics_is_prometheus_text_not_json();
    rejections_carry_distinct_statuses_and_codes();
    errors_use_the_openai_envelope();
    if (failures == 0) {
        std::printf("router: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
