#include "tatara/service/router.h"

#include <sstream>

namespace tatara::service {
namespace {

std::string_view path_of(std::string_view target) {
    const auto query = target.find('?');
    return query == std::string_view::npos ? target : target.substr(0, query);
}

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char character : value) {
        if (character == '"' || character == '\\') {
            out.push_back('\\');
        }
        out.push_back(character);
    }
    return out;
}

} // namespace

Route resolve_route(std::string_view method, std::string_view target) {
    const std::string_view path = path_of(target);
    const bool is_post = method == "POST";
    const bool is_get = method == "GET";

    if (path == "/v1/chat/completions") {
        return is_post ? Route::ChatCompletions : Route::MethodNotAllowed;
    }
    if (path == "/v1/completions") {
        return is_post ? Route::Completions : Route::MethodNotAllowed;
    }
    if (path == "/v1/models") {
        return is_get ? Route::Models : Route::MethodNotAllowed;
    }
    // Liveness and readiness are separate paths, not one endpoint with a flag,
    // because a supervisor wires them to different actions.
    if (path == "/health/live") {
        return is_get ? Route::HealthLive : Route::MethodNotAllowed;
    }
    if (path == "/health/ready") {
        return is_get ? Route::HealthReady : Route::MethodNotAllowed;
    }
    if (path == "/metrics") {
        return is_get ? Route::Metrics : Route::MethodNotAllowed;
    }
    return Route::NotFound;
}

std::string render_error(std::string_view message, std::string_view type, std::string_view code) {
    std::ostringstream out;
    out << "{\"error\":{\"message\":\"" << json_escape(message) << "\",\"type\":\""
        << json_escape(type) << "\",\"code\":\"" << json_escape(code) << "\"}}";
    return out.str();
}

RoutedResponse route_operational(Route route, const ServiceSnapshot& snapshot) {
    switch (route) {
    case Route::HealthLive:
        // Liveness never consults the model: a supervisor restarting a loading
        // process would loop forever.
        return {is_live(snapshot) ? 200 : 503, "application/json", render_health_json(snapshot),
                false};
    case Route::HealthReady:
        return {is_ready(snapshot) ? 200 : 503, "application/json", render_health_json(snapshot),
                false};
    case Route::Metrics:
        return {200, "text/plain; version=0.0.4", render_prometheus(snapshot), false};
    case Route::Models: {
        std::ostringstream out;
        out << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(snapshot.model_package_id)
            << "\",\"object\":\"model\",\"owned_by\":\"tatara\"";
        if (!snapshot.record_digest.empty()) {
            out << ",\"tatara\":{\"record\":\"" << json_escape(snapshot.record_digest) << "\"}";
        }
        out << "}]}";
        return {200, "application/json", out.str(), false};
    }
    case Route::MethodNotAllowed:
        return {405, "application/json",
                render_error("method not allowed for this path", "invalid_request_error",
                             "tatara.invalid_request"),
                false};
    case Route::NotFound:
        return {404, "application/json",
                render_error("no such endpoint", "invalid_request_error", "tatara.invalid_request"),
                false};
    case Route::ChatCompletions:
    case Route::Completions:
        return {500, "application/json",
                render_error("completion routing is handled by the engine path", "server_error",
                             "tatara.internal"),
                false};
    }
    return {500, "application/json", render_error("unrouted", "server_error", "tatara.internal"),
            false};
}

RoutedResponse reject(Rejection rejection) {
    const int status = rejection_http_status(rejection);
    std::string_view message;
    std::string_view type = "invalid_request_error";
    switch (rejection) {
    case Rejection::InvalidRequest:
        message = "the request identity is invalid";
        break;
    case Rejection::DuplicateRequest:
        message = "the request identity is already active";
        break;
    case Rejection::ContextExceeded:
        message = "prompt plus requested output exceeds the configured context";
        break;
    case Rejection::QueueFull:
        message = "the request queue is at its configured bound";
        type = "server_error";
        break;
    case Rejection::NotReady:
        message = "the model is not resident or required gates are unmet";
        type = "server_error";
        break;
    case Rejection::Draining:
        message = "the service is draining and cannot accept new work";
        type = "server_error";
        break;
    case Rejection::DeadlineExceeded:
        message = "the request deadline elapsed before it could complete";
        type = "server_error";
        break;
    case Rejection::DeadlineOverflow:
        message = "the request deadline cannot be represented";
        break;
    case Rejection::None:
        return {200, "application/json", "{}", false};
    }
    return {status, "application/json", render_error(message, type, rejection_code(rejection)),
            false};
}

} // namespace tatara::service
