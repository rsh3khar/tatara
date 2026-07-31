#pragma once

#include "tatara/service/admission.h"
#include "tatara/service/http_message.h"
#include "tatara/service/observability.h"

#include <string>

namespace tatara::service {

// Maps an HTTP request onto the service surface. Pure: it decides what the
// response should be, and never touches a socket or the engine. Routing and
// admission are decided here so both are testable without either.

enum class Route : std::uint8_t {
    ChatCompletions,
    Completions,
    Models,
    HealthLive,
    HealthReady,
    Metrics,
    NotFound,
    MethodNotAllowed,
};

Route resolve_route(std::string_view method, std::string_view target);

struct RoutedResponse {
    int status{200};
    std::string content_type{"application/json"};
    std::string body;
    bool streaming{false};
};

// The OpenAI-shaped error envelope, with a stable Tatara code inside so clients
// parse the shape they expect while operators get a specific cause.
std::string render_error(std::string_view message, std::string_view type, std::string_view code);

RoutedResponse route_operational(Route route, const ServiceSnapshot& snapshot);

RoutedResponse reject(Rejection rejection);

} // namespace tatara::service
