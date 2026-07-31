#pragma once

#include "tatara/service/configuration.h"

#include <string_view>
#include <vector>

namespace tatara::tools {

// The capabilities of the engine currently composed behind `tatara serve`.
// Both `config check` and `serve` use this exact value, so a file cannot pass
// the dry check and then acquire a different execution meaning.
service::EngineCapabilities serve_capabilities();

// `tatara serve <config.toml>`. Boots the model, serves the OpenAI-compatible
// surface, and drains on SIGINT or SIGTERM.
int run_serve(std::vector<std::string_view> arguments);

} // namespace tatara::tools
