#pragma once

#include <string_view>
#include <vector>

namespace tatara::tools {

// `tatara benchmark RECORD ARTIFACT_ROOT STATES [--json]`.
// Boots the model and measures single-stream decode. Separated from the CLI
// translation unit so the command surface stays free of runtime headers.
int run_benchmark(std::vector<std::string_view> arguments);

} // namespace tatara::tools
