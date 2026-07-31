#pragma once

#include <cstdint>
#include <string_view>

namespace tatara {

inline constexpr std::string_view kVersion = "0.1.0";

// Evidence-backed release support claim. Serving admission remains an
// independent model-and-memory computation and can admit a larger context.
inline constexpr std::uint32_t kQualifiedContextTokens = 100'000;

} // namespace tatara
