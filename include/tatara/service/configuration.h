#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::service {

// Loopback is the default bind because an exposed service without an
// authentication boundary is a production defect, not a convenience.
inline constexpr std::string_view kDefaultBind = "127.0.0.1";
inline constexpr std::uint16_t kDefaultPort = 11434;
inline constexpr std::uint32_t kSchemaVersion = 1;

struct ModelConfiguration {
    std::string record;
    std::string artifact_root;
};

struct ServiceConfiguration {
    std::string bind{kDefaultBind};
    std::uint16_t port{kDefaultPort};
    // Zero selects the largest capacity admitted from the live host and the
    // explicit memory profile. A nonzero value requests that exact capacity;
    // it is not an independent product ceiling.
    std::uint32_t max_context_tokens{0};
    std::uint32_t default_max_output_tokens{16};
    std::uint32_t max_concurrent_requests{8};
    std::uint32_t queue_depth{32};
    std::uint32_t request_deadline_milliseconds{120000};
    std::uint32_t drain_timeout_milliseconds{30000};
};

struct CacheConfiguration {
    bool prompt_reuse{true};
    std::uint64_t budget_bytes{0};
    bool budget_bytes_specified{false};
};

struct SpeculativeConfiguration {
    bool enabled{true};
    std::string draft_checkpoint;
};

struct MemoryConfiguration {
    std::uint64_t os_runtime_reserve_bytes{0};
    std::uint64_t unified_external_occupancy_bytes{0};
    std::uint64_t metal_external_occupancy_bytes{0};
    std::uint64_t graph_object_budget_bytes{0};
    std::uint32_t graph_scratch_lanes{3};

    bool os_runtime_reserve_bytes_specified{false};
    bool unified_external_occupancy_bytes_specified{false};
    bool metal_external_occupancy_bytes_specified{false};
    bool graph_object_budget_bytes_specified{false};
};

struct ObservabilityConfiguration {
    std::string log_format{"json"};
    bool metrics{true};
};

struct Configuration {
    std::uint32_t schema_version{kSchemaVersion};
    ModelConfiguration model;
    ServiceConfiguration service;
    CacheConfiguration cache;
    SpeculativeConfiguration speculative;
    MemoryConfiguration memory;
    ObservabilityConfiguration observability;
};

struct ConfigurationDiagnostic {
    std::size_t line{0};
    std::string message;
};

struct ConfigurationResult {
    bool ok{false};
    Configuration configuration;
    std::vector<ConfigurationDiagnostic> diagnostics;
};

// What the composed engine behind one service command actually implements.
// Configuration syntax is model-independent; this second boundary prevents a
// syntactically valid file from claiming more context, concurrency, cache, or
// lifecycle behavior than the selected engine owns.
struct EngineCapabilities {
    std::uint32_t context_capacity{0};
    std::uint32_t concurrent_requests{0};
    bool queued_admission{false};
    bool request_deadlines{false};
    bool bounded_drain{false};
    bool prompt_reuse{false};
};

// Parses the fixed service schema. Not a general TOML reader: the schema is
// closed, so an unknown key is an error rather than something to preserve.
ConfigurationResult parse_configuration(std::string_view text);

// Pure composed-boundary validation. Every returned diagnostic is fatal:
// ignoring one would make the accepted configuration differ from execution.
std::vector<ConfigurationDiagnostic>
validate_configuration_for_engine(const Configuration& configuration,
                                  const EngineCapabilities& capabilities);

// True only for the loopback literal implemented by the schema-v1 IPv4
// listener. Other loopback spellings need their own socket/address gate.
bool is_loopback_address(std::string_view address);

} // namespace tatara::service
