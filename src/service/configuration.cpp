#include "tatara/service/configuration.h"

#include <charconv>
#include <limits>

namespace tatara::service {
namespace {

std::string_view trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

std::string_view strip_comment(std::string_view value) {
    bool quoted = false;
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '"') {
            quoted = !quoted;
        } else if (value[index] == '#' && !quoted) {
            return value.substr(0, index);
        }
    }
    return value;
}

bool parse_unsigned(std::string_view value, std::uint64_t limit, std::uint64_t& out) {
    if (value.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed > limit) {
        return false;
    }
    out = parsed;
    return true;
}

bool parse_boolean(std::string_view value, bool& out) {
    if (value == "true") {
        out = true;
        return true;
    }
    if (value == "false") {
        out = false;
        return true;
    }
    return false;
}

bool parse_string(std::string_view value, std::string& out) {
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
        return false;
    }
    out.assign(value.substr(1, value.size() - 2));
    return true;
}

} // namespace

bool is_loopback_address(std::string_view address) {
    return address == "127.0.0.1";
}

ConfigurationResult parse_configuration(std::string_view text) {
    ConfigurationResult result;
    Configuration& configuration = result.configuration;
    auto fail = [&result](std::size_t line, std::string message) {
        result.diagnostics.push_back({line, std::move(message)});
    };

    std::string section;
    bool saw_schema_version = false;
    std::size_t number = 0;
    std::size_t cursor = 0;
    while (cursor <= text.size()) {
        const auto newline = text.find('\n', cursor);
        const auto raw = text.substr(
            cursor, newline == std::string_view::npos ? std::string_view::npos : newline - cursor);
        cursor = newline == std::string_view::npos ? text.size() + 1 : newline + 1;
        ++number;

        const auto line = trim(strip_comment(raw));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[') {
            if (line.back() != ']') {
                fail(number, "unterminated section header");
                continue;
            }
            section.assign(trim(line.substr(1, line.size() - 2)));
            if (section != "model" && section != "service" &&
                section != "cache" && section != "memory" &&
                section != "speculative" &&
                section != "observability") {
                fail(number, "unknown section [" + section + "]");
            }
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string_view::npos) {
            fail(number, "expected key = value");
            continue;
        }
        const auto key = trim(line.substr(0, equals));
        const auto value = trim(line.substr(equals + 1));
        const std::string qualified =
            section.empty() ? std::string(key) : section + "." + std::string(key);

        std::uint64_t number_value = 0;
        auto expect_unsigned = [&](std::uint64_t limit) {
            if (!parse_unsigned(value, limit, number_value)) {
                fail(number, "value for " + qualified + " is not in range");
                return false;
            }
            return true;
        };
        auto expect_string = [&](std::string& out) {
            if (!parse_string(value, out)) {
                fail(number, "value for " + qualified + " must be a quoted string");
                return false;
            }
            return true;
        };
        auto expect_boolean = [&](bool& out) {
            if (!parse_boolean(value, out)) {
                fail(number, "value for " + qualified + " must be true or false");
                return false;
            }
            return true;
        };

        if (qualified == "schema_version") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.schema_version = static_cast<std::uint32_t>(number_value);
                saw_schema_version = true;
            }
        } else if (qualified == "model.record") {
            expect_string(configuration.model.record);
        } else if (qualified == "model.artifact_root") {
            expect_string(configuration.model.artifact_root);
        } else if (qualified == "speculative.enabled") {
            expect_boolean(configuration.speculative.enabled);
        } else if (qualified == "speculative.draft_checkpoint") {
            expect_string(configuration.speculative.draft_checkpoint);
        } else if (qualified == "service.bind") {
            expect_string(configuration.service.bind);
        } else if (qualified == "service.port") {
            if (expect_unsigned(std::numeric_limits<std::uint16_t>::max())) {
                configuration.service.port = static_cast<std::uint16_t>(number_value);
            }
        } else if (qualified == "service.max_context_tokens") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.max_context_tokens = static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "service.default_max_output_tokens") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.default_max_output_tokens =
                    static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "service.max_concurrent_requests") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.max_concurrent_requests =
                    static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "service.queue_depth") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.queue_depth = static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "service.request_deadline_ms") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.request_deadline_milliseconds =
                    static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "service.drain_timeout_ms") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.service.drain_timeout_milliseconds =
                    static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "cache.prompt_reuse") {
            expect_boolean(configuration.cache.prompt_reuse);
        } else if (qualified == "cache.budget_bytes") {
            if (expect_unsigned(std::numeric_limits<std::uint64_t>::max())) {
                configuration.cache.budget_bytes = number_value;
                configuration.cache.budget_bytes_specified = true;
            }
        } else if (qualified == "memory.os_runtime_reserve_bytes") {
            if (expect_unsigned(std::numeric_limits<std::uint64_t>::max())) {
                configuration.memory.os_runtime_reserve_bytes = number_value;
                configuration.memory.os_runtime_reserve_bytes_specified = true;
            }
        } else if (qualified ==
                   "memory.unified_external_occupancy_bytes") {
            if (expect_unsigned(std::numeric_limits<std::uint64_t>::max())) {
                configuration.memory.unified_external_occupancy_bytes =
                    number_value;
                configuration.memory
                    .unified_external_occupancy_bytes_specified = true;
            }
        } else if (qualified == "memory.metal_external_occupancy_bytes") {
            if (expect_unsigned(std::numeric_limits<std::uint64_t>::max())) {
                configuration.memory.metal_external_occupancy_bytes =
                    number_value;
                configuration.memory
                    .metal_external_occupancy_bytes_specified = true;
            }
        } else if (qualified == "memory.graph_object_budget_bytes") {
            if (expect_unsigned(std::numeric_limits<std::uint64_t>::max())) {
                configuration.memory.graph_object_budget_bytes = number_value;
                configuration.memory.graph_object_budget_bytes_specified =
                    true;
            }
        } else if (qualified == "memory.graph_scratch_lanes") {
            if (expect_unsigned(std::numeric_limits<std::uint32_t>::max())) {
                configuration.memory.graph_scratch_lanes =
                    static_cast<std::uint32_t>(number_value);
            }
        } else if (qualified == "observability.log_format") {
            if (expect_string(configuration.observability.log_format) &&
                configuration.observability.log_format != "json" &&
                configuration.observability.log_format != "text") {
                fail(number, "observability.log_format must be \"json\" or \"text\"");
            }
        } else if (qualified == "observability.metrics") {
            expect_boolean(configuration.observability.metrics);
        } else {
            // A silently ignored key is how an operator comes to believe a
            // limit is set when it is not.
            fail(number, "unknown key " + qualified);
        }
    }

    if (!saw_schema_version) {
        fail(0, "schema_version is required");
    } else if (configuration.schema_version != kSchemaVersion) {
        fail(0, "schema_version " + std::to_string(configuration.schema_version) +
                    " is not supported, expected " + std::to_string(kSchemaVersion));
    }
    if (configuration.model.record.empty()) {
        fail(0, "model.record is required");
    }
    if (configuration.model.artifact_root.empty()) {
        fail(0, "model.artifact_root is required");
    }
    if (!is_loopback_address(configuration.service.bind)) {
        fail(0, "service.bind " + configuration.service.bind +
                    " is not loopback; schema version 1 has no authentication boundary");
    }
    if (configuration.service.max_concurrent_requests == 0) {
        fail(0, "service.max_concurrent_requests must be at least 1");
    }
    if (configuration.service.default_max_output_tokens == 0) {
        fail(0, "service.default_max_output_tokens must be at least 1");
    }
    if (!configuration.cache.budget_bytes_specified) {
        fail(0, "cache.budget_bytes is required (zero disables the cache budget)");
    }
    if (!configuration.memory.os_runtime_reserve_bytes_specified ||
        configuration.memory.os_runtime_reserve_bytes == 0) {
        fail(0, "memory.os_runtime_reserve_bytes is required and must be nonzero");
    }
    if (!configuration.memory.unified_external_occupancy_bytes_specified) {
        fail(0, "memory.unified_external_occupancy_bytes is required (zero asserts an exclusive-machine envelope)");
    }
    if (!configuration.memory.metal_external_occupancy_bytes_specified) {
        fail(0, "memory.metal_external_occupancy_bytes is required (zero asserts an exclusive-Metal envelope)");
    }
    if (!configuration.memory.graph_object_budget_bytes_specified) {
        fail(0, "memory.graph_object_budget_bytes is required");
    }
    if (configuration.memory.graph_scratch_lanes == 0) {
        fail(0, "memory.graph_scratch_lanes must be at least 1");
    }

    result.ok = result.diagnostics.empty();
    return result;
}

std::vector<ConfigurationDiagnostic>
validate_configuration_for_engine(const Configuration& configuration,
                                  const EngineCapabilities& capabilities) {
    std::vector<ConfigurationDiagnostic> diagnostics;
    auto fail = [&diagnostics](std::string message) {
        diagnostics.push_back({0, std::move(message)});
    };

    if (capabilities.context_capacity == 0) {
        fail("engine context capacity is unavailable");
    } else if (configuration.service.max_context_tokens != 0 &&
               configuration.service.max_context_tokens >
                   capabilities.context_capacity) {
        fail("service.max_context_tokens " +
             std::to_string(configuration.service.max_context_tokens) +
             " exceeds the engine allocation " + std::to_string(capabilities.context_capacity));
    }
    if (capabilities.concurrent_requests == 0) {
        fail("engine request capacity is unavailable");
    } else if (configuration.service.max_concurrent_requests > capabilities.concurrent_requests) {
        fail("service.max_concurrent_requests " +
             std::to_string(configuration.service.max_concurrent_requests) +
             " exceeds the composed engine capacity " +
             std::to_string(capabilities.concurrent_requests));
    }
    if (configuration.speculative.enabled &&
        configuration.speculative.draft_checkpoint.empty()) {
        fail("speculative.enabled requires speculative.draft_checkpoint");
    }
    if (configuration.service.queue_depth != 0 && !capabilities.queued_admission) {
        fail("service.queue_depth requires queued admission, which this engine has not wired");
    }
    if (configuration.service.request_deadline_milliseconds != 0 &&
        !capabilities.request_deadlines) {
        fail("service.request_deadline_ms requires deadline enforcement, which this engine has not "
             "wired");
    }
    if (configuration.service.drain_timeout_milliseconds != 0 && !capabilities.bounded_drain) {
        fail("service.drain_timeout_ms requires bounded drain, which this engine has not wired");
    }
    if (configuration.cache.prompt_reuse && !capabilities.prompt_reuse) {
        fail("cache.prompt_reuse is enabled, but this engine has no prompt-state cache");
    }
    return diagnostics;
}

} // namespace tatara::service
