// The native `tatara` command: the shipped distribution carries its own CLI
// with no Python runtime, and commands that do not touch the GPU must not
// load Metal or open the model, which is what keeps `doctor` and `--help`
// instant.

#include "tatara_benchmark.h"
#include "tatara_serve.h"

#include "tatara/generated/model_plan.h"
#include "tatara/host/capability.h"
#include "tatara/service/configuration.h"
#include "tatara/service/record_validation.h"
#include "tatara/version.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Exit contract: only 0 and 1 carry a verdict; every other code means the
// verdict is unknown.
constexpr int kExitVerdictPositive = 0;
constexpr int kExitUsage = 2;
constexpr int kExitConfigurationInvalid = 3;
constexpr int kExitRecordInvalid = 4;
constexpr int kExitHostUnsupported = 5;

struct Command {
    std::string_view name;
    std::string_view summary;
    int (*run)(std::vector<std::string_view>);
};

int usage(std::FILE* stream);

std::string read_file(std::string_view path, bool& ok) {
    std::ifstream file{std::string(path), std::ios::binary};
    if (!file) {
        ok = false;
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    ok = true;
    return buffer.str();
}

int run_config_check(std::vector<std::string_view> arguments) {
    if (arguments.size() != 1) {
        std::fprintf(stderr, "usage: tatara config check <path>\n");
        return kExitUsage;
    }
    bool ok = false;
    const std::string text = read_file(arguments[0], ok);
    if (!ok) {
        std::fprintf(stderr, "config: cannot read %.*s\n", static_cast<int>(arguments[0].size()),
                     arguments[0].data());
        return kExitConfigurationInvalid;
    }
    const auto result = tatara::service::parse_configuration(text);
    if (!result.ok) {
        std::printf("verdict: failed\n");
        for (const auto& diagnostic : result.diagnostics) {
            if (diagnostic.line == 0) {
                std::printf("  %s\n", diagnostic.message.c_str());
            } else {
                std::printf("  line %zu: %s\n", diagnostic.line, diagnostic.message.c_str());
            }
        }
        return kExitConfigurationInvalid;
    }
    const auto& configuration = result.configuration;
    const auto execution_diagnostics = tatara::service::validate_configuration_for_engine(
        configuration, tatara::tools::serve_capabilities(configuration.service.max_concurrent_requests,
                                     configuration.service.queue_depth));
    if (!execution_diagnostics.empty()) {
        std::printf("verdict: failed\n");
        for (const auto& diagnostic : execution_diagnostics) {
            std::printf("  %s\n", diagnostic.message.c_str());
        }
        return kExitConfigurationInvalid;
    }
    std::printf("schema: %u\n", configuration.schema_version);
    std::printf("model: %s\n", configuration.model.record.c_str());
    std::printf("bind: %s:%u%s\n", configuration.service.bind.c_str(), configuration.service.port,
                tatara::service::is_loopback_address(configuration.service.bind)
                    ? " (loopback)"
                    : " (exposed, authenticated)");
    if (configuration.service.max_context_tokens == 0) {
        std::printf(
            "limits: context automatic (model maximum %u),"
            " %u concurrent, queue %u\n",
            tatara::tools::serve_capabilities(configuration.service.max_concurrent_requests,
                                     configuration.service.queue_depth).context_capacity,
            configuration.service.max_concurrent_requests,
            configuration.service.queue_depth);
    } else {
        std::printf(
            "limits: requested context %u (model maximum %u),"
            " %u concurrent, queue %u\n",
            configuration.service.max_context_tokens,
            tatara::tools::serve_capabilities(configuration.service.max_concurrent_requests,
                                     configuration.service.queue_depth).context_capacity,
            configuration.service.max_concurrent_requests,
            configuration.service.queue_depth);
    }
    std::printf(
        "memory: cache %llu, OS/runtime reserve %llu,"
        " unified external %llu, Metal external %llu,"
        " graph objects %llu, graph lanes %u\n",
        static_cast<unsigned long long>(
            configuration.cache.budget_bytes),
        static_cast<unsigned long long>(
            configuration.memory.os_runtime_reserve_bytes),
        static_cast<unsigned long long>(
            configuration.memory.unified_external_occupancy_bytes),
        static_cast<unsigned long long>(
            configuration.memory.metal_external_occupancy_bytes),
        static_cast<unsigned long long>(
            configuration.memory.graph_object_budget_bytes),
        configuration.memory.graph_scratch_lanes);
    std::printf("verdict: valid\n");
    return kExitVerdictPositive;
}

int run_config(std::vector<std::string_view> arguments) {
    if (arguments.empty()) {
        std::fprintf(stderr, "usage: tatara config check <path>\n");
        return kExitUsage;
    }
    const auto subcommand = arguments.front();
    arguments.erase(arguments.begin());
    if (subcommand == "check") {
        return run_config_check(std::move(arguments));
    }
    std::fprintf(stderr, "config: unknown subcommand %.*s\n", static_cast<int>(subcommand.size()),
                 subcommand.data());
    return kExitUsage;
}

int run_doctor(std::vector<std::string_view> arguments) {
    bool as_json = false;
    for (const auto& argument : arguments) {
        if (argument == "--json") {
            as_json = true;
        } else {
            std::fprintf(stderr, "doctor: unknown argument %.*s\n",
                         static_cast<int>(argument.size()), argument.data());
            return kExitUsage;
        }
    }
    const auto report = tatara::host::assess(tatara::host::read_host_facts());
    const std::string text =
        as_json ? tatara::host::render_json(report) : tatara::host::render_human(report);
    std::fputs(text.c_str(), stdout);
    return report.supported() ? kExitVerdictPositive : kExitHostUnsupported;
}

int run_validate(std::vector<std::string_view> arguments) {
    if (arguments.size() != 1) {
        std::fprintf(stderr, "usage: tatara validate <record.tatara>\n");
        return kExitUsage;
    }
    bool ok = false;
    const std::string bytes = read_file(arguments[0], ok);
    if (!ok) {
        std::fprintf(stderr, "validate: cannot read %.*s\n", static_cast<int>(arguments[0].size()),
                     arguments[0].data());
        return kExitRecordInvalid;
    }
    const auto parsed = tatara::model::parse_prepared_checkpoint(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()));
    if (!parsed) {
        std::printf("verdict: record invalid\n");
        return kExitRecordInvalid;
    }
    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    const tatara::model::PreparedCheckpointExpectation expectation{plan.id, plan.package_sha256,
                                                                   plan.artifact};
    const auto facts = tatara::host::read_host_facts();
    const auto report =
        tatara::service::assess_record(*parsed.checkpoint, expectation, facts.memory_bytes);
    std::fputs(tatara::service::render_human(report).c_str(), stdout);
    return tatara::service::verdict_exit_code(report.verdict);
}

int run_version(std::vector<std::string_view>) {
    std::printf("%.*s\n", static_cast<int>(tatara::kVersion.size()),
                tatara::kVersion.data());
    return kExitVerdictPositive;
}

int run_help(std::vector<std::string_view>) {
    return usage(stdout);
}

constexpr Command kCommands[] = {
    {"doctor", "report host capability and capacity", run_doctor},
    {"validate", "check a prepared record against this build and machine", run_validate},
    {"benchmark", "measure single-stream decode with tails", tatara::tools::run_benchmark},
    {"serve", "serve the OpenAI-compatible surface", tatara::tools::run_serve},
    {"config", "validate a service configuration file", run_config},
    {"version", "print the engine version", run_version},
    {"help", "print this message", run_help},
};

int usage(std::FILE* stream) {
    std::fprintf(stream, "tatara %.*s, Apple-silicon inference engine\n\n",
                 static_cast<int>(tatara::kVersion.size()),
                 tatara::kVersion.data());
    std::fprintf(stream, "usage: tatara <command> [arguments]\n\ncommands:\n");
    for (const auto& command : kCommands) {
        std::fprintf(stream, "  %-10.*s %.*s\n", static_cast<int>(command.name.size()),
                     command.name.data(), static_cast<int>(command.summary.size()),
                     command.summary.data());
    }
    std::fprintf(stream, "\nA command appears here only when it works.\n");
    return stream == stdout ? kExitVerdictPositive : kExitUsage;
}

} // namespace

int main(int argument_count, char** arguments) {
    std::vector<std::string_view> tokens;
    tokens.reserve(static_cast<std::size_t>(argument_count));
    for (int index = 1; index < argument_count; ++index) {
        tokens.emplace_back(arguments[index]);
    }
    if (tokens.empty()) {
        return usage(stderr);
    }
    const auto name = tokens.front();
    tokens.erase(tokens.begin());
    if (name == "--version") {
        return run_version(std::move(tokens));
    }
    if (name == "--help" || name == "-h") {
        return usage(stdout);
    }
    for (const auto& command : kCommands) {
        if (command.name == name) {
            return command.run(std::move(tokens));
        }
    }
    std::fprintf(stderr, "tatara: unknown command %.*s\n", static_cast<int>(name.size()),
                 name.data());
    return kExitUsage;
}
