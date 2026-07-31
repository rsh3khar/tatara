#include "tatara/service/configuration.h"

#include <cstdio>
#include <string>
#include <string_view>

namespace {

using tatara::service::parse_configuration;
using tatara::service::validate_configuration_for_engine;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool has_diagnostic(const tatara::service::ConfigurationResult& result, std::string_view fragment) {
    for (const auto& diagnostic : result.diagnostics) {
        if (diagnostic.message.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string minimal(std::string_view extra = {}) {
    std::string text = "schema_version = 1\n"
                       "[model]\n"
                       "record = \"/tmp/model.tatara\"\n"
                       "artifact_root = \"/tmp/weights\"\n";
    text.append(extra);
    text.append("[cache]\n"
                "budget_bytes = 0\n"
                "[memory]\n"
                "os_runtime_reserve_bytes = 4294967296\n"
                "unified_external_occupancy_bytes = 0\n"
                "metal_external_occupancy_bytes = 0\n"
                "graph_object_budget_bytes = 0\n"
                "graph_scratch_lanes = 3\n");
    return text;
}

void accepts_a_minimal_configuration() {
    const auto result = parse_configuration(minimal());
    check(result.ok, "minimal configuration is accepted");
    check(result.configuration.service.bind == "127.0.0.1", "bind defaults to loopback");
    check(result.configuration.service.port == 11434, "port default");
    check(result.configuration.service.default_max_output_tokens == 16,
          "default output budget is explicit");
    check(result.configuration.service.max_context_tokens == 0,
          "context capacity defaults to computed admission");
    check(result.configuration.cache.prompt_reuse, "prompt reuse defaults on");
    check(result.configuration.memory.graph_scratch_lanes == 3,
          "bounded graph scratch lanes are an explicit profile value");
}

void requires_schema_version() {
    const auto result = parse_configuration("[model]\nrecord = \"/a\"\nartifact_root = \"/b\"\n");
    check(!result.ok, "missing schema_version is rejected");
    check(has_diagnostic(result, "schema_version is required"), "names the missing key");
}

void rejects_a_future_schema() {
    const auto result = parse_configuration(
        "schema_version = 2\n[model]\nrecord = \"/a\"\nartifact_root = \"/b\"\n");
    check(!result.ok, "unsupported schema is rejected");
    check(has_diagnostic(result, "not supported"), "explains the version");
}

void rejects_an_unknown_key() {
    const auto result = parse_configuration(minimal("[service]\nmax_contex_tokens = 10\n"));
    check(!result.ok, "a typo is an error, not a silent default");
    check(has_diagnostic(result, "unknown key service.max_contex_tokens"), "names the key");
}

void rejects_an_unknown_section() {
    const auto result = parse_configuration(minimal("[serivce]\nport = 1\n"));
    check(!result.ok, "unknown section is rejected");
    check(has_diagnostic(result, "unknown section"), "names the section");
}

// The rule that matters most: an exposed bind without an authentication
// boundary must fail, not warn.
void refuses_an_exposed_bind_without_authentication() {
    const auto result = parse_configuration(minimal("[service]\nbind = \"0.0.0.0\"\n"));
    check(!result.ok, "exposed bind without auth is refused");
    check(has_diagnostic(result, "not loopback"), "explains why");
}

void refuses_an_authentication_placeholder() {
    const auto result =
        parse_configuration(minimal("[service]\nbind = \"0.0.0.0\"\n[auth]\nmode = \"token\"\n"));
    check(!result.ok, "an auth placeholder cannot admit an exposed bind");
    check(has_diagnostic(result, "unknown section [auth]"),
          "schema v1 names the unsupported auth section");
    check(has_diagnostic(result, "not loopback"),
          "the exposed bind remains independently refused");
}

void refuses_unimplemented_loopback_spellings() {
    const auto ipv6 = parse_configuration(minimal("[service]\nbind = \"::1\"\n"));
    check(!ipv6.ok, "IPv6 loopback is refused before the IPv4 listener");
    const auto hostname = parse_configuration(minimal("[service]\nbind = \"localhost\"\n"));
    check(!hostname.ok, "hostname loopback is refused before the numeric IPv4 listener");
}

void requires_the_model() {
    const auto result = parse_configuration("schema_version = 1\n");
    check(!result.ok, "model is required");
    check(has_diagnostic(result, "model.record is required"), "names the record");
    check(has_diagnostic(result, "model.artifact_root is required"), "names the root");
}

void rejects_zero_limits() {
    const auto result = parse_configuration(minimal("[service]\nmax_concurrent_requests = 0\n"));
    check(!result.ok, "zero concurrency is rejected");
}

void has_no_independent_output_ceiling() {
    const auto automatic = parse_configuration(
        minimal("[service]\nmax_context_tokens = 0\n"
                "default_max_output_tokens = 65536\n"));
    check(automatic.ok,
          "automatic context and a large convenience default parse");

    const auto removed = parse_configuration(
        minimal("[service]\nmaximum_output_tokens = 1024\n"));
    check(!removed.ok,
          "the removed engine output ceiling is not silently accepted");
    check(has_diagnostic(removed,
                         "unknown key service.maximum_output_tokens"),
          "the removed ceiling is named as unknown");
}

void rejects_a_port_out_of_range() {
    const auto result = parse_configuration(minimal("[service]\nport = 70000\n"));
    check(!result.ok, "port above 65535 is rejected");
}

void rejects_an_unquoted_string() {
    const auto result = parse_configuration(
        "schema_version = 1\n[model]\nrecord = /tmp/x\nartifact_root = \"/b\"\n");
    check(!result.ok, "unquoted string is rejected");
    check(has_diagnostic(result, "quoted string"), "explains the requirement");
}

void ignores_comments_and_blank_lines() {
    const auto result = parse_configuration(
        minimal("\n# a comment\n[service]\nport = 8080 # trailing comment\n\n"));
    check(result.ok, "comments and blanks are ignored");
    check(result.configuration.service.port == 8080, "trailing comment is stripped");
}

void keeps_a_hash_inside_a_quoted_value() {
    const auto result = parse_configuration(
        minimal("[model]\nrecord = \"/tmp/a#b.tatara\"\n"));
    check(result.ok, "a hash inside quotes is not a comment");
    check(result.configuration.model.record == "/tmp/a#b.tatara", "value survives intact");
}

void reports_the_offending_line() {
    const auto result = parse_configuration(minimal("[service]\nbogus = 1\n"));
    check(!result.ok, "unknown key rejected");
    bool located = false;
    for (const auto& diagnostic : result.diagnostics) {
        if (diagnostic.line == 6) {
            located = true;
        }
    }
    check(located, "diagnostic carries the line number");
}

tatara::service::EngineCapabilities decode_only_capabilities() {
    return {
        .context_capacity = 16384,
        .concurrent_requests = 1,
        .queued_admission = false,
        .request_deadlines = false,
        .bounded_drain = false,
        .prompt_reuse = false,
    };
}

void composed_validation_rejects_unallocated_context() {
    auto parsed = parse_configuration(
        minimal("[service]\nmax_context_tokens = 16385\nmax_concurrent_requests = 1\n"
                "queue_depth = 0\nrequest_deadline_ms = 0\ndrain_timeout_ms = 0\n"
                "[cache]\nprompt_reuse = false\n"));
    check(parsed.ok, "the context-overflow fixture is syntactically valid");
    const auto diagnostics =
        validate_configuration_for_engine(parsed.configuration, decode_only_capabilities());
    check(!diagnostics.empty() && diagnostics.front().message.find(
                                      "exceeds the engine allocation") != std::string::npos,
          "context past the allocated cache is refused before execution");
}

void composed_validation_names_every_unwired_feature() {
    const auto parsed = parse_configuration(minimal());
    check(parsed.ok, "default configuration parses");
    const auto diagnostics =
        validate_configuration_for_engine(parsed.configuration, decode_only_capabilities());
    check(diagnostics.size() == 6,
          "concurrency, queue, deadline, drain, prompt cache and the"
          " promoted speculative default are independently refused");
}

void composed_validation_accepts_only_implemented_behavior() {
    const auto parsed = parse_configuration(
        minimal("[service]\nmax_context_tokens = 16384\nmax_concurrent_requests = 1\n"
                "queue_depth = 0\nrequest_deadline_ms = 0\ndrain_timeout_ms = 0\n"
                "[cache]\nprompt_reuse = false\nbudget_bytes = 0\n"
                "[speculative]\nenabled = false\n"));
    check(parsed.ok, "decode-only configuration parses");
    check(
        validate_configuration_for_engine(parsed.configuration, decode_only_capabilities()).empty(),
        "the exact composed capability set is accepted");
}

} // namespace


void speculative_defaults_on() {
    const auto result = parse_configuration(minimal(""));
    check(result.ok, "minimal configuration parses");
    check(result.configuration.speculative.enabled,
          "speculation defaults on");
    check(result.configuration.speculative.draft_checkpoint.empty(),
          "no default draft checkpoint");
}

void speculative_path_keeps_the_default_enabled() {
    const auto result = parse_configuration(minimal(
        "[speculative]\ndraft_checkpoint = \"/tmp/draft\"\n"));
    check(result.ok, "path-only speculative section parses");
    check(result.configuration.speculative.enabled,
          "a path-only section stays on the promoted default");
    check(result.configuration.speculative.draft_checkpoint == "/tmp/draft",
          "draft path is recorded");
}

void speculative_default_requires_a_path_or_explicit_opt_out() {
    const auto parsed = parse_configuration(minimal(""));
    check(parsed.ok, "minimal configuration parses");
    const auto validated = validate_configuration_for_engine(
        parsed.configuration, decode_only_capabilities());
    bool missing_path = false;
    for (const auto& diagnostic : validated) {
        missing_path = missing_path ||
                       diagnostic.message.find("requires "
                                               "speculative.draft_"
                                               "checkpoint") !=
                           std::string::npos;
    }
    check(missing_path,
          "the promoted default without a path is refused by name");

    const auto opted_out = parse_configuration(minimal(
        "[speculative]\nenabled = false\n"));
    check(opted_out.ok, "explicit opt-out parses");
    for (const auto& diagnostic : validate_configuration_for_engine(
             opted_out.configuration, decode_only_capabilities())) {
        check(diagnostic.message.find("speculative") == std::string::npos,
              "explicit opt-out raises no speculative diagnostic");
    }
}

void speculative_enabled_with_a_path_validates() {
    const auto parsed = parse_configuration(minimal(
        "[speculative]\nenabled = true\n"
        "draft_checkpoint = \"/tmp/draft\"\n"));
    check(parsed.ok, "enabled with path parses");
    const auto validated = validate_configuration_for_engine(
        parsed.configuration, decode_only_capabilities());
    for (const auto& diagnostic : validated) {
        check(diagnostic.message.find("speculative") == std::string::npos,
              "enabled with a path raises no speculative diagnostic");
    }
}

int main() {
    accepts_a_minimal_configuration();
    requires_schema_version();
    rejects_a_future_schema();
    rejects_an_unknown_key();
    rejects_an_unknown_section();
    refuses_an_exposed_bind_without_authentication();
    refuses_an_authentication_placeholder();
    refuses_unimplemented_loopback_spellings();
    requires_the_model();
    rejects_zero_limits();
    has_no_independent_output_ceiling();
    rejects_a_port_out_of_range();
    rejects_an_unquoted_string();
    ignores_comments_and_blank_lines();
    keeps_a_hash_inside_a_quoted_value();
    reports_the_offending_line();
    composed_validation_rejects_unallocated_context();
    composed_validation_names_every_unwired_feature();
    composed_validation_accepts_only_implemented_behavior();
    speculative_defaults_on();
    speculative_path_keeps_the_default_enabled();
    speculative_default_requires_a_path_or_explicit_opt_out();
    speculative_enabled_with_a_path_validates();
    if (failures == 0) {
        std::printf("service configuration: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
