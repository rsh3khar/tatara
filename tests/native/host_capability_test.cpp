#include "tatara/host/capability.h"

#include <cstdio>
#include <string>

namespace {

using namespace tatara::host;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

HostFacts supported_host() {
    HostFacts facts;
    facts.system = "Darwin";
    facts.os_version = "26.0";
    facts.architecture = "arm64";
    facts.chip = "Apple M4 Pro";
    facts.memory_bytes = 48ull * kGibiByte;
    facts.performance_cores = 10;
    facts.efficiency_cores = 4;
    facts.metal_device_present = true;
    facts.metal_device_name = "Apple M4 Pro";
    return facts;
}

bool blocked_on(const CapabilityReport& report, const std::string& name) {
    for (const auto& check : report.checks) {
        if (check.name == name) {
            return check.required && !check.passed;
        }
    }
    return false;
}

void a_supported_host_passes() {
    const auto report = assess(supported_host());
    check(report.supported(), "an M-series host with a Metal device is supported");
}

void intel_is_unsupported() {
    auto facts = supported_host();
    facts.architecture = "x86_64";
    facts.chip = "Intel(R) Core(TM) i9";

    const auto report = assess(facts);

    check(!report.supported(), "x86_64 is unsupported");
    check(blocked_on(report, "Apple silicon"), "blocked on architecture");
    check(blocked_on(report, "M-series chip"), "blocked on chip family");
}

void a_missing_metal_device_blocks() {
    auto facts = supported_host();
    facts.metal_device_present = false;
    facts.metal_device_name.clear();

    const auto report = assess(facts);

    check(!report.supported(), "no Metal device is unsupported");
    check(blocked_on(report, "Metal device"), "blocked on the device");
}

// Memory is capacity, not capability: a small machine still runs the engine,
// it just cannot hold this model. Reporting it as unsupported would be wrong.
void small_memory_is_a_note_not_a_block() {
    auto facts = supported_host();
    facts.memory_bytes = 16ull * kGibiByte;

    const auto report = assess(facts);

    check(report.supported(), "16 GiB is still a supported host");
    bool noted = false;
    for (const auto& entry : report.checks) {
        if (entry.name.find("memory") != std::string::npos) {
            noted = !entry.passed && !entry.required;
        }
    }
    check(noted, "insufficient memory is reported as a non-required note");
}

void human_output_names_the_verdict() {
    const auto text = render_human(assess(supported_host()));
    check(text.find("verdict: supported") != std::string::npos, "human output carries a verdict");
    check(text.find("Apple M4 Pro") != std::string::npos, "human output names the chip");
}

void json_output_is_shaped_and_versioned() {
    const auto text = render_json(assess(supported_host()));
    check(text.find("\"schema_version\": 1") != std::string::npos, "json carries a schema version");
    check(text.find("\"supported\": true") != std::string::npos, "json carries the verdict");
    check(text.find("\"memory_bytes\": 51539607552") != std::string::npos, "json carries memory");
}

void json_escapes_quotes() {
    auto facts = supported_host();
    facts.chip = "Apple \"M4\" Pro";

    const auto text = render_json(assess(facts));

    check(text.find("Apple \\\"M4\\\" Pro") != std::string::npos, "quotes are escaped");
}

} // namespace

int main() {
    a_supported_host_passes();
    intel_is_unsupported();
    a_missing_metal_device_blocks();
    small_memory_is_a_note_not_a_block();
    human_output_names_the_verdict();
    json_output_is_shaped_and_versioned();
    json_escapes_quotes();
    if (failures == 0) {
        std::printf("host capability: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
