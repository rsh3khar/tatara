#include "tatara/host/capability.h"

#import <Metal/Metal.h>

#include <sys/sysctl.h>
#include <sys/types.h>

#include <cstdio>
#include <sstream>

namespace tatara::host {
namespace {

std::string sysctl_string(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::uint64_t sysctl_unsigned(const char* name) {
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
        return 0;
    }
    return value;
}

std::string gibibytes(std::uint64_t bytes) {
    std::ostringstream out;
    out.precision(1);
    out << std::fixed << static_cast<double>(bytes) / static_cast<double>(kGibiByte) << " GiB";
    return out.str();
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (char character : value) {
        if (character == '"' || character == '\\') {
            out.push_back('\\');
        }
        out.push_back(character);
    }
    return out;
}

} // namespace

bool CapabilityReport::supported() const {
    for (const auto& check : checks) {
        if (check.required && !check.passed) {
            return false;
        }
    }
    return true;
}

HostFacts read_host_facts() {
    HostFacts facts;
    facts.system = sysctl_string("kern.ostype");
    facts.os_version = sysctl_string("kern.osproductversion");
    facts.chip = sysctl_string("machdep.cpu.brand_string");
    facts.memory_bytes = sysctl_unsigned("hw.memsize");
    facts.performance_cores =
        static_cast<std::uint32_t>(sysctl_unsigned("hw.perflevel0.logicalcpu"));
    facts.efficiency_cores =
        static_cast<std::uint32_t>(sysctl_unsigned("hw.perflevel1.logicalcpu"));

    // hw.machine reports the kernel architecture, which is what decides
    // whether the engine can run at all.
    facts.architecture = sysctl_string("hw.machine");

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device != nil) {
            facts.metal_device_present = true;
            facts.metal_device_name = device.name.UTF8String;
            facts.metal_recommended_working_set_bytes =
                device.recommendedMaxWorkingSetSize;
            facts.metal_maximum_buffer_bytes = device.maxBufferLength;
        }
    }
    return facts;
}

CapabilityReport assess(const HostFacts& facts) {
    CapabilityReport report;
    report.facts = facts;
    const bool apple_silicon = facts.architecture == "arm64";
    const bool m_series = facts.chip.find("Apple M") != std::string::npos;

    report.checks.push_back(
        {"macOS", true, facts.system == "Darwin", facts.system.empty() ? "unknown" : facts.system});
    report.checks.push_back({"Apple silicon", true, apple_silicon,
                             facts.architecture.empty() ? "unknown" : facts.architecture});
    report.checks.push_back(
        {"M-series chip", true, m_series, facts.chip.empty() ? "unknown" : facts.chip});
    report.checks.push_back({"Metal device", true, facts.metal_device_present,
                             facts.metal_device_present ? facts.metal_device_name : "none"});
    report.checks.push_back({"memory for the first supported slice", false,
                             facts.memory_bytes >= kFirstSliceRequiredBytes,
                             facts.memory_bytes == 0
                                 ? "unknown"
                                 : gibibytes(facts.memory_bytes) + " of " +
                                       gibibytes(kFirstSliceRequiredBytes) + " needed"});
    return report;
}

std::string render_human(const CapabilityReport& report) {
    std::ostringstream out;
    const auto& facts = report.facts;
    out << "host: " << (facts.chip.empty() ? "unknown" : facts.chip);
    if (facts.memory_bytes != 0) {
        out << ", " << gibibytes(facts.memory_bytes) << " unified";
    }
    if (!facts.os_version.empty()) {
        out << ", macOS " << facts.os_version;
    }
    out << "\n";
    if (facts.performance_cores != 0 || facts.efficiency_cores != 0) {
        out << "cores: " << facts.performance_cores << " performance, " << facts.efficiency_cores
            << " efficiency\n";
    }
    out << "metal: " << (facts.metal_device_present ? facts.metal_device_name : "no device")
        << "\n";
    if (facts.metal_recommended_working_set_bytes != 0) {
        out << "metal recommended working set: "
            << gibibytes(facts.metal_recommended_working_set_bytes)
            << "\n";
    }
    if (facts.metal_maximum_buffer_bytes != 0) {
        out << "metal maximum buffer: "
            << gibibytes(facts.metal_maximum_buffer_bytes) << "\n";
    }
    for (const auto& check : report.checks) {
        if (!check.passed) {
            out << (check.required ? "  BLOCKED " : "  note    ") << check.name << ": "
                << check.detail << "\n";
        }
    }
    out << "verdict: " << (report.supported() ? "supported" : "unsupported") << "\n";
    return out.str();
}

std::string render_json(const CapabilityReport& report) {
    std::ostringstream out;
    const auto& facts = report.facts;
    out << "{\n  \"schema_version\": 1,\n";
    out << "  \"chip\": \"" << json_escape(facts.chip) << "\",\n";
    out << "  \"os_version\": \"" << json_escape(facts.os_version) << "\",\n";
    out << "  \"architecture\": \"" << json_escape(facts.architecture) << "\",\n";
    out << "  \"memory_bytes\": " << facts.memory_bytes << ",\n";
    out << "  \"performance_cores\": " << facts.performance_cores << ",\n";
    out << "  \"efficiency_cores\": " << facts.efficiency_cores << ",\n";
    out << "  \"metal_device\": \"" << json_escape(facts.metal_device_name) << "\",\n";
    out << "  \"metal_recommended_working_set_bytes\": "
        << facts.metal_recommended_working_set_bytes << ",\n";
    out << "  \"metal_maximum_buffer_bytes\": "
        << facts.metal_maximum_buffer_bytes << ",\n";
    out << "  \"checks\": [\n";
    for (std::size_t index = 0; index < report.checks.size(); ++index) {
        const auto& check = report.checks[index];
        out << "    {\"name\": \"" << json_escape(check.name)
            << "\", \"required\": " << (check.required ? "true" : "false")
            << ", \"passed\": " << (check.passed ? "true" : "false") << ", \"detail\": \""
            << json_escape(check.detail) << "\"}"
            << (index + 1 == report.checks.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"supported\": " << (report.supported() ? "true" : "false") << "\n}\n";
    return out.str();
}

} // namespace tatara::host
