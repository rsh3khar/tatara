#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tatara::host {

// Weights plus persistent state, scratch and OS headroom for the first
// supported slice. Capacity is reported against this rather than against raw
// installed memory, because raw memory is not the number that decides whether
// a model runs.
inline constexpr std::uint64_t kGibiByte = 1024ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kFirstSliceRequiredBytes = 32ull * kGibiByte;

struct HostFacts {
    std::string system;
    std::string os_version;
    std::string architecture;
    std::string chip;
    std::uint64_t memory_bytes{0};
    std::uint32_t performance_cores{0};
    std::uint32_t efficiency_cores{0};
    bool metal_device_present{false};
    std::string metal_device_name;
    std::uint64_t metal_recommended_working_set_bytes{0};
    std::uint64_t metal_maximum_buffer_bytes{0};
};

struct Check {
    std::string name;
    bool required{false};
    bool passed{false};
    std::string detail;
};

struct CapabilityReport {
    HostFacts facts;
    std::vector<Check> checks;

    bool supported() const;
};

// Reads the host. The only side effect is sysctl and one Metal device query.
HostFacts read_host_facts();

// Pure: turns facts into checks and a verdict.
CapabilityReport assess(const HostFacts& facts);

std::string render_human(const CapabilityReport& report);
std::string render_json(const CapabilityReport& report);

} // namespace tatara::host
