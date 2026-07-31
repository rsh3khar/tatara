#pragma once

#include "tatara/model/sha256.h"
#include "tatara/runtime/prefill_command_plan.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace tatara::runtime {

inline constexpr std::uint32_t kExecutionIdentitySchemaVersion = 1;
inline constexpr std::uint32_t kDecodeExecutionSchemaVersion = 2;
inline constexpr std::uint32_t kDecodeBindingSchemaVersion = 1;

struct DecodePipelines;
struct DecodeStep;
struct PrefillExecutionPolicy;
struct PrefillPipelines;

// Stable identity serialization is domain-separated, length-prefixed and
// explicitly little-endian. It never hashes native padding or raw bool bytes.
class ExecutionIdentityEncoder {
  public:
    explicit ExecutionIdentityEncoder(std::string_view domain) noexcept;

    void append_u8(std::uint8_t value) noexcept;
    void append_u32(std::uint32_t value) noexcept;
    void append_u64(std::uint64_t value) noexcept;
    void append_bool(bool value) noexcept;
    void append_text(std::string_view value) noexcept;
    void append_bytes(std::span<const std::byte> value) noexcept;
    void append_identity(const PrefillCommandIdentity& value) noexcept;
    PrefillCommandIdentity finish() noexcept;

  private:
    model::Sha256 hash_;
};

PrefillCommandIdentity execution_model_package_identity() noexcept;
PrefillCommandIdentity execution_model_plan_identity() noexcept;
PrefillCommandIdentity execution_tokenizer_identity() noexcept;
PrefillCommandIdentity execution_host_identity() noexcept;
PrefillCommandIdentity
execution_prepared_record_identity(std::span<const std::byte> record) noexcept;
PrefillCommandIdentity execution_prepared_image_identity(const DecodeStep& decode) noexcept;
PrefillCommandIdentity execution_decode_pipeline_identity(
    const DecodePipelines& pipelines) noexcept;
PrefillCommandIdentity execution_prefill_pipeline_identity(
    const PrefillPipelines& pipelines) noexcept;
PrefillCommandIdentity prefill_execution_policy_identity(
    const PrefillExecutionPolicy& policy) noexcept;
PrefillCommandIdentity execution_kernel_library_identity() noexcept;
PrefillCommandIdentity
execution_pipeline_identity(std::span<const std::uint64_t> identities) noexcept;

} // namespace tatara::runtime
