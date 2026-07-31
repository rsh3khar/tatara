#pragma once

#include <cstdint>
#include <string_view>

namespace tatara::model {

struct ArtifactIdentity {
    std::string_view id;
    std::string_view model_type;
    std::string_view format;
    std::string_view source_repository;
    std::string_view source_revision;
    std::string_view manifest_sha256;
    std::uint32_t tensor_count;
    std::uint64_t tensor_bytes;
    std::uint32_t file_count;
    std::uint32_t weight_file_count;
};

} // namespace tatara::model
