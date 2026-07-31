#include "tatara/model/prepared_checkpoint.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tatara::model::ArtifactIdentity;
using tatara::model::parse_prepared_checkpoint;
using tatara::model::PreparedCheckpointError;
using tatara::model::PreparedCheckpointExpectation;
using tatara::model::PreparedCheckpointIdentityError;
using tatara::model::validate_prepared_checkpoint_identity;

std::vector<std::byte> read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    const std::vector<char> characters((std::istreambuf_iterator<char>(stream)),
                                       std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes;
    bytes.reserve(characters.size());
    for (const char character : characters) {
        bytes.push_back(std::byte{static_cast<unsigned char>(character)});
    }
    return bytes;
}

bool replace_first(std::vector<std::byte>& bytes, std::string_view needle,
                   std::string_view replacement) {
    if (needle.size() != replacement.size()) {
        return false;
    }
    const auto found = std::search(
        bytes.begin(), bytes.end(), needle.begin(), needle.end(), [](std::byte byte, char value) {
            return std::to_integer<unsigned char>(byte) == static_cast<unsigned char>(value);
        });
    if (found == bytes.end()) {
        return false;
    }
    for (std::size_t index = 0; index < replacement.size(); ++index) {
        found[static_cast<std::ptrdiff_t>(index)] =
            std::byte{static_cast<unsigned char>(replacement[index])};
    }
    return true;
}

int test_valid_record(const std::vector<std::byte>& bytes) {
    const auto result = parse_prepared_checkpoint(bytes);
    if (!result || !result.checkpoint) {
        return 1;
    }
    const auto& checkpoint = *result.checkpoint;
    if (checkpoint.identity().package_id != "fixture-model" || checkpoint.shards().size() != 2 ||
        checkpoint.tensors().size() != 3 || checkpoint.tensor_payload_bytes() != 28) {
        return 2;
    }
    const PreparedCheckpointExpectation expectation = {
        .package_id = "fixture-model",
        .package_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
        .artifact =
            ArtifactIdentity{
                .id = "fixture-artifact",
                .model_type = "fixture_type",
                .format = "safetensors",
                .source_repository = "fixture/repository",
                .source_revision = "3333333333333333333333333333333333333333",
                .manifest_sha256 =
                    "2222222222222222222222222222222222222222222222222222222222222222",
                .tensor_count = 3,
                .tensor_bytes = 28,
                .file_count = 4,
                .weight_file_count = 2,
            },
    };
    if (validate_prepared_checkpoint_identity(checkpoint, expectation) !=
        PreparedCheckpointIdentityError::None) {
        return 3;
    }
    auto wrong = expectation;
    wrong.package_id = "other-model";
    if (validate_prepared_checkpoint_identity(checkpoint, wrong) !=
        PreparedCheckpointIdentityError::PackageIdMismatch) {
        return 4;
    }
    return 0;
}

int test_corrupt_records(const std::vector<std::byte>& valid) {
    auto bad_magic = valid;
    bad_magic[0] = std::byte{0};
    if (parse_prepared_checkpoint(bad_magic).error != PreparedCheckpointError::InvalidMagic) {
        return 5;
    }

    auto bad_flags = valid;
    bad_flags[12] = std::byte{1};
    if (parse_prepared_checkpoint(bad_flags).error != PreparedCheckpointError::InvalidFlags) {
        return 6;
    }

    auto trailing = valid;
    trailing.push_back(std::byte{0});
    if (parse_prepared_checkpoint(trailing).error != PreparedCheckpointError::SizeMismatch) {
        return 7;
    }

    auto bad_digest = valid;
    if (!replace_first(bad_digest, std::string(64, '2'), std::string(63, '2') + "g") ||
        parse_prepared_checkpoint(bad_digest).error != PreparedCheckpointError::InvalidDigest) {
        return 8;
    }

    auto unsafe_path = valid;
    if (!replace_first(unsafe_path, "model-00001.safetensors", "/odel-00001.safetensors") ||
        parse_prepared_checkpoint(unsafe_path).error != PreparedCheckpointError::InvalidShardPath) {
        return 9;
    }

    auto duplicate_tensor = valid;
    if (!replace_first(duplicate_tensor, "tensor-b", "tensor-a")) {
        return 10;
    }
    if (parse_prepared_checkpoint(duplicate_tensor).error !=
        PreparedCheckpointError::NonCanonicalOrder) {
        return 11;
    }
    return 0;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 2) {
        return 100;
    }
    const auto bytes = read_file(arguments[1]);
    if (bytes.empty()) {
        return 101;
    }
    if (const int result = test_valid_record(bytes); result != 0) {
        return result;
    }
    return test_corrupt_records(bytes);
}
