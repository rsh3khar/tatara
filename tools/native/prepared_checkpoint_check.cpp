#include "tatara/generated/model_plan.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

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

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 2 && argument_count != 3) {
        std::cerr << "usage: tatara_prepared_checkpoint_check RECORD [ARTIFACT_ROOT]\n";
        return 2;
    }
    const auto bytes = read_file(arguments[1]);
    if (bytes.empty()) {
        std::cerr << "prepared checkpoint is empty or unreadable\n";
        return 2;
    }
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    if (!parsed || !parsed.checkpoint) {
        std::cerr << "prepared checkpoint parse failed: " << static_cast<unsigned int>(parsed.error)
                  << '\n';
        return 3;
    }
    const tatara::model::PreparedCheckpointExpectation expectation = {
        .package_id = tatara::model::generated::kModelPackageId,
        .package_sha256 = tatara::model::generated::kModelPackageSha256,
        .artifact = tatara::model::generated::kArtifactIdentity,
    };
    const auto identity =
        tatara::model::validate_prepared_checkpoint_identity(*parsed.checkpoint, expectation);
    if (identity != tatara::model::PreparedCheckpointIdentityError::None) {
        std::cerr << "prepared checkpoint identity failed: " << static_cast<unsigned int>(identity)
                  << '\n';
        return 4;
    }
    if (argument_count == 3) {
        const auto opened =
            tatara::model::open_source_shards(arguments[2], parsed.checkpoint->shards());
        if (!opened || !opened.shard_set) {
            std::cerr << "source shard open failed: " << static_cast<unsigned int>(opened.error)
                      << " path=" << opened.path << " errno=" << opened.system_error << '\n';
            return 5;
        }
        std::cout << "source shards: PASS\n"
                  << "  opened: " << opened.shard_set->shards().size() << '\n';
    }
    std::cout << "prepared checkpoint: PASS\n"
              << "  shards: " << parsed.checkpoint->shards().size() << '\n'
              << "  tensors: " << parsed.checkpoint->tensors().size() << '\n'
              << "  tensor bytes: " << parsed.checkpoint->tensor_payload_bytes() << '\n';
    return 0;
}
