#include "tatara/backend/metal/model_backing.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/model_plan.h"
#include "tatara/model/image_population.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
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

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 3) {
        std::cerr << "usage: tatara_model_load_probe RECORD ARTIFACT_ROOT\n";
        return 2;
    }

    const auto record_bytes = read_file(arguments[1]);
    if (record_bytes.empty()) {
        std::cerr << "prepared checkpoint is empty or unreadable\n";
        return 3;
    }
    const auto parsed = tatara::model::parse_prepared_checkpoint(record_bytes);
    if (!parsed || !parsed.checkpoint) {
        std::cerr << "prepared checkpoint parse failed: " << static_cast<unsigned int>(parsed.error)
                  << '\n';
        return 4;
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
        return 5;
    }

    const auto layout = tatara::model::plan_image_layout(parsed.checkpoint->tensors(),
                                                         tatara::model::kTensorAlignmentBytes);
    if (!layout || !layout.layout) {
        std::cerr << "image layout planning failed: " << static_cast<unsigned int>(layout.error)
                  << '\n';
        return 6;
    }

    auto shards = tatara::model::open_source_shards(arguments[2], parsed.checkpoint->shards());
    if (!shards || !shards.shard_set) {
        std::cerr << "source shard open failed: " << static_cast<unsigned int>(shards.error)
                  << " path=" << shards.path << " errno=" << shards.system_error << '\n';
        return 7;
    }

    auto device = tatara::backend::metal::create_system_device();
    if (!device || !device.device) {
        std::cerr << "system Metal device creation failed\n";
        return 8;
    }
    const auto allocation_start = std::chrono::steady_clock::now();
    auto buffer =
        tatara::backend::metal::create_shared_buffer(*device.device, layout.layout->total_bytes);
    if (!buffer || !buffer.buffer) {
        std::cerr << "shared model buffer creation failed\n";
        return 9;
    }
    void* contents = buffer.buffer->contents();
    if (contents == nullptr) {
        std::cerr << "shared model buffer has no CPU contents\n";
        return 10;
    }
    const double allocation_seconds = seconds_since(allocation_start);

    const std::span<std::byte> destination(static_cast<std::byte*>(contents),
                                           static_cast<std::size_t>(layout.layout->total_bytes));
    const auto populate_start = std::chrono::steady_clock::now();
    const auto populated = tatara::model::populate_model_image(
        *parsed.checkpoint, *layout.layout, shards.shard_set->shards(), destination);
    if (!populated) {
        std::cerr << "population failed: " << static_cast<unsigned int>(populated.error)
                  << " shard=" << populated.shard_index << " errno=" << populated.system_error
                  << " computed=" << populated.computed_digest_hex << '\n';
        return 11;
    }
    const double populate_seconds = seconds_since(populate_start);

    auto backing = tatara::backend::metal::adopt_buffer_as_image(
        std::move(*buffer.buffer), tatara::model::kTensorAlignmentBytes);
    if (!backing || !backing.image) {
        std::cerr << "model image adoption failed: " << static_cast<unsigned int>(backing.error)
                  << " image_error=" << static_cast<unsigned int>(backing.image_error) << '\n';
        return 12;
    }

    std::cout << "model load: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  shards: " << parsed.checkpoint->shards().size() << '\n'
              << "  tensors: " << parsed.checkpoint->tensors().size() << '\n'
              << "  tensor bytes: " << parsed.checkpoint->tensor_payload_bytes() << '\n'
              << "  image bytes: " << backing.image->size_bytes() << '\n'
              << "  image alignment: " << backing.image->alignment_bytes() << '\n';
    for (std::size_t index = 0; index < populated.shard_digests_hex.size(); ++index) {
        std::cout << "  shard " << index << " sha256: " << populated.shard_digests_hex[index]
                  << '\n';
    }
    std::cout << "  allocation seconds: " << allocation_seconds << '\n'
              << "  populate seconds: " << populate_seconds << '\n'
              << "  command buffers submitted: 0\n";
    return 0;
}
