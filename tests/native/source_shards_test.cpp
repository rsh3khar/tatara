#include "tatara/model/source_shards.h"

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using tatara::model::open_source_shards;
using tatara::model::PreparedShard;
using tatara::model::SourceShardError;
using tatara::model::SourceShardSet;

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::error_code error;
            const auto candidate = std::filesystem::temp_directory_path() /
                                   ("tatara-source-shards-" + std::to_string(::getpid()) + "-" +
                                    std::to_string(attempt));
            if (std::filesystem::create_directory(candidate, error)) {
                path_ = candidate;
                break;
            }
        }
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

  private:
    std::filesystem::path path_;
};

PreparedShard shard(std::string path, std::uint64_t file_size_bytes) {
    return {
        .path = std::move(path),
        .sha256 = std::string(64, 'a'),
        .file_size_bytes = file_size_bytes,
        .data_offset_bytes = 10,
        .data_size_bytes = file_size_bytes >= 10 ? file_size_bytes - 10 : 0,
    };
}

void write_file(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

int test_failures(const std::filesystem::path& root) {
    if (open_source_shards(root.string(), std::span<const PreparedShard>{}).error !=
        SourceShardError::EmptyShardSet) {
        return 1;
    }
    const std::array missing = {shard("missing.safetensors", 10)};
    if (open_source_shards(root.string(), missing).error != SourceShardError::OpenFailed) {
        return 2;
    }

    write_file(root / "wrong.safetensors", "1234567890");
    const std::array wrong = {shard("wrong.safetensors", 11)};
    if (open_source_shards(root.string(), wrong).error != SourceShardError::SizeMismatch) {
        return 3;
    }

    std::filesystem::create_directory(root / "directory.safetensors");
    const std::array directory = {shard("directory.safetensors", 0)};
    if (open_source_shards(root.string(), directory).error != SourceShardError::NotRegularFile) {
        return 4;
    }

    write_file(root / "target.safetensors", "1234567890");
    std::filesystem::create_symlink("target.safetensors", root / "link.safetensors");
    const std::array link = {shard("link.safetensors", 10)};
    if (open_source_shards(root.string(), link).error != SourceShardError::OpenFailed) {
        return 5;
    }

    std::filesystem::create_directory(root / "nested");
    write_file(root / "nested" / "file.safetensors", "1234567890");
    std::filesystem::create_directory_symlink("nested", root / "alias");
    const std::array nested_link = {shard("alias/file.safetensors", 10)};
    if (open_source_shards(root.string(), nested_link).error != SourceShardError::OpenFailed) {
        return 6;
    }

    const std::array absolute = {shard("/tmp/outside.safetensors", 10)};
    if (open_source_shards(root.string(), absolute).error != SourceShardError::InvalidPath) {
        return 7;
    }
    return 0;
}

int test_ownership(const std::filesystem::path& root) {
    std::filesystem::create_directory(root / "owned");
    write_file(root / "owned" / "a.safetensors", "1234567890");
    write_file(root / "b.safetensors", "123456789012");
    const std::array shards = {
        shard("b.safetensors", 12),
        shard("owned/a.safetensors", 10),
    };
    auto opened = open_source_shards(root.string(), shards);
    if (!opened || !opened.shard_set || opened.shard_set->shards().size() != 2) {
        return 8;
    }
    const int first_descriptor = opened.shard_set->shards()[0].borrowed_file_descriptor;
    const int second_descriptor = opened.shard_set->shards()[1].borrowed_file_descriptor;
    if (::fcntl(first_descriptor, F_GETFD) < 0 || ::fcntl(second_descriptor, F_GETFD) < 0) {
        return 9;
    }
    {
        SourceShardSet first = std::move(*opened.shard_set);
        SourceShardSet second = std::move(first);
        if (first || !second || second.shards()[1].path != "owned/a.safetensors") {
            return 10;
        }
        SourceShardSet third;
        third = std::move(second);
        if (second || !third || third.shards()[0].borrowed_file_descriptor != first_descriptor) {
            return 11;
        }
    }
    errno = 0;
    if (::fcntl(first_descriptor, F_GETFD) != -1 || errno != EBADF) {
        return 12;
    }
    errno = 0;
    if (::fcntl(second_descriptor, F_GETFD) != -1 || errno != EBADF) {
        return 13;
    }
    return 0;
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<SourceShardSet>);
    static_assert(!std::is_copy_assignable_v<SourceShardSet>);
    static_assert(std::is_nothrow_move_constructible_v<SourceShardSet>);
    static_assert(std::is_nothrow_move_assignable_v<SourceShardSet>);

    TemporaryDirectory temporary;
    if (temporary.path().empty()) {
        return 100;
    }
    if (const int result = test_failures(temporary.path()); result != 0) {
        return result;
    }
    return test_ownership(temporary.path());
}
