#include "tatara/model/source_shards.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <utility>

namespace tatara::model {
namespace {

class UniqueFile {
  public:
    UniqueFile() noexcept = default;
    explicit UniqueFile(int descriptor) noexcept : descriptor_(descriptor) {}

    ~UniqueFile() {
        reset();
    }

    UniqueFile(const UniqueFile&) = delete;
    UniqueFile& operator=(const UniqueFile&) = delete;

    UniqueFile(UniqueFile&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}

    UniqueFile& operator=(UniqueFile&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.descriptor_, -1));
        }
        return *this;
    }

    int get() const noexcept {
        return descriptor_;
    }

    explicit operator bool() const noexcept {
        return descriptor_ >= 0;
    }

    void reset(int descriptor = -1) noexcept {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
        descriptor_ = descriptor;
    }

  private:
    int descriptor_ = -1;
};

OpenSourceShardsResult failure(SourceShardError error, std::string_view path,
                               int system_error = 0) {
    return {
        .error = error,
        .system_error = system_error,
        .path = std::string(path),
        .shard_set = std::nullopt,
    };
}

bool valid_relative_path(std::string_view path) noexcept {
    if (path.empty() || path.front() == '/' || path.back() == '/' || path.find('\0') != path.npos ||
        path.find('\\') != path.npos) {
        return false;
    }
    std::size_t begin = 0;
    while (begin < path.size()) {
        const std::size_t end = path.find('/', begin);
        const std::string_view part = path.substr(begin, end - begin);
        if (part.empty() || part == "." || part == "..") {
            return false;
        }
        if (end == path.npos) {
            break;
        }
        begin = end + 1;
    }
    return true;
}

UniqueFile open_relative_file(int root_descriptor, std::string_view path, int& system_error) {
    UniqueFile directory;
    int directory_descriptor = root_descriptor;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = path.find('/', begin);
        const std::string component(path.substr(begin, end - begin));
        if (end == path.npos) {
            const int descriptor = ::openat(directory_descriptor, component.c_str(),
                                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            if (descriptor < 0) {
                system_error = errno;
            }
            return UniqueFile(descriptor);
        }
        const int next = ::openat(directory_descriptor, component.c_str(),
                                  O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY);
        if (next < 0) {
            system_error = errno;
            return UniqueFile();
        }
        directory.reset(next);
        directory_descriptor = directory.get();
        begin = end + 1;
    }
}

} // namespace

struct SourceShardSet::Storage {
    std::vector<UniqueFile> files;
};

SourceShardSet::SourceShardSet() noexcept = default;
SourceShardSet::~SourceShardSet() = default;
SourceShardSet::SourceShardSet(SourceShardSet&&) noexcept = default;
SourceShardSet& SourceShardSet::operator=(SourceShardSet&&) noexcept = default;

SourceShardSet::SourceShardSet(std::unique_ptr<Storage> storage, std::vector<SourceShardView> views)
    : storage_(std::move(storage)), views_(std::move(views)) {}

std::span<const SourceShardView> SourceShardSet::shards() const noexcept {
    return views_;
}

SourceShardSet::operator bool() const noexcept {
    return storage_ != nullptr && storage_->files.size() == views_.size() && !views_.empty();
}

OpenSourceShardsResult open_source_shards(std::string_view root,
                                          std::span<const PreparedShard> shards) {
    if (root.empty() || root.find('\0') != root.npos) {
        return failure(SourceShardError::EmptyRoot, root);
    }
    if (shards.empty()) {
        return failure(SourceShardError::EmptyShardSet, {});
    }
    const std::string root_path(root);
    UniqueFile root_directory(::open(root_path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
    if (!root_directory) {
        return failure(SourceShardError::RootOpenFailed, root, errno);
    }

    auto storage = std::make_unique<SourceShardSet::Storage>();
    storage->files.reserve(shards.size());
    std::vector<SourceShardView> views;
    views.reserve(shards.size());
    for (const PreparedShard& shard : shards) {
        if (!valid_relative_path(shard.path)) {
            return failure(SourceShardError::InvalidPath, shard.path);
        }
        int system_error = 0;
        UniqueFile file = open_relative_file(root_directory.get(), shard.path, system_error);
        if (!file) {
            return failure(SourceShardError::OpenFailed, shard.path, system_error);
        }
        struct stat status{};
        if (::fstat(file.get(), &status) != 0) {
            return failure(SourceShardError::OpenFailed, shard.path, errno);
        }
        if (!S_ISREG(status.st_mode)) {
            return failure(SourceShardError::NotRegularFile, shard.path);
        }
        if (status.st_size < 0 ||
            static_cast<std::uint64_t>(status.st_size) != shard.file_size_bytes) {
            return failure(SourceShardError::SizeMismatch, shard.path);
        }
        const int borrowed_file_descriptor = file.get();
        storage->files.push_back(std::move(file));
        views.push_back({
            .path = shard.path,
            .borrowed_file_descriptor = borrowed_file_descriptor,
            .file_size_bytes = shard.file_size_bytes,
            .data_offset_bytes = shard.data_offset_bytes,
            .data_size_bytes = shard.data_size_bytes,
        });
    }
    return {
        .error = SourceShardError::None,
        .system_error = 0,
        .path = {},
        .shard_set = SourceShardSet(std::move(storage), std::move(views)),
    };
}

} // namespace tatara::model
