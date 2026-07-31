#pragma once

#include "tatara/model/prepared_checkpoint.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::model {

struct OpenSourceShardsResult;

struct SourceShardView {
    std::string path;
    int borrowed_file_descriptor;
    std::uint64_t file_size_bytes;
    std::uint64_t data_offset_bytes;
    std::uint64_t data_size_bytes;
};

class SourceShardSet {
  public:
    SourceShardSet() noexcept;
    ~SourceShardSet();

    SourceShardSet(const SourceShardSet&) = delete;
    SourceShardSet& operator=(const SourceShardSet&) = delete;
    SourceShardSet(SourceShardSet&&) noexcept;
    SourceShardSet& operator=(SourceShardSet&&) noexcept;

    std::span<const SourceShardView> shards() const noexcept;
    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend OpenSourceShardsResult open_source_shards(std::string_view,
                                                     std::span<const PreparedShard>);

    SourceShardSet(std::unique_ptr<Storage> storage, std::vector<SourceShardView> views);

    std::unique_ptr<Storage> storage_;
    std::vector<SourceShardView> views_;
};

enum class SourceShardError : std::uint8_t {
    None,
    EmptyRoot,
    EmptyShardSet,
    RootOpenFailed,
    InvalidPath,
    OpenFailed,
    NotRegularFile,
    SizeMismatch,
};

struct OpenSourceShardsResult {
    SourceShardError error;
    int system_error;
    std::string path;
    std::optional<SourceShardSet> shard_set;

    explicit operator bool() const noexcept {
        return error == SourceShardError::None && shard_set.has_value();
    }
};

OpenSourceShardsResult open_source_shards(std::string_view root,
                                          std::span<const PreparedShard> shards);

} // namespace tatara::model
