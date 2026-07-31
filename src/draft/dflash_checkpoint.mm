#include "tatara/draft/dflash_checkpoint.h"

#import <Foundation/Foundation.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace tatara::draft {

namespace {

constexpr std::uint64_t kMaximumHeaderBytes = 8ull * 1024 * 1024;

struct ExpectedTensor {
    std::string_view name;
    std::array<std::uint32_t, 2> shape;
};

// The complete frozen inventory (DFLASH_DRAFT_SPEC.md section 1.3): three
// top-level tensors plus eleven per layer. Vectors are recorded as [n, 1].
constexpr std::array<ExpectedTensor, 3> kTopLevel{{
    {"fc.weight", {2048, 16384}},
    {"hidden_norm.weight", {2048, 1}},
    {"norm.weight", {2048, 1}},
}};

constexpr std::array<ExpectedTensor, 11> kPerLayer{{
    {"input_layernorm.weight", {2048, 1}},
    {"self_attn.q_proj.weight", {4096, 2048}},
    {"self_attn.k_proj.weight", {1024, 2048}},
    {"self_attn.v_proj.weight", {1024, 2048}},
    {"self_attn.o_proj.weight", {2048, 4096}},
    {"self_attn.q_norm.weight", {128, 1}},
    {"self_attn.k_norm.weight", {128, 1}},
    {"post_attention_layernorm.weight", {2048, 1}},
    {"mlp.gate_proj.weight", {6144, 2048}},
    {"mlp.up_proj.weight", {6144, 2048}},
    {"mlp.down_proj.weight", {2048, 6144}},
}};

constexpr std::array<std::string_view, 4> kBannedFragments{
    "embed_tokens", "lm_head", "bias", "rotary"};

bool expected_shape(std::string_view name, std::array<std::uint32_t, 2>& shape) {
    for (const ExpectedTensor& tensor : kTopLevel) {
        if (name == tensor.name) {
            shape = tensor.shape;
            return true;
        }
    }
    constexpr std::string_view prefix = "layers.";
    if (name.substr(0, prefix.size()) != prefix) {
        return false;
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos || dot == prefix.size()) {
        return false;
    }
    const std::string_view layer_text = name.substr(prefix.size(), dot - prefix.size());
    if (layer_text.empty() || layer_text.size() > 1 ||
        layer_text[0] < '0' || layer_text[0] > '5') {
        return false;
    }
    const std::string_view stem = name.substr(dot + 1);
    for (const ExpectedTensor& tensor : kPerLayer) {
        if (stem == tensor.name) {
            shape = tensor.shape;
            return true;
        }
    }
    return false;
}

struct Mapping {
    void* address{nullptr};
    std::uint64_t bytes{0};
};

void unmap(Mapping mapping) {
    if (mapping.address != nullptr && mapping.bytes != 0) {
        ::munmap(mapping.address, mapping.bytes);
    }
}

} // namespace

std::string_view draft_checkpoint_error_name(DraftCheckpointError error) noexcept {
    switch (error) {
        case DraftCheckpointError::None: return "None";
        case DraftCheckpointError::FileUnreadable: return "FileUnreadable";
        case DraftCheckpointError::HeaderTruncated: return "HeaderTruncated";
        case DraftCheckpointError::HeaderLengthInvalid: return "HeaderLengthInvalid";
        case DraftCheckpointError::HeaderParse: return "HeaderParse";
        case DraftCheckpointError::TensorEntryMalformed: return "TensorEntryMalformed";
        case DraftCheckpointError::TensorCount: return "TensorCount";
        case DraftCheckpointError::UnknownTensor: return "UnknownTensor";
        case DraftCheckpointError::MissingTensor: return "MissingTensor";
        case DraftCheckpointError::BannedTensor: return "BannedTensor";
        case DraftCheckpointError::DtypeNotBf16: return "DtypeNotBf16";
        case DraftCheckpointError::ShapeMismatch: return "ShapeMismatch";
        case DraftCheckpointError::OffsetsInvalid: return "OffsetsInvalid";
        case DraftCheckpointError::PayloadMismatch: return "PayloadMismatch";
        case DraftCheckpointError::ParameterCount: return "ParameterCount";
    }
    return "Unknown";
}

DraftCheckpoint::DraftCheckpoint(DraftCheckpoint&& other) noexcept {
    *this = std::move(other);
}

DraftCheckpoint& DraftCheckpoint::operator=(DraftCheckpoint&& other) noexcept {
    if (this != &other) {
        unmap(Mapping{mapping_, mapping_bytes_});
        mapping_ = other.mapping_;
        mapping_bytes_ = other.mapping_bytes_;
        payload_bytes_ = other.payload_bytes_;
        entries_ = std::move(other.entries_);
        other.mapping_ = nullptr;
        other.mapping_bytes_ = 0;
        other.payload_bytes_ = 0;
        other.entries_.clear();
    }
    return *this;
}

DraftCheckpoint::~DraftCheckpoint() {
    unmap(Mapping{mapping_, mapping_bytes_});
}

DraftTensorView DraftCheckpoint::tensor(std::string_view name) const noexcept {
    for (const Entry& entry : entries_) {
        if (entry.name == name) {
            return DraftTensorView{
                .data = static_cast<const std::byte*>(mapping_) + entry.offset,
                .elements = entry.elements,
                .shape = entry.shape,
            };
        }
    }
    return DraftTensorView{};
}

DraftCheckpointLoad load_draft_checkpoint(std::string_view root) {
    DraftCheckpointLoad result;
    const std::string path = std::string(root) + "/model.safetensors";

    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        result.error = DraftCheckpointError::FileUnreadable;
        result.detail = path;
        return result;
    }
    struct stat file_stat {};
    if (::fstat(descriptor, &file_stat) != 0 || file_stat.st_size < 8) {
        ::close(descriptor);
        result.error = DraftCheckpointError::HeaderTruncated;
        result.detail = path;
        return result;
    }
    const std::uint64_t file_bytes = static_cast<std::uint64_t>(file_stat.st_size);
    void* address = ::mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
    ::close(descriptor);
    if (address == MAP_FAILED) {
        result.error = DraftCheckpointError::FileUnreadable;
        result.detail = path;
        return result;
    }
    Mapping mapping{address, file_bytes};

    std::uint64_t header_bytes = 0;
    std::memcpy(&header_bytes, address, 8);
    if (header_bytes == 0 || header_bytes > kMaximumHeaderBytes ||
        8 + header_bytes > file_bytes) {
        unmap(mapping);
        result.error = DraftCheckpointError::HeaderLengthInvalid;
        return result;
    }
    const std::uint64_t payload_bytes = file_bytes - 8 - header_bytes;

    NSData* header_data =
        [NSData dataWithBytesNoCopy:static_cast<std::uint8_t*>(address) + 8
                             length:static_cast<NSUInteger>(header_bytes)
                       freeWhenDone:NO];
    NSError* json_error = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:header_data
                                                options:0
                                                  error:&json_error];
    if (json_error != nil || ![parsed isKindOfClass:[NSDictionary class]]) {
        unmap(mapping);
        result.error = DraftCheckpointError::HeaderParse;
        return result;
    }
    NSDictionary* header = static_cast<NSDictionary*>(parsed);

    std::vector<DraftCheckpoint::Entry> entries;
    entries.reserve(kDraftTensorCount);
    std::uint64_t parameters = 0;
    DraftCheckpointError error = DraftCheckpointError::None;
    std::string detail;

    for (NSString* key in header) {
        const std::string name = key.UTF8String;
        if (name == "__metadata__") {
            continue;
        }
        for (const std::string_view banned : kBannedFragments) {
            if (name.find(banned) != std::string::npos) {
                error = DraftCheckpointError::BannedTensor;
                detail = name;
            }
        }
        if (error != DraftCheckpointError::None) {
            break;
        }
        std::array<std::uint32_t, 2> shape{};
        if (!expected_shape(name, shape)) {
            error = DraftCheckpointError::UnknownTensor;
            detail = name;
            break;
        }
        id value = header[key];
        if (![value isKindOfClass:[NSDictionary class]]) {
            error = DraftCheckpointError::TensorEntryMalformed;
            detail = name;
            break;
        }
        NSDictionary* entry = static_cast<NSDictionary*>(value);
        NSString* dtype = entry[@"dtype"];
        if (![dtype isKindOfClass:[NSString class]] || ![dtype isEqualToString:@"BF16"]) {
            error = DraftCheckpointError::DtypeNotBf16;
            detail = name;
            break;
        }
        NSArray* dims = entry[@"shape"];
        NSArray* offsets = entry[@"data_offsets"];
        if (![dims isKindOfClass:[NSArray class]] ||
            ![offsets isKindOfClass:[NSArray class]] || offsets.count != 2 ||
            dims.count < 1 || dims.count > 2) {
            error = DraftCheckpointError::TensorEntryMalformed;
            detail = name;
            break;
        }
        std::array<std::uint32_t, 2> actual{0, 1};
        for (NSUInteger index = 0; index < dims.count; ++index) {
            actual[index] =
                static_cast<std::uint32_t>([dims[index] unsignedLongLongValue]);
        }
        if (actual != shape) {
            error = DraftCheckpointError::ShapeMismatch;
            detail = name;
            break;
        }
        const std::uint64_t begin = [offsets[0] unsignedLongLongValue];
        const std::uint64_t end = [offsets[1] unsignedLongLongValue];
        const std::uint64_t elements =
            std::uint64_t{shape[0]} * std::uint64_t{shape[1]};
        if (end < begin || end > payload_bytes || end - begin != elements * 2) {
            error = DraftCheckpointError::OffsetsInvalid;
            detail = name;
            break;
        }
        parameters += elements;
        entries.push_back(DraftCheckpoint::Entry{
            .name = name,
            .offset = 8 + header_bytes + begin,
            .elements = elements,
            .shape = shape,
        });
    }

    if (error == DraftCheckpointError::None && entries.size() != kDraftTensorCount) {
        error = entries.size() < kDraftTensorCount
                    ? DraftCheckpointError::MissingTensor
                    : DraftCheckpointError::TensorCount;
    }
    if (error == DraftCheckpointError::None) {
        // Contiguous, non-overlapping coverage of the payload.
        std::vector<DraftCheckpoint::Entry> ordered = entries;
        std::sort(ordered.begin(), ordered.end(),
                  [](const DraftCheckpoint::Entry& a, const DraftCheckpoint::Entry& b) {
                      return a.offset < b.offset;
                  });
        std::uint64_t cursor = 8 + header_bytes;
        for (const DraftCheckpoint::Entry& entry : ordered) {
            if (entry.offset != cursor) {
                error = DraftCheckpointError::PayloadMismatch;
                detail = entry.name;
                break;
            }
            cursor = entry.offset + entry.elements * 2;
        }
        if (error == DraftCheckpointError::None && cursor != file_bytes) {
            error = DraftCheckpointError::PayloadMismatch;
        }
    }
    if (error == DraftCheckpointError::None && parameters != kDraftParameterCount) {
        error = DraftCheckpointError::ParameterCount;
    }

    if (error != DraftCheckpointError::None) {
        unmap(mapping);
        result.error = error;
        result.detail = detail;
        return result;
    }
    result.checkpoint.mapping_ = mapping.address;
    result.checkpoint.mapping_bytes_ = mapping.bytes;
    result.checkpoint.payload_bytes_ = payload_bytes;
    result.checkpoint.entries_ = std::move(entries);
    return result;
}

} // namespace tatara::draft
