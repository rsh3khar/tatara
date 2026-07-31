// Direct-only test for the DFlash draft checkpoint gate (A36 U2).
//
// Positive arm: loads the real local draft checkpoint (path via argv[1])
// and requires the complete frozen inventory, spot-checked tensor views,
// and the frozen parameter count. Adversarial arm: builds small synthetic
// safetensors files in a scratch directory (argv[2]) and requires the
// exact typed refusal for each corruption. Not registered with CTest (the
// frozen 31-target registry is unchanged); run directly.
//
// Exit codes: 0 PASS; 65 gate failure; 64 usage.

#include "tatara/draft/dflash_checkpoint.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using tatara::draft::DraftCheckpointError;

int failures = 0;

void expect(bool condition, const char* label) {
    if (!condition) {
        std::printf("FAIL: %s\n", label);
        ++failures;
    }
}

std::string write_synthetic(const std::string& directory, const std::string& name,
                            const std::string& header_json,
                            std::uint64_t payload_bytes,
                            std::uint64_t declared_header_bytes_override = 0) {
    const std::string root = directory + "/" + name;
    const std::string path = root + "/model.safetensors";
    std::string command = "mkdir -p '" + root + "'";
    std::system(command.c_str());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::uint64_t header_bytes = declared_header_bytes_override != 0
                                     ? declared_header_bytes_override
                                     : header_json.size();
    out.write(reinterpret_cast<const char*>(&header_bytes), 8);
    out.write(header_json.data(),
              static_cast<std::streamsize>(header_json.size()));
    const std::vector<char> payload(payload_bytes, '\0');
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    return root;
}

void expect_error(const std::string& root, DraftCheckpointError expected,
                  const char* label) {
    const auto load = tatara::draft::load_draft_checkpoint(root);
    if (load.error != expected) {
        std::printf("FAIL: %s -> %s (expected %s)\n", label,
                    std::string(draft_checkpoint_error_name(load.error)).c_str(),
                    std::string(draft_checkpoint_error_name(expected)).c_str());
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "usage: dflash_checkpoint_test DRAFT_ROOT SCRATCH_DIR\n");
        return 64;
    }

    // Positive arm on the real checkpoint.
    const auto load = tatara::draft::load_draft_checkpoint(argv[1]);
    expect(static_cast<bool>(load), "real checkpoint loads");
    if (load) {
        const auto fc = load.checkpoint.tensor("fc.weight");
        expect(fc.data != nullptr && fc.shape[0] == 2048 && fc.shape[1] == 16384 &&
                   fc.elements == 2048ull * 16384ull,
               "fc.weight view");
        const auto q5 = load.checkpoint.tensor("layers.5.self_attn.q_proj.weight");
        expect(q5.data != nullptr && q5.shape[0] == 4096 && q5.shape[1] == 2048,
               "layers.5 q_proj view");
        const auto norm = load.checkpoint.tensor("layers.0.self_attn.k_norm.weight");
        expect(norm.data != nullptr && norm.elements == 128, "k_norm view");
        expect(!load.checkpoint.tensor("lm_head.weight").data,
               "absent tensor yields null view");
        expect(load.checkpoint.payload_bytes() ==
                   tatara::draft::kDraftParameterCount * 2,
               "payload equals parameter count x bf16");
    }

    // Adversarial arm: synthetic corruptions, each must refuse typed.
    const std::string scratch = argv[2];
    expect_error(scratch + "/definitely-missing",
                 DraftCheckpointError::FileUnreadable, "missing file");
    {
        const auto root = write_synthetic(
            scratch, "unknown",
            R"({"mystery.weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}})",
            32);
        expect_error(root, DraftCheckpointError::UnknownTensor, "unknown tensor");
    }
    {
        const auto root = write_synthetic(
            scratch, "banned",
            R"({"lm_head.weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}})",
            32);
        expect_error(root, DraftCheckpointError::BannedTensor, "banned tensor");
    }
    {
        const auto root = write_synthetic(
            scratch, "dtype",
            R"({"fc.weight":{"dtype":"F32","shape":[2048,16384],"data_offsets":[0,134217728]}})",
            64);
        expect_error(root, DraftCheckpointError::DtypeNotBf16, "wrong dtype");
    }
    {
        const auto root = write_synthetic(
            scratch, "shape",
            R"({"fc.weight":{"dtype":"BF16","shape":[2048,999],"data_offsets":[0,4091904]}})",
            64);
        expect_error(root, DraftCheckpointError::ShapeMismatch, "wrong shape");
    }
    {
        const auto root = write_synthetic(
            scratch, "offsets",
            R"({"fc.weight":{"dtype":"BF16","shape":[2048,16384],"data_offsets":[0,1]}})",
            64);
        expect_error(root, DraftCheckpointError::OffsetsInvalid, "bad offsets");
    }
    {
        const auto root = write_synthetic(
            scratch, "missing-tensors",
            R"({"fc.weight":{"dtype":"BF16","shape":[2048,16384],"data_offsets":[0,67108864]}})",
            67108864);
        expect_error(root, DraftCheckpointError::MissingTensor, "incomplete inventory");
    }
    {
        const auto root = write_synthetic(scratch, "hdrlen", "{}", 0, 1u << 30);
        expect_error(root, DraftCheckpointError::HeaderLengthInvalid,
                     "oversized declared header");
    }
    {
        const auto root = write_synthetic(scratch, "parse", "{not json", 8, 9);
        expect_error(root, DraftCheckpointError::HeaderParse, "invalid json");
    }

    if (failures != 0) {
        std::printf("RESULT: FAIL (%d)\n", failures);
        return 65;
    }
    std::printf("RESULT: PASS — frozen inventory verified on the real draft "
                "checkpoint; 9 corruption classes refused typed.\n");
    return 0;
}
