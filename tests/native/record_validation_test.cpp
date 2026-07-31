#include "tatara/service/record_validation.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace tatara::service;
using tatara::model::PreparedCheckpointExpectation;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Builds a record in the on-disk layout so the test exercises the real parser
// rather than a hand-made object; the checkpoint constructor is private to
// parse_prepared_checkpoint precisely so no other path can fabricate one.
struct RecordBuilder {
    std::vector<std::byte> bytes;

    void raw(const void* data, std::size_t size) {
        const auto* source = static_cast<const std::byte*>(data);
        bytes.insert(bytes.end(), source, source + size);
    }
    void u32(std::uint32_t value) {
        raw(&value, sizeof(value));
    }
    void u64(std::uint64_t value) {
        raw(&value, sizeof(value));
    }
    void text(const std::string& value) {
        u32(static_cast<std::uint32_t>(value.size()));
        raw(value.data(), value.size());
    }
};

const std::string kDigestA(64, 'a');
const std::string kDigestB(64, 'b');
const std::string kPackageSha(64, 'c');
const std::string kManifestSha(64, 'd');

std::vector<std::byte> build_record(const std::string& package_sha, std::uint64_t shard_bytes) {
    RecordBuilder body;
    for (const std::string& value :
         {std::string("qwen36-35b-a3b"), package_sha, std::string("qwen-artifact"), kManifestSha,
          std::string("qwen3_5_moe"), std::string("safetensors"), std::string("local"),
          std::string("main")}) {
        body.text(value);
    }
    body.text("model-00001.safetensors");
    body.text(kDigestA);
    body.u64(shard_bytes + 128);
    body.u64(128);
    body.u64(shard_bytes);

    body.text("language_model.model.embed_tokens.weight");
    body.u32(5); // BF16
    body.u32(0);
    body.u32(2);
    body.u32(0);
    body.u64(0);
    body.u64(1024);
    body.u64(32);
    body.u64(16);

    RecordBuilder record;
    record.raw("TATCKPT\0", 8);
    record.u32(1);
    record.u32(0);
    record.u64(48 + body.bytes.size());
    record.u32(1);
    record.u32(1);
    record.u32(1);
    record.u32(0);
    record.u64(1024);
    record.bytes.insert(record.bytes.end(), body.bytes.begin(), body.bytes.end());
    return record.bytes;
}

PreparedCheckpointExpectation expectation() {
    return PreparedCheckpointExpectation{
        "qwen36-35b-a3b", kPackageSha,
        tatara::model::ArtifactIdentity{.id = "qwen-artifact",
                                        .model_type = "qwen3_5_moe",
                                        .format = "safetensors",
                                        .source_repository = "local",
                                        .source_revision = "main",
                                        .manifest_sha256 = kManifestSha,
                                        .tensor_count = 1,
                                        .tensor_bytes = 1024,
                                        .file_count = 1,
                                        .weight_file_count = 1}};
}

void the_exit_contract_maps_verdicts_correctly() {
    check(verdict_exit_code(Verdict::ProductionSupported) == 0, "production-supported exits 0");
    check(verdict_exit_code(Verdict::Validated) == 0, "validated exits 0");
    check(verdict_exit_code(Verdict::Importable) == 1, "importable exits 1");
    check(verdict_exit_code(Verdict::MissingCapability) == 1, "missing-capability exits 1");
    // Capacity gets its own code: the operator's remedy is a different machine
    // or configuration, not a different artifact.
    check(verdict_exit_code(Verdict::CannotFitSafely) == 6, "cannot-fit-safely exits 6");
}

void verdict_names_are_the_vision_spelling() {
    check(verdict_name(Verdict::ProductionSupported) == "production-supported", "spelling");
    check(verdict_name(Verdict::CannotFitSafely) == "cannot-fit-safely", "spelling");
}

void a_matching_record_on_a_large_machine_is_importable() {
    const auto bytes = build_record(kPackageSha, 4096);
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    check(static_cast<bool>(parsed), "record parses");
    if (!parsed) {
        return;
    }
    const auto report =
        assess_record(*parsed.checkpoint, expectation(), 48ull * 1024 * 1024 * 1024);
    check(report.verdict == Verdict::Importable,
          "record-only identity and capacity evidence is importable, not validated");
    check(!report.findings.empty() && report.findings.front().check == "execution",
          "the missing execution evidence is named");
}

void production_support_is_never_implied() {
    const auto bytes = build_record(kPackageSha, 4096);
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    if (!parsed) {
        return;
    }
    const auto report =
        assess_record(*parsed.checkpoint, expectation(), 48ull * 1024 * 1024 * 1024);
    check(report.verdict != Verdict::ProductionSupported,
          "validate never claims production support; that needs benchmark and quality evidence");
    check(render_human(report).find("not validated") != std::string::npos,
          "the human output says what remains");
}

// The real failure this command exists to catch: a model.toml edit re-derives
// the package digest and orphans an already-prepared record.
void a_digest_mismatch_is_missing_capability() {
    const auto bytes = build_record(kDigestB, 4096);
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    if (!parsed) {
        return;
    }
    const auto report =
        assess_record(*parsed.checkpoint, expectation(), 48ull * 1024 * 1024 * 1024);
    check(report.verdict == Verdict::MissingCapability, "a foreign package is missing-capability");
    check(!report.findings.empty() && report.findings.front().check == "package-digest",
          "the finding names the package digest");
}

void a_machine_that_is_too_small_cannot_fit_safely() {
    const auto bytes = build_record(kPackageSha, 40ull * 1024 * 1024 * 1024);
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    if (!parsed) {
        return;
    }
    const auto report =
        assess_record(*parsed.checkpoint, expectation(), 16ull * 1024 * 1024 * 1024);
    check(report.verdict == Verdict::CannotFitSafely, "40 GiB of weights does not fit 16 GiB");
    check(report.provisional_lower_bound_bytes > report.weight_bytes,
          "the provisional floor exceeds raw weights because state, scratch and headroom count");
}

void unknown_memory_does_not_fabricate_a_capacity_verdict() {
    const auto bytes = build_record(kPackageSha, 40ull * 1024 * 1024 * 1024);
    const auto parsed = tatara::model::parse_prepared_checkpoint(bytes);
    if (!parsed) {
        return;
    }
    const auto report = assess_record(*parsed.checkpoint, expectation(), 0);
    check(report.verdict != Verdict::CannotFitSafely,
          "unknown memory is not evidence that it does not fit");
}

} // namespace

int main() {
    the_exit_contract_maps_verdicts_correctly();
    verdict_names_are_the_vision_spelling();
    a_matching_record_on_a_large_machine_is_importable();
    production_support_is_never_implied();
    a_digest_mismatch_is_missing_capability();
    a_machine_that_is_too_small_cannot_fit_safely();
    unknown_memory_does_not_fabricate_a_capacity_verdict();
    if (failures == 0) {
        std::printf("record validation: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
