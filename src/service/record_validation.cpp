#include "tatara/service/record_validation.h"

#include <sstream>

namespace tatara::service {
namespace {

using model::PreparedCheckpointIdentityError;

// This is an intentionally conservative lower bound, not the complete capacity
// model VISION.md promises. If even this lower bound exceeds memory, refusal is
// sound. Passing it is not evidence that the configured slots/context fit.
constexpr double kProvisionalStateAndScratchFactor = 1.18;
constexpr std::uint64_t kOperatingSystemHeadroomBytes = 6ull * 1024 * 1024 * 1024;

std::string_view identity_error_name(PreparedCheckpointIdentityError error) {
    switch (error) {
    case PreparedCheckpointIdentityError::None:
        return "none";
    case PreparedCheckpointIdentityError::PackageIdMismatch:
        return "package-id";
    case PreparedCheckpointIdentityError::PackageDigestMismatch:
        return "package-digest";
    case PreparedCheckpointIdentityError::ArtifactIdMismatch:
        return "artifact-id";
    case PreparedCheckpointIdentityError::ManifestDigestMismatch:
        return "manifest-digest";
    case PreparedCheckpointIdentityError::ModelTypeMismatch:
        return "model-type";
    case PreparedCheckpointIdentityError::FormatMismatch:
        return "format";
    case PreparedCheckpointIdentityError::RepositoryMismatch:
        return "source-repository";
    case PreparedCheckpointIdentityError::RevisionMismatch:
        return "source-revision";
    case PreparedCheckpointIdentityError::ArtifactFileCountMismatch:
        return "artifact-file-count";
    case PreparedCheckpointIdentityError::WeightFileCountMismatch:
        return "weight-file-count";
    case PreparedCheckpointIdentityError::TensorCountMismatch:
        return "tensor-count";
    case PreparedCheckpointIdentityError::TensorBytesMismatch:
        return "tensor-bytes";
    }
    return "unknown";
}

std::string gibibytes(std::uint64_t bytes) {
    std::ostringstream out;
    out.precision(2);
    out << std::fixed << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return out.str();
}

} // namespace

std::string_view verdict_name(Verdict verdict) {
    switch (verdict) {
    case Verdict::ProductionSupported:
        return "production-supported";
    case Verdict::Validated:
        return "validated";
    case Verdict::Importable:
        return "importable";
    case Verdict::MissingCapability:
        return "missing-capability";
    case Verdict::CannotFitSafely:
        return "cannot-fit-safely";
    }
    return "unknown";
}

int verdict_exit_code(Verdict verdict) {
    switch (verdict) {
    case Verdict::ProductionSupported:
    case Verdict::Validated:
        return 0;
    case Verdict::Importable:
    case Verdict::MissingCapability:
        return 1;
    case Verdict::CannotFitSafely:
        return 6;
    }
    return 1;
}

RecordReport assess_record(const model::PreparedCheckpoint& checkpoint,
                           const model::PreparedCheckpointExpectation& expectation,
                           std::uint64_t available_bytes) {
    RecordReport report;
    report.available_bytes = available_bytes;
    for (const auto& shard : checkpoint.shards()) {
        report.weight_bytes += shard.data_size_bytes;
    }
    report.provisional_lower_bound_bytes =
        static_cast<std::uint64_t>(static_cast<double>(report.weight_bytes) *
                                   kProvisionalStateAndScratchFactor) +
        kOperatingSystemHeadroomBytes;

    const auto identity = validate_prepared_checkpoint_identity(checkpoint, expectation);
    if (identity != PreparedCheckpointIdentityError::None) {
        report.findings.push_back(
            {std::string(identity_error_name(identity)),
             "the record does not describe the model this build was generated for"});
        // A record for a different package is not a capacity question and not
        // a broken file: the engine simply has no plan for it.
        report.verdict = Verdict::MissingCapability;
        return report;
    }

    if (available_bytes != 0 && report.provisional_lower_bound_bytes > available_bytes) {
        report.findings.push_back({"capacity", gibibytes(report.provisional_lower_bound_bytes) +
                                                   " needed, " + gibibytes(available_bytes) +
                                                   " present"});
        report.verdict = Verdict::CannotFitSafely;
        return report;
    }

    // Identity and the provisional memory lower bound hold. The binding
    // definition of `validated` additionally requires execution, tokenizer and
    // quality gates, none of which a record-only command has run.
    report.findings.push_back(
        {"execution", "record identity passed; execution, tokenizer and quality were not run"});
    report.verdict = Verdict::Importable;
    return report;
}

std::string render_human(const RecordReport& report) {
    std::ostringstream out;
    out << "weights: " << gibibytes(report.weight_bytes) << "\n";
    out << "provisional lower bound: " << gibibytes(report.provisional_lower_bound_bytes)
        << " including estimated state, scratch and OS headroom\n";
    if (report.available_bytes != 0) {
        out << "available: " << gibibytes(report.available_bytes) << "\n";
    }
    for (const auto& finding : report.findings) {
        out << "  " << finding.check << ": " << finding.detail << "\n";
    }
    out << "verdict: " << verdict_name(report.verdict);
    if (report.verdict == Verdict::Importable) {
        out << " — not validated until execution, tokenizer and quality gates pass";
    }
    out << "\n";
    return out.str();
}

} // namespace tatara::service
