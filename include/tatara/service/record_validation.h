#pragma once

#include "tatara/model/prepared_checkpoint.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tatara::service {

// The five outcomes for any artifact presented to Tatara.
// Closed set: a command that cannot complete its gates reports no verdict at
// all rather than guessing one.
enum class Verdict : std::uint8_t {
    ProductionSupported,
    Validated,
    Importable,
    MissingCapability,
    CannotFitSafely,
};

struct RecordFinding {
    std::string check;
    std::string detail;
};

struct RecordReport {
    Verdict verdict{Verdict::MissingCapability};
    std::vector<RecordFinding> findings;
    std::uint64_t weight_bytes{0};
    // This record-only floor may prove a non-fit but never proves a fit. The
    // complete runtime capacity model replaces it before `validated` exists.
    std::uint64_t provisional_lower_bound_bytes{0};
    std::uint64_t available_bytes{0};
};

std::string_view verdict_name(Verdict verdict);

// Exit code for a verdict, per the exit contract: 0 and 1 carry a verdict,
// and capacity gets its own code because the operator's remedy differs.
int verdict_exit_code(Verdict verdict);

// Pure. Takes a parsed record plus the compiled expectation and the machine's
// memory, and returns the verdict with the findings behind it.
RecordReport assess_record(const model::PreparedCheckpoint& checkpoint,
                           const model::PreparedCheckpointExpectation& expectation,
                           std::uint64_t available_bytes);

std::string render_human(const RecordReport& report);

} // namespace tatara::service
