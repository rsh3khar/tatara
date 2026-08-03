#include "fixture_batteries.h"

#include <iostream>
#include <span>
#include <string_view>

// Safe-math fixture suite (seventh boundary contract,
// the component execution contract).
//
// Re-derives the EXEC1-EXEC5 semantics bank against the library that actually
// ships. established that every adjudication earned before
// it is a property of the fast-math library: a fixture compares a kernel
// against an observation kernel from the same library, so both move together
// under either math mode and agree. The bank is therefore not wrong so much as
// not-yet-evidence, and this suite is what converts it back.
//
// One binary rather than six, because every probe binary already changed SHA
// under MATH1 (pipeline.mm links into all of them), so no re-run-unchanged path
// exists. Six probes would cost six adversarial reviews and six launches at a
// serialized, human-gated resource whose expense is the cycle around it, not
// the ten seconds of GPU time.

namespace {

struct Battery {
    const char* name;
    int index; // Fixes the exit-code namespace; never renumber.
    int (*run)();
    std::span<const std::string_view> pins_of_record;
};

// Pins of record, as earned under fast math. These are printed beside each
// battery's own observations so a single log carries record and result
// adjacently. The HELD/MOVED determination is an analysis step recorded in the
// ledger rather than a computation here: extracting each battery's observed
// winner as data would require editing the battery bodies, and the boundary
// contract forbids that -- byte-identical battery semantics are what make the
// comparison against the archived fast-math logs meaningful at all.

constexpr std::string_view kKernelPins[] = {
    "precise::rsqrt            device-defined via point query (matched neither candidate)",
    "float arithmetic          flush-to-zero on operands and results",
    "multiply-add              fused contraction",
    "simd_sum                  adjacent-pairs binary tree",
    "bfloat multiply           single rounding under FTZ",
};

constexpr std::string_view kGdnPins[] = {
    "bfloat add                single rounding under FTZ",
    "recurrence chain          contracts to fused FMA",
    "Q4 composite              device-defined via same-source observation   [PREDICTED MOVER]",
    "transcendental composites device-defined via same-source observation",
};

constexpr std::string_view kAttentionPins[] = {
    "exp / trig                device-queried at fixture arguments",
    "value accumulation        fused FMA, pinned exactly-one",
    "partitioned merge         byte-equal to the single kernel",
};

constexpr std::string_view kMoePins[] = {
    "router selection ties     resolve by tree position (bit-reversed index)  [TIE1]",
    "float32 division          device-defined; both CPU candidates inexact    [DIV1, PREDICTED "
    "MOVER]",
    "packed-Q4 and Q8 dots     device-defined via same-source observation     [PREDICTED MOVER]",
    "slot accumulation         degenerate at fixtures; pins to observed mirror",
};

constexpr std::string_view kHeadPins[] = {
    "argmax ties               global lowest index (explicit index compare per merge)",
    "rounding                  none beyond bfloat16 widening; zero device queries",
};

constexpr std::string_view kCompositionPins[] = {
    "no adjudication           dual-execution equality; self-consistent under either math mode",
};

constexpr Battery kBatteries[] = {
    {"kernel", 1, run_kernel_battery, kKernelPins},
    {"gdn", 2, run_gdn_battery, kGdnPins},
    {"attention", 3, run_attention_battery, kAttentionPins},
    {"moe", 4, run_moe_battery, kMoePins},
    {"head", 5, run_head_battery, kHeadPins},
    {"composition", 6, run_composition_battery, kCompositionPins},
};

// The process exit status is truncated to 8 bits by POSIX and the guard
// propagates only those bits, so the exit cannot carry family and battery code
// together: batteries already return codes above 99, and a composite like
// head 12 -> 512 truncates to 0, reporting a failed battery as PASS. Worse,
// several composites alias onto the guard's own statuses -- 2 wall timeout,
// 3 CPU-stall wedge, 4 binary not executable -- which would make a clean typed
// failure read as a wedge and halt all autonomous work, or the reverse.
//
// So the exit identifies the failing FAMILY only, in a band that collides with
// nothing: 0 for a clean sweep, 64 + family index otherwise. The full
// family * 100 + code diagnostic is printed in the summary, where it has no
// width limit and cannot be truncated.
constexpr int kExitNamespaceStride = 100;
constexpr int kExitFamilyBase = 64;

void print_rule() {
    std::cout << "------------------------------------------------------------\n";
}

} // namespace

int main() {
    std::cout << "safe-math fixture suite: re-deriving the EXEC1-EXEC5 semantics bank\n"
              << "compilation mode: MTLMathModeSafe\n"
              << "batteries: 6, independent, continue-on-mismatch\n";

    int results[std::size(kBatteries)] = {};
    int first_failure = 0;
    int failed = 0;

    for (std::size_t slot = 0; slot < std::size(kBatteries); ++slot) {
        const Battery& battery = kBatteries[slot];

        print_rule();
        std::cout << "battery " << battery.index << "/6: " << battery.name << '\n'
                  << "pins of record (fast-math library):\n";
        for (const std::string_view pin : battery.pins_of_record) {
            std::cout << "  " << pin << '\n';
        }
        std::cout << "observations under safe math follow:\n";
        std::cout.flush();

        // A battery that mismatches does not stop the suite: all six run and all
        // six verdicts are reported, preserving the per-family diagnostic
        // granularity that six separate windows would have bought, at one
        // launch. A wedge is categorically different -- it kills the process,
        // and the standing rule then halts all autonomous work.
        const int code = battery.run();
        results[slot] = code;

        if (code != 0) {
            ++failed;
            if (first_failure == 0) {
                first_failure = battery.index;
            }
            std::cout << battery.name << " battery: TYPED FAIL code " << code << " (diagnostic "
                      << battery.index * kExitNamespaceStride + code << ")\n";
        } else {
            std::cout << battery.name << " battery: PASS\n";
        }
        std::cout.flush();
    }

    print_rule();
    std::cout << "suite summary\n";
    for (std::size_t slot = 0; slot < std::size(kBatteries); ++slot) {
        const Battery& battery = kBatteries[slot];
        const int code = results[slot];
        std::cout << "  " << battery.index << " " << battery.name << ": "
                  << (code == 0 ? "PASS" : "FAIL") << " code " << code;
        if (code != 0) {
            std::cout << " diagnostic " << battery.index * kExitNamespaceStride + code;
        }
        std::cout << '\n';
    }

    if (failed != 0) {
        std::cout << "suite: " << failed << " of 6 batteries failed; exit "
                  << kExitFamilyBase + first_failure << " names the first failing family\n";
        std::cout
            << "NOTE: a moved pin is an expected outcome of MATH1, not a defect. Compare each\n"
            << "      battery's observations against its printed record before concluding.\n";
        return kExitFamilyBase + first_failure;
    }

    std::cout << "suite: PASS 6/6 -- every battery byte-exact under safe math\n";
    return 0;
}
