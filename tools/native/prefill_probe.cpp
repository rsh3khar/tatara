#include "decode_harness.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_step.h"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

// Serial teacher-forced prefill (eighth boundary contract in
// the component execution contract).
//
// Drives the sealed single-token decode step at contexts 0..N-1 from a cold
// cache, injecting the prompt's own token id before each step instead of
// feeding back the argmax. No new kernel and no block execution: this composes
// machinery already proven byte-exact, and the coverage it adds is
// accumulating state from an EMPTY cache rather than from a loaded record,
// which nothing in Phase 3 had exercised.
//
// The parity target is the decode path teacher-forced over the same
// ids -- the same kernel set. Its own prefill path binds *_blk kernels, which
// are different Metal functions, so gating against those would be a false gate
// wearing a byte-exact label.

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using tatara::model::qwen36::LayerKind;

constexpr std::uint32_t kMaxPositions = 4096;

// A long walk blocks in the wait with almost no CPU duty, so a guard kill
// would otherwise leave an empty log and no way to tell a wedge from slow
// progress. Progress is flushed every partition's worth of positions, which
// also puts a marker exactly where the attention path changes numeric family.
constexpr std::uint32_t kProgressInterval = 256;

// The guard owns 2 (wall timeout), 3 (CPU-stall wedge) and 4 (not executable),
// and passes the probe's own code through unchanged, so a probe that returns
// any of them is indistinguishable from a guard kill. Under a policy where one
// wedge halts all autonomous work, that ambiguity is costly in both
// directions. Boot failures are lifted into a band of their own; everything
// this probe raises itself sits at 60 or above.
constexpr int kBootExitBase = 40;
constexpr int kExitUsage = 60;
constexpr int kExitIdsUnreadable = 61;
constexpr int kExitIdsInvalid = 62;
constexpr int kExitCapacity = 63;
constexpr int kExitCommandBuffer = 70;
constexpr int kExitComputePass = 71;
constexpr int kExitEncode = 72;
constexpr int kExitEndPass = 73;
constexpr int kExitCommit = 74;
constexpr int kExitExecution = 75;
constexpr int kExitTokenRange = 76;
constexpr int kExitDump = 77;
constexpr int kExitOccupancy = 78;
constexpr int kExitStalled = 79;
constexpr int kExitBoundaryArguments = 64;
constexpr int kExitBoundaryDump = 80;
constexpr int kExitBoundaryStalled = 81;

struct PromptIds {
    bool valid = false;
    std::vector<std::uint32_t> ids;
};

// The ids file is a u32 count followed by that many u32 token ids.
PromptIds parse_prompt_ids(std::span<const std::byte> bytes, std::uint32_t vocabulary) {
    PromptIds parsed;
    if (bytes.size() < 4) {
        std::cerr << "ids file is shorter than its header\n";
        return parsed;
    }
    std::uint32_t count = 0;
    std::memcpy(&count, bytes.data(), 4);
    if (count == 0 || count > kMaxPositions) {
        std::cerr << "ids count " << count << " outside 1.." << kMaxPositions << '\n';
        return parsed;
    }
    if (bytes.size() != 4 + std::size_t{count} * 4) {
        std::cerr << "ids file is " << bytes.size() << " bytes, expected "
                  << 4 + std::size_t{count} * 4 << " for " << count << " ids\n";
        return parsed;
    }
    parsed.ids.resize(count);
    std::memcpy(parsed.ids.data(), bytes.data() + 4, std::size_t{count} * 4);
    for (std::size_t index = 0; index < parsed.ids.size(); ++index) {
        if (parsed.ids[index] >= vocabulary) {
            std::cerr << "prompt id " << index << " is " << parsed.ids[index]
                      << ", outside the vocabulary\n";
            return parsed;
        }
    }
    parsed.valid = true;
    return parsed;
}

// Equality alone is not a passing fixture, and a teacher-forced walk is
// especially easy to pass vacuously: the injected ids are inputs, so they
// prove nothing about what ran. A walk that silently did nothing would leave
// the cache exactly as allocated -- all zeros -- and still write a
// well-formed record. Every state plane must therefore carry at least one
// nonzero byte, which is the degeneracy and evolution gate in one: starting
// cold, a nonzero plane can only have been produced by the walk.
struct Occupancy {
    std::size_t planes = 0;
    std::size_t filled = 0;
    std::uint64_t nonzero_bytes = 0;
    std::uint64_t total_bytes = 0;
};

bool any_nonzero(const void* data, std::uint64_t bytes, std::uint64_t& nonzero) {
    const auto* cursor = static_cast<const unsigned char*>(data);
    bool seen = false;
    for (std::uint64_t index = 0; index < bytes; ++index) {
        if (cursor[index] != 0) {
            ++nonzero;
            seen = true;
        }
    }
    return seen;
}

Occupancy measure_occupancy(const DecodeStep& step, std::span<const LayerKind> schedule,
                            std::uint32_t capacity, std::uint32_t positions) {
    Occupancy report;
    const std::uint64_t head_bytes = std::uint64_t{positions} * 256 * 2;
    const std::uint64_t cache_head_stride = std::uint64_t{capacity} * 256 * 2;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        const DecodeLayerState& state = step.state.layers[layer];
        if (schedule[layer] == LayerKind::GatedDelta) {
            // Scan the same planes the record writes, or a stale-plane defect
            // would pass the gate and still be dumped.
            const bool live_out = tatara::tools::gated_delta_live_is_out(state.swapped);
            const struct {
                const void* data;
                std::uint64_t bytes;
            } planes[] = {
                {live_out ? state.first_out.contents() : state.first.contents(),
                 step.geometry.gdn_conv_state_bytes},
                {live_out ? state.second_out.contents() : state.second.contents(),
                 step.geometry.gdn_recurrent_state_bytes},
            };
            for (const auto& plane : planes) {
                ++report.planes;
                report.total_bytes += plane.bytes;
                report.filled += any_nonzero(plane.data, plane.bytes, report.nonzero_bytes) ? 1 : 0;
            }
        } else {
            for (const MetalBuffer* plane : {&state.first, &state.second}) {
                for (std::uint64_t head = 0; head < 2; ++head) {
                    ++report.planes;
                    report.total_bytes += head_bytes;
                    const auto* base =
                        static_cast<const std::byte*>(plane->contents()) + head * cache_head_stride;
                    report.filled += any_nonzero(base, head_bytes, report.nonzero_bytes) ? 1 : 0;
                }
            }
        }
    }
    return report;
}

// ---- layer-boundary capture.
//
// The eighth boundary's state record captures layer *states*; the divergence
// it found at N=2048 originates in a layer *output*, which the state record
// does not hold. Both engines already retain every inter-layer hidden in
// host-visible buffers with identical geometry, so the instrument is a pure
// host read after the completion wait -- no new dispatch, no kernel change and
// nothing on the encode path moves. The record layout (`Q36BND01`) is frozen
// by the component execution contract and must match the decode path's
// byte-for-byte, including this hash, or the comparison is meaningless.

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kBoundaryStreams = 4;

// FNV-1a 64 folded over the little-endian u64 words of one layer's hidden
// block. hidden_bytes is 4096, a multiple of 8, so no tail case exists.
std::uint64_t boundary_hash(const void* data, std::uint64_t bytes) {
    std::uint64_t hash = kFnvOffsetBasis;
    const auto* words = static_cast<const std::uint64_t*>(data);
    for (std::uint64_t index = 0; index < bytes / 8; ++index) {
        hash ^= words[index];
        hash *= kFnvPrime;
    }
    return hash;
}

bool write_boundary_record(const char* path, std::uint32_t positions, std::uint32_t layers,
                           std::uint32_t hidden, std::uint32_t window_first,
                           std::uint32_t window_count, std::span<const std::uint64_t> hashes,
                           std::span<const std::byte> raw) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "cannot open boundary record " << path << '\n';
        return false;
    }
    const std::uint32_t header[6] = {positions,        layers,       hidden,
                                     kBoundaryStreams, window_first, window_count};
    out.write("Q36BND01", 8);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(hashes.data()),
              static_cast<std::streamsize>(hashes.size_bytes()));
    if (!raw.empty()) {
        out.write(reinterpret_cast<const char*>(raw.data()),
                  static_cast<std::streamsize>(raw.size_bytes()));
    }
    // The size is validated rather than assumed, exactly as the state record
    // does: a short write must be a typed failure, not a truncated artifact
    // that a comparator would happily read.
    const std::streamoff written = out.tellp();
    out.close();
    const std::streamoff expected = 32 + static_cast<std::streamoff>(hashes.size_bytes()) +
                                    static_cast<std::streamoff>(raw.size_bytes());
    if (!out || written != expected) {
        std::cerr << "boundary record is " << written << " bytes, expected " << expected << '\n';
        return false;
    }
    return true;
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 4 || argument_count > 8) {
        std::cerr << "usage: tatara_prefill_probe RECORD ARTIFACT_ROOT IDS [DUMP_OUT]"
                     " [BOUNDARY_OUT [BOUNDARY_FIRST [BOUNDARY_COUNT]]]\n";
        return kExitUsage;
    }
    const char* const dump_path = argument_count >= 5 ? arguments[4] : nullptr;
    const char* const boundary_path = argument_count >= 6 ? arguments[5] : nullptr;
    std::uint32_t boundary_first = 0;
    std::uint32_t boundary_count = 0;
    for (int index = 6; index < argument_count; ++index) {
        const std::string_view text(arguments[index]);
        std::uint32_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
            std::cerr << "boundary window argument " << index << " is not a number: " << text
                      << '\n';
            return kExitBoundaryArguments;
        }
        (index == 6 ? boundary_first : boundary_count) = value;
    }

    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    const auto ids_bytes = tatara::tools::read_file(arguments[3]);
    if (ids_bytes.empty()) {
        std::cerr << "ids file is empty or unreadable\n";
        return kExitIdsUnreadable;
    }
    const auto prompt = parse_prompt_ids(ids_bytes, plan.dimensions.vocabulary);
    if (!prompt.valid) {
        return kExitIdsInvalid;
    }

    auto harness = tatara::tools::boot_decode(arguments[1], arguments[2]);
    if (!harness) {
        return kBootExitBase + harness.exit_code;
    }
    DecodeStep& step = *harness.step;
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    const auto positions = static_cast<std::uint32_t>(prompt.ids.size());
    if (positions > harness.capacity) {
        std::cerr << "prompt of " << positions << " exceeds capacity " << harness.capacity << '\n';
        return kExitCapacity;
    }

    // Boundary capture sized against the actual walk. A window past the last
    // step would record zeros for steps that never ran, and on a guarded
    // window that is a wasted launch reporting a match that never happened.
    const auto layers = static_cast<std::uint32_t>(schedule.size());
    const std::uint64_t hidden_bytes = step.geometry.hidden_bytes;
    const std::uint64_t boundary_stream_bytes = std::uint64_t{layers} * hidden_bytes;
    std::vector<std::uint64_t> boundary_hashes;
    std::vector<std::byte> boundary_raw;
    if (boundary_path != nullptr) {
        if (std::uint64_t{boundary_first} + boundary_count > positions) {
            std::cerr << "boundary window " << boundary_first << "+" << boundary_count
                      << " extends past the last step " << positions << '\n';
            return kExitBoundaryArguments;
        }
        // The record's shape is proven from the geometry rather than assumed,
        // so a layout change upstream fails here instead of silently writing a
        // record the comparator would misread.
        if (step.geometry.layer_stream_bytes != boundary_stream_bytes) {
            std::cerr << "layer stream is " << step.geometry.layer_stream_bytes
                      << " bytes, expected " << boundary_stream_bytes << " for " << layers
                      << " layers\n";
            return kExitBoundaryArguments;
        }
        boundary_hashes.assign(std::uint64_t{positions} * kBoundaryStreams * layers, 0);
        boundary_raw.assign(
            std::uint64_t{boundary_count} * kBoundaryStreams * boundary_stream_bytes, std::byte{0});
    }

    // The step is constructed with every state buffer zeroed, so a freshly
    // booted harness IS the cold cache; there is nothing to clear.
    const auto prefill_start = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> predictions;
    predictions.reserve(positions);
    std::uint32_t submissions = 0;
    for (std::uint32_t index = 0; index < positions; ++index) {
        // Injection happens before the step, so the embed reads the prompt's
        // id rather than the argmax the previous step wrote back.
        std::memcpy(step.token_id.contents(), &prompt.ids[index], 4);
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return kExitCommandBuffer;
        }
        auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return kExitComputePass;
        }
        if (encode_token(step, *pass.compute_pass, index) != MetalCommandError::None) {
            std::cerr << "encode_token failed at position " << index << '\n';
            return kExitEncode;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return kExitEndPass;
        }
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return kExitCommit;
        }
        ++submissions;
        if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
            !execution) {
            std::cerr << "position " << index
                      << " execution failed: " << execution.failure_description.view() << '\n';
            return kExitExecution;
        }
        const std::uint32_t produced = *static_cast<std::uint32_t*>(step.token_id.contents());
        if (produced >= plan.dimensions.vocabulary) {
            std::cerr << "position " << index << ": out of range prediction " << produced << '\n';
            return kExitTokenRange;
        }
        predictions.push_back(produced);
        if ((index + 1) % kProgressInterval == 0 || index + 1 == positions) {
            std::cout << "  prefilled " << (index + 1) << "/" << positions << " positions, "
                      << "partitions now "
                      << ((index + 1 + kProgressInterval - 1) / kProgressInterval)
                      << ", last prediction " << produced << '\n'
                      << std::flush;
        }
        // Layer-BOUNDARY capture. wait_until_completed has already returned
        // and every buffer is StorageModeShared, so these host reads see the
        // finished GPU writes. Pure reads, taken before the gated-delta swap:
        // no dispatch, no ordering and no arithmetic is touched, and all four
        // streams are rewritten in full every step so nothing here is stale.
        if (boundary_path != nullptr) {
            const MetalBuffer* const streams[kBoundaryStreams] = {
                &step.branch_stream, &step.residual_stream, &step.moe_stream, &step.layer_stream};
            for (std::uint32_t stream = 0; stream < kBoundaryStreams; ++stream) {
                const auto* base = static_cast<const std::byte*>(streams[stream]->contents());
                for (std::uint32_t layer = 0; layer < layers; ++layer) {
                    boundary_hashes[(std::uint64_t{index} * kBoundaryStreams + stream) * layers +
                                    layer] =
                        boundary_hash(base + std::uint64_t{layer} * hidden_bytes, hidden_bytes);
                }
            }
            if (index >= boundary_first && index < boundary_first + boundary_count) {
                const std::uint64_t slot = index - boundary_first;
                for (std::uint32_t stream = 0; stream < kBoundaryStreams; ++stream) {
                    std::memcpy(boundary_raw.data() +
                                    (slot * kBoundaryStreams + stream) * boundary_stream_bytes,
                                streams[stream]->contents(), boundary_stream_bytes);
                }
            }
        }
        advance_decode_state(step);
    }
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();

    const Occupancy occupancy = measure_occupancy(step, schedule, harness.capacity, positions);
    std::size_t stalled_layer = 0;
    const bool advanced = tatara::tools::gated_delta_advanced(step, schedule, stalled_layer);

    // Teacher-forcing discards the model's own prediction at every position.
    // Those discarded ids are the only correctness signal this probe carries,
    // because the inputs were supplied rather than produced -- so they are
    // printed for comparison against the reference engine's.
    std::cout << "prefill predictions (discarded by teacher forcing, the cross-check):\n";
    for (std::uint32_t index = 0; index < positions; ++index) {
        std::cout << "  position " << index << ": input " << prompt.ids[index] << " -> predicted "
                  << predictions[index] << '\n';
    }

    const bool occupied = occupancy.filled == occupancy.planes;
    const bool passed = occupied && advanced;
    std::cout << (passed ? "serial prefill: PASS\n" : "serial prefill: FAIL\n")
              << "  device: " << harness.device->name() << '\n'
              << "  positions: " << positions << ", capacity: " << harness.capacity << '\n'
              << "  image load: " << harness.load_seconds << " s, prefill: " << prefill_seconds
              << " s\n"
              << "  state planes occupied: " << occupancy.filled << "/" << occupancy.planes << '\n'
              << "  state nonzero bytes: " << occupancy.nonzero_bytes << "/"
              << occupancy.total_bytes << '\n'
              << "  gated-delta advanced on the final step: " << (advanced ? "yes" : "no") << '\n'
              << "  command buffers submitted: " << submissions << '\n';

    // The dump runs even on a failed gate. A divergent or stalled state is
    // diagnostic, the window is spent either way, and discarding the artifact
    // would mean re-spending a guarded window just to see what went wrong.
    if (dump_path != nullptr) {
        if (!tatara::tools::write_state_record(dump_path, step, schedule, harness.capacity,
                                               positions)) {
            return kExitDump;
        }
        std::cout << "  post-prefill states: written to " << dump_path << " at " << positions
                  << " positions\n";
    }

    if (boundary_path != nullptr) {
        // The same degeneracy discipline the state dump carries: a walk that
        // silently did nothing would still write a well-formed record, so an
        // unvarying hash column is a typed failure rather than an artifact to
        // be compared and believed.
        const std::uint64_t stride = std::uint64_t{kBoundaryStreams} * layers;
        bool evolved = false;
        for (std::uint64_t index = stride; index < boundary_hashes.size(); ++index) {
            if (boundary_hashes[index] != boundary_hashes[index - stride]) {
                evolved = true;
                break;
            }
        }
        if (positions > 1 && !evolved) {
            std::cerr << "every step hashes identically; the boundary streams did not evolve\n";
            return kExitBoundaryStalled;
        }
        if (!write_boundary_record(boundary_path, positions, layers,
                                   static_cast<std::uint32_t>(hidden_bytes / 2), boundary_first,
                                   boundary_count, boundary_hashes, boundary_raw)) {
            return kExitBoundaryDump;
        }
        std::cout << "  layer boundaries: written to " << boundary_path << " at " << positions
                  << " steps, " << layers << " layers, " << kBoundaryStreams << " streams, window "
                  << boundary_first << "+" << boundary_count << ", evolved yes\n";
    }

    if (!occupied) {
        std::cerr
            << "a state plane is still zero after the walk; the prefill did not populate it\n";
        return kExitOccupancy;
    }
    if (!advanced) {
        std::cerr << "gated-delta layer " << stalled_layer
                  << " is identical to its previous-step state; the walk stalled\n";
        return kExitStalled;
    }
    return 0;
}
