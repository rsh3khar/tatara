#include "tatara/runtime/decode_step.h"

#include <utility>

namespace {

using namespace tatara::runtime;
using tatara::backend::metal::MetalCommandError;
using tatara::backend::metal::MetalComputePass;
using tatara::model::qwen36::LayerKind;

DecodeStateSlot slot_for(DecodeStep& step) {
    DecodeStateSlot slot;
    slot.capacity = step.capacity;
    slot.schedule_identity = step.schedule.data();
    slot.layers.resize(step.schedule.size());
    return slot;
}

} // namespace

int main() {
    DecodeStep step;
    step.capacity = 128;
    step.schedule = {
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
    };
    step.state = slot_for(step);
    DecodeStateSlot external = slot_for(step);
    if (!decode_state_slot_compatible(step, step.state) ||
        !decode_state_slot_compatible(step, external)) {
        return 1;
    }
    if (!decode_state_slot_available(step, step.state) ||
        !decode_state_slot_available(step, external) || decode_state_slot_ready(step, step.state) ||
        decode_state_slot_ready(step, external)) {
        return 2;
    }

    DecodeStateSlot wrong_capacity = slot_for(step);
    wrong_capacity.capacity = 127;
    if (decode_state_slot_compatible(step, wrong_capacity)) {
        return 3;
    }
    DecodeStateSlot wrong_count = slot_for(step);
    wrong_count.layers.pop_back();
    if (decode_state_slot_compatible(step, wrong_count)) {
        return 4;
    }
    DecodeStep other;
    other.capacity = step.capacity;
    other.schedule = step.schedule;
    DecodeStateSlot wrong_owner = slot_for(other);
    if (decode_state_slot_compatible(step, wrong_owner)) {
        return 5;
    }

    // std::vector move transfers its allocation. Slots retain the exact
    // schedule-data identity across the DecodeStep move used by factories,
    // optionals, and the boot harness.
    DecodeStep moved = std::move(step);
    if (!decode_state_slot_compatible(moved, moved.state) ||
        !decode_state_slot_compatible(moved, external)) {
        return 6;
    }

    MetalComputePass invalid_pass;
    if (encode_token(moved, wrong_owner, invalid_pass, 0) != MetalCommandError::StateSlotMismatch) {
        return 7;
    }
    if (encode_token(moved, external, invalid_pass, moved.capacity) !=
        MetalCommandError::ContextOutOfRange) {
        return 8;
    }
    if (encode_token(moved, external, invalid_pass, 0) != MetalCommandError::StateSlotMismatch) {
        return 9;
    }

    advance_decode_state(moved, external);
    if (!external.layers[0].swapped || external.layers[1].swapped || !external.layers[2].swapped) {
        return 10;
    }
    external.status = DecodeStateSlotStatus::SnapshotPending;
    external.active_transfer_generation = 1;
    if (decode_state_slot_available(moved, external)) {
        return 11;
    }
    advance_decode_state(moved, external);
    if (!external.layers[0].swapped || external.layers[1].swapped || !external.layers[2].swapped) {
        return 12;
    }
    advance_decode_state(moved, wrong_owner);
    if (wrong_owner.layers[0].swapped || wrong_owner.layers[1].swapped ||
        wrong_owner.layers[2].swapped) {
        return 13;
    }

    return 0;
}
