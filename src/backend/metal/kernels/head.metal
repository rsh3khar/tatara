kernel void lmhead_q4(device const bfloat* x [[buffer(0)]], device const uint* words [[buffer(1)]],
                      device const bfloat* scales [[buffer(2)]],
                      device const bfloat* biases [[buffer(3)]], device bfloat* out [[buffer(4)]],
                      uint tid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kVocabularyRows) {
        return;
    }
    const ulong word_row = ulong(row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(row) * ulong(kGroupsPerRow);
    const float acc =
        q4_dot(x, words + word_row, scales + group_row, biases + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        out[row] = static_cast<bfloat>(acc);
    }
}

// Two-stage vocabulary argmax over 256 threadgroups of 256 threads. Every
// comparison carries the explicit rule "greater, or equal with lower index",
// so exact ties resolve to the global lowest index; the family rounds
// nothing beyond the bfloat16-to-float read.
kernel void logits_argmax_stage1(device const bfloat* logits [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 device float* group_values [[buffer(2)]],
                                 device uint* group_indices [[buffer(3)]],
                                 uint threadgroup_id [[threadgroup_position_in_grid]],
                                 uint thread_in_group [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage_values[8];
    threadgroup uint stage_indices[8];
    float best = -INFINITY;
    uint best_index = 0xffffffffu;
    for (uint i = threadgroup_id * 256u + thread_in_group; i < count; i += 65536u) {
        const float value = float(logits[i]);
        if (value > best) {
            best = value;
            best_index = i;
        }
    }
    for (uint off = 16u; off; off >>= 1u) {
        const float other_value = simd_shuffle_down(best, off);
        const uint other_index = simd_shuffle_down(best_index, off);
        if (other_value > best || (other_value == best && other_index < best_index)) {
            best = other_value;
            best_index = other_index;
        }
    }
    if (lane == 0u) {
        stage_values[simdgroup] = best;
        stage_indices[simdgroup] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_in_group == 0u) {
        float group_best = stage_values[0];
        uint group_best_index = stage_indices[0];
        for (uint s = 1u; s < 8u; ++s) {
            if (stage_values[s] > group_best ||
                (stage_values[s] == group_best && stage_indices[s] < group_best_index)) {
                group_best = stage_values[s];
                group_best_index = stage_indices[s];
            }
        }
        group_values[threadgroup_id] = group_best;
        group_indices[threadgroup_id] = group_best_index;
    }
}

kernel void logits_argmax_stage2(device const float* group_values [[buffer(0)]],
                                 device const uint* group_indices [[buffer(1)]],
                                 device uint* token [[buffer(2)]]) {
    float best = group_values[0];
    uint best_index = group_indices[0];
    for (uint g = 1u; g < 256u; ++g) {
        if (group_values[g] > best || (group_values[g] == best && group_indices[g] < best_index)) {
            best = group_values[g];
            best_index = group_indices[g];
        }
    }
    token[0] = best_index;
}
