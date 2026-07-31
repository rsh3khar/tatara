// Causal width-four convolution over one block. The logical sequence is the
// three incoming state columns followed by block projection rows. Only the
// last admitted row commits the outgoing convolution state.
kernel void gdn_conv_blk(device const bfloat* projection [[buffer(0)]],
                         device const bfloat* conv_state [[buffer(1)]],
                         device const bfloat* conv_weights [[buffer(2)]],
                         device bfloat* qk [[buffer(3)]],
                         device bfloat* value_out [[buffer(4)]],
                         device bfloat* gate_out [[buffer(5)]],
                         device bfloat* new_conv_state [[buffer(6)]],
                         constant uint& state_step [[buffer(7)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint2 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup float activation[kGdnHeadDimension];
    threadgroup float inverse_shared[1];
    const uint lane = thread_position.x;
    const uint position = group.y;
    device const bfloat* projection_row =
        projection + position * kGdnProjectionRows;
    const uint conv_groups = kGdnConvChannels / kGdnHeadDimension;
    if (group.x < conv_groups) {
        const uint channel = group.x * kGdnHeadDimension + lane;
        float taps[kGdnConvWidth];
        for (uint tap = 0u; tap < kGdnConvWidth; ++tap) {
            const uint logical = position + tap;
            taps[tap] =
                logical < kGdnConvWidth - 1u
                    ? float(conv_state[logical * kGdnConvChannels + channel])
                    : float(projection[(logical - (kGdnConvWidth - 1u)) *
                                           kGdnProjectionRows +
                                       channel]);
        }
        const ulong weight_base = ulong(channel) * kGdnConvWidth;
        // Preserve gdn_prepare's exact four-term evaluation shape. A loop
        // beginning at zero is mathematically equivalent but is not the same
        // floating-point source contract.
        const float convolved =
            float(conv_weights[weight_base]) * taps[0] +
            float(conv_weights[weight_base + 1u]) * taps[1] +
            float(conv_weights[weight_base + 2u]) * taps[2] +
            float(conv_weights[weight_base + 3u]) * taps[3];
        const bfloat convolved_b = static_cast<bfloat>(convolved);
        const bfloat exp_b = static_cast<bfloat>(exp(fabs(float(convolved_b))));
        const bfloat sigmoid_low = bfloat(1.0f) / (bfloat(1.0f) + exp_b);
        const bfloat sigmoid_b =
            float(convolved_b) < 0.0f
                ? sigmoid_low
                : static_cast<bfloat>(bfloat(1.0f) - sigmoid_low);
        const bfloat activated_b = convolved_b * sigmoid_b;
        if (position == state_step - 1u) {
            for (uint tap = 0u; tap < kGdnConvWidth - 1u; ++tap) {
                new_conv_state[tap * kGdnConvChannels + channel] =
                    static_cast<bfloat>(taps[tap + 1u]);
            }
        }
        if (group.x < kGdnQkValues / kGdnHeadDimension) {
            activation[lane] = float(activated_b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < kSimdgroupWidth) {
                const uint base = lane * 4u;
                float total = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const float value = activation[base + i];
                    total += value * value;
                }
                total = simd_sum(total);
                if (lane == 0u) {
                    inverse_shared[0] = precise::rsqrt(
                        total / float(kGdnHeadDimension) + kRmsNormEpsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bfloat normalized =
                static_cast<bfloat>(activation[lane] * inverse_shared[0]);
            const bfloat scale =
                group.x < kGdnKeyHeads ? bfloat(kGdnQueryScale) : bfloat(kGdnKeyScale);
            qk[position * kGdnQkValues + channel] = normalized * scale;
        } else {
            value_out[position * kGdnValueValues + channel - kGdnQkValues] =
                activated_b;
        }
    } else {
        const uint index =
            (group.x - conv_groups) * kGdnHeadDimension + lane;
        gate_out[position * kGdnValueValues + index] =
            projection_row[kGdnQkvRows + index];
    }
}

// Gate/β hoist for the register-loop diagnostic. The scalar source chain is
// identical to that loop but is evaluated once per (position, head) rather
// than once per value dimension. C1's conservative gate uses the permanent
// one-step recurrence pipeline and does not select this optimization.
kernel void gdn_gates_blk(
    device const bfloat* projection [[buffer(0)]],
    device const bfloat* a_log [[buffer(1)]],
    device const bfloat* dt_bias [[buffer(2)]],
    device float* decay_out [[buffer(3)]],
    device float* beta_out [[buffer(4)]],
    constant uint& steps [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint value_head = position.x;
    const uint step = position.y;
    if (value_head >= kGdnValueHeads || step >= steps) {
        return;
    }
    device const bfloat* projection_row =
        projection + step * kGdnProjectionRows;
    const bfloat a_b = projection_row[kGdnARowOffset + value_head];
    const bfloat b_b = projection_row[kGdnBRowOffset + value_head];
    const bfloat shifted = a_b + dt_bias[value_head];
    const bfloat branch_max =
        float(shifted) > 0.0f ? shifted : bfloat(0.0f);
    const bfloat branch_min =
        float(shifted) > 0.0f ? bfloat(0.0f) : shifted;
    const bfloat difference = branch_min - branch_max;
    const bfloat exp_b = static_cast<bfloat>(exp(float(difference)));
    const float one_plus = 1.0f + float(exp_b);
    const bfloat log1p_b =
        one_plus == 1.0f
            ? exp_b
            : static_cast<bfloat>(
                  float(exp_b) * (log(one_plus) / (one_plus - 1.0f)));
    const bfloat softplus_b = branch_max + log1p_b;
    decay_out[step * kGdnValueHeads + value_head] =
        exp(-exp(float(a_log[value_head])) * float(softplus_b));
    const bfloat beta_exp =
        static_cast<bfloat>(exp(fabs(float(b_b))));
    const bfloat beta_low =
        bfloat(1.0f) / (bfloat(1.0f) + beta_exp);
    beta_out[step * kGdnValueHeads + value_head] =
        float(b_b) < 0.0f
            ? float(beta_low)
            : float(static_cast<bfloat>(bfloat(1.0f) - beta_low));
}

kernel void gdn_recurrence_gates_blk(
    device const bfloat* qk [[buffer(0)]],
    device const bfloat* value [[buffer(1)]],
    device const float* decay [[buffer(2)]],
    device const bfloat* a_log [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device const float* state_in [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant uint& steps [[buffer(8)]],
    uint3 position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    (void)a_log;
    const uint value_dimension = position.y;
    const uint value_head = position.z;
    const uint key_head = value_head >> 1u;
    const uint query_base = key_head * kGdnHeadDimension;
    const uint key_base = kGdnKOffset + query_base;
    const ulong state_base =
        (ulong(value_head) * kGdnHeadDimension + value_dimension) *
        kGdnHeadDimension;
    float state[4];
    for (uint i = 0u; i < 4u; ++i) {
        state[i] = state_in[state_base + lane * 4u + i];
    }
    for (uint step = 0u; step < steps; ++step) {
        device const bfloat* qk_row = qk + step * kGdnQkValues;
        const float step_decay =
            decay[step * kGdnValueHeads + value_head];
        const float step_beta =
            beta[step * kGdnValueHeads + value_head];
        float key_value = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] *= step_decay;
            key_value += state[i] * float(qk_row[key_base + element]);
        }
        key_value = simd_sum(key_value);
        const float delta =
            (float(value[step * kGdnValueValues +
                         value_head * kGdnHeadDimension + value_dimension]) -
             key_value) *
            step_beta;
        float result = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] += float(qk_row[key_base + element]) * delta;
            result += state[i] * float(qk_row[query_base + element]);
        }
        result = simd_sum(result);
        if (lane == 0u) {
            output[step * kGdnValueValues +
                   value_head * kGdnHeadDimension + value_dimension] =
                static_cast<bfloat>(result);
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        state_out[state_base + lane * 4u + i] = state[i];
    }
}

// Each simdgroup retains one recurrent state row in registers while walking
// positions in ascending order and writes persistent state once after the
// final position. Its source expressions mirror gdn_recurrence. A direct N=96
// discriminator produced the same complete state bytes as repeated one-step
// dispatches, but this path remains default-off until the upstream GDN
// preparation mismatch is corrected and complete equality closes.
kernel void gdn_recurrence_blk(
    device const bfloat* qk [[buffer(0)]], device const bfloat* value [[buffer(1)]],
    device const bfloat* projection [[buffer(2)]],
    device const bfloat* a_log [[buffer(3)]],
    device const bfloat* dt_bias [[buffer(4)]],
    device const float* state_in [[buffer(5)]], device bfloat* output [[buffer(6)]],
    device float* state_out [[buffer(7)]], constant uint& steps [[buffer(8)]],
    uint3 position [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint value_dimension = position.y;
    const uint value_head = position.z;
    const uint key_head = value_head >> 1u;
    const uint query_base = key_head * kGdnHeadDimension;
    const uint key_base = kGdnKOffset + query_base;
    const ulong state_base =
        (ulong(value_head) * kGdnHeadDimension + value_dimension) *
        kGdnHeadDimension;
    float state[4];
    for (uint i = 0u; i < 4u; ++i) {
        state[i] = state_in[state_base + lane * 4u + i];
    }
    for (uint step = 0u; step < steps; ++step) {
        device const bfloat* projection_row =
            projection + step * kGdnProjectionRows;
        device const bfloat* qk_row = qk + step * kGdnQkValues;
        const bfloat a_b = projection_row[kGdnARowOffset + value_head];
        const bfloat b_b = projection_row[kGdnBRowOffset + value_head];
        const bfloat shifted = a_b + dt_bias[value_head];
        const bfloat branch_max =
            float(shifted) > 0.0f ? shifted : bfloat(0.0f);
        const bfloat branch_min =
            float(shifted) > 0.0f ? bfloat(0.0f) : shifted;
        const bfloat difference = branch_min - branch_max;
        const bfloat exp_b = static_cast<bfloat>(exp(float(difference)));
        const float one_plus = 1.0f + float(exp_b);
        const bfloat log1p_b =
            one_plus == 1.0f
                ? exp_b
                : static_cast<bfloat>(
                      float(exp_b) * (log(one_plus) / (one_plus - 1.0f)));
        const bfloat softplus_b = branch_max + log1p_b;
        const float decay =
            exp(-exp(float(a_log[value_head])) * float(softplus_b));
        const bfloat beta_exp =
            static_cast<bfloat>(exp(fabs(float(b_b))));
        const bfloat beta_low = bfloat(1.0f) / (bfloat(1.0f) + beta_exp);
        const float beta =
            float(b_b) < 0.0f
                ? float(beta_low)
                : float(static_cast<bfloat>(bfloat(1.0f) - beta_low));
        float key_value = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] *= decay;
            key_value += state[i] * float(qk_row[key_base + element]);
        }
        key_value = simd_sum(key_value);
        const float delta =
            (float(value[step * kGdnValueValues +
                         value_head * kGdnHeadDimension + value_dimension]) -
             key_value) *
            beta;
        float result = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] += float(qk_row[key_base + element]) * delta;
            result += state[i] * float(qk_row[query_base + element]);
        }
        result = simd_sum(result);
        if (lane == 0u) {
            output[step * kGdnValueValues +
                   value_head * kGdnHeadDimension + value_dimension] =
                static_cast<bfloat>(result);
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        state_out[state_base + lane * 4u + i] = state[i];
    }
}

kernel void gdn_gate_norm_blk(
    device const bfloat* recurrence_out [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup float head_values[kGdnHeadDimension];
    threadgroup float inverse_shared[1];
    const uint lane = thread_position.x;
    const uint index = group.y * kGdnValueValues +
                       group.x * kGdnHeadDimension + lane;
    head_values[lane] = float(recurrence_out[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < kSimdgroupWidth) {
        const uint base = lane * 4u;
        float total = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const float value = head_values[base + i];
            total += value * value;
        }
        total = simd_sum(total);
        if (lane == 0u) {
            inverse_shared[0] =
                precise::rsqrt(total / float(kGdnHeadDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bfloat normalized =
        weight[lane] *
        static_cast<bfloat>(head_values[lane] * inverse_shared[0]);
    const float gate_value = float(gate[index]);
    const float sigmoid_low = 1.0f / (1.0f + exp(abs(gate_value)));
    const float sigmoid =
        gate_value < 0.0f ? sigmoid_low : 1.0f - sigmoid_low;
    output[index] =
        static_cast<bfloat>((gate_value * sigmoid) * float(normalized));
}

kernel void gdn_conv_tape_blk(device const bfloat* projection [[buffer(0)]],
                         device const bfloat* conv_state [[buffer(1)]],
                         device const bfloat* conv_weights [[buffer(2)]],
                         device bfloat* qk [[buffer(3)]],
                         device bfloat* value_out [[buffer(4)]],
                         device bfloat* gate_out [[buffer(5)]],
                         device bfloat* new_conv_state [[buffer(6)]],
                         constant uint& state_step [[buffer(7)]],
                         device bfloat* conv_tape [[buffer(8)]],
                         constant uint& tape_layer [[buffer(9)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint2 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup float activation[kGdnHeadDimension];
    threadgroup float inverse_shared[1];
    const uint lane = thread_position.x;
    const uint position = group.y;
    device const bfloat* projection_row =
        projection + position * kGdnProjectionRows;
    const uint conv_groups = kGdnConvChannels / kGdnHeadDimension;
    if (group.x < conv_groups) {
        const uint channel = group.x * kGdnHeadDimension + lane;
        float taps[kGdnConvWidth];
        for (uint tap = 0u; tap < kGdnConvWidth; ++tap) {
            const uint logical = position + tap;
            taps[tap] =
                logical < kGdnConvWidth - 1u
                    ? float(conv_state[logical * kGdnConvChannels + channel])
                    : float(projection[(logical - (kGdnConvWidth - 1u)) *
                                           kGdnProjectionRows +
                                       channel]);
        }
        const ulong weight_base = ulong(channel) * kGdnConvWidth;
        // Preserve gdn_prepare's exact four-term evaluation shape. A loop
        // beginning at zero is mathematically equivalent but is not the same
        // floating-point source contract.
        const float convolved =
            float(conv_weights[weight_base]) * taps[0] +
            float(conv_weights[weight_base + 1u]) * taps[1] +
            float(conv_weights[weight_base + 2u]) * taps[2] +
            float(conv_weights[weight_base + 3u]) * taps[3];
        const bfloat convolved_b = static_cast<bfloat>(convolved);
        const bfloat exp_b = static_cast<bfloat>(exp(fabs(float(convolved_b))));
        const bfloat sigmoid_low = bfloat(1.0f) / (bfloat(1.0f) + exp_b);
        const bfloat sigmoid_b =
            float(convolved_b) < 0.0f
                ? sigmoid_low
                : static_cast<bfloat>(bfloat(1.0f) - sigmoid_low);
        const bfloat activated_b = convolved_b * sigmoid_b;
        if (position == state_step - 1u) {
            for (uint tap = 0u; tap < kGdnConvWidth - 1u; ++tap) {
                new_conv_state[tap * kGdnConvChannels + channel] =
                    static_cast<bfloat>(taps[tap + 1u]);
            }
        }
        // A40 tape: the same expression the state write uses, recorded for
        // EVERY position so rollback can restore the tail at any row.
        if (state_step <= kGdnTapeRows) {
            for (uint tap = 0u; tap < kGdnConvWidth - 1u; ++tap) {
                conv_tape[((ulong(tape_layer) * kGdnTapeRows + position) *
                               (kGdnConvWidth - 1u) +
                           tap) *
                              kGdnConvChannels +
                          channel] = static_cast<bfloat>(taps[tap + 1u]);
            }
        }
        if (group.x < kGdnQkValues / kGdnHeadDimension) {
            activation[lane] = float(activated_b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < kSimdgroupWidth) {
                const uint base = lane * 4u;
                float total = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const float value = activation[base + i];
                    total += value * value;
                }
                total = simd_sum(total);
                if (lane == 0u) {
                    inverse_shared[0] = precise::rsqrt(
                        total / float(kGdnHeadDimension) + kRmsNormEpsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bfloat normalized =
                static_cast<bfloat>(activation[lane] * inverse_shared[0]);
            const bfloat scale =
                group.x < kGdnKeyHeads ? bfloat(kGdnQueryScale) : bfloat(kGdnKeyScale);
            qk[position * kGdnQkValues + channel] = normalized * scale;
        } else {
            value_out[position * kGdnValueValues + channel - kGdnQkValues] =
                activated_b;
        }
    } else {
        const uint index =
            (group.x - conv_groups) * kGdnHeadDimension + lane;
        gate_out[position * kGdnValueValues + index] =
            projection_row[kGdnQkvRows + index];
    }
}

// Gate/β hoist for the register-loop diagnostic. The scalar source chain is
// identical to that loop but is evaluated once per (position, head) rather
// than once per value dimension. C1's conservative gate uses the permanent
// one-step recurrence pipeline and does not select this optimization.

kernel void gdn_recurrence_gates_tape_blk(
    device const bfloat* qk [[buffer(0)]],
    device const bfloat* value [[buffer(1)]],
    device const float* decay [[buffer(2)]],
    device const bfloat* a_log [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device const float* state_in [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant uint& steps [[buffer(8)]],
    device float* tape [[buffer(9)]],
    constant uint& tape_layer [[buffer(10)]],
    uint3 position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    (void)a_log;
    const uint value_dimension = position.y;
    const uint value_head = position.z;
    const uint key_head = value_head >> 1u;
    const uint query_base = key_head * kGdnHeadDimension;
    const uint key_base = kGdnKOffset + query_base;
    const ulong state_base =
        (ulong(value_head) * kGdnHeadDimension + value_dimension) *
        kGdnHeadDimension;
    float state[4];
    for (uint i = 0u; i < 4u; ++i) {
        state[i] = state_in[state_base + lane * 4u + i];
    }
    for (uint step = 0u; step < steps; ++step) {
        device const bfloat* qk_row = qk + step * kGdnQkValues;
        const float step_decay =
            decay[step * kGdnValueHeads + value_head];
        const float step_beta =
            beta[step * kGdnValueHeads + value_head];
        float key_value = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] *= step_decay;
            key_value += state[i] * float(qk_row[key_base + element]);
        }
        key_value = simd_sum(key_value);
        const float delta =
            (float(value[step * kGdnValueValues +
                         value_head * kGdnHeadDimension + value_dimension]) -
             key_value) *
            step_beta;
        // A40 tape: delta is the only in-flight quantity; k and the gates
        // are copied because the per-row scratch is overwritten per layer.
        if (steps <= kGdnTapeRows) {
            const ulong tape_row = ulong(tape_layer) * kGdnTapeRows + step;
            if (lane == 0u) {
                tape[(tape_row * kGdnValueHeads + value_head) *
                         kGdnHeadDimension +
                     value_dimension] = delta;
            }
            if (value_dimension == 0u) {
                if ((value_head & 1u) == 0u) {
                    for (uint i = 0u; i < 4u; ++i) {
                        const uint element = lane * 4u + i;
                        tape[kGdnTapeKeyOffset +
                             (tape_row * kGdnKeyHeads + key_head) *
                                 kGdnHeadDimension +
                             element] = float(qk_row[key_base + element]);
                    }
                }
                if (lane == 0u) {
                    tape[kGdnTapeGateOffset +
                         (tape_row * kGdnValueHeads + value_head) * 2u] =
                        step_decay;
                    tape[kGdnTapeGateOffset +
                         (tape_row * kGdnValueHeads + value_head) * 2u + 1u] =
                        step_beta;
                }
            }
        }
        float result = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] += float(qk_row[key_base + element]) * delta;
            result += state[i] * float(qk_row[query_base + element]);
        }
        result = simd_sum(result);
        if (lane == 0u) {
            output[step * kGdnValueValues +
                   value_head * kGdnHeadDimension + value_dimension] =
                static_cast<bfloat>(result);
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        state_out[state_base + lane * 4u + i] = state[i];
    }
}

// Each simdgroup retains one recurrent state row in registers while walking
// positions in ascending order and writes persistent state once after the
// final position. Its source expressions mirror gdn_recurrence. A direct N=96
// discriminator produced the same complete state bytes as repeated one-step
// dispatches, but this path remains default-off until the upstream GDN
// preparation mismatch is corrected and complete equality closes.

// A40 replay: rebuilds the recurrent state from the pre-band snapshot to
// `steps` committed rows using the tape. The two state-update statements
// are the forward scan's own expressions (generation pin); kv/output are
// not recomputed because they never feed the state trajectory.
kernel void gdn_tape_replay_blk(
    device const float* tape [[buffer(0)]],
    device const float* state_in [[buffer(1)]],
    device float* state_out [[buffer(2)]],
    constant uint& steps [[buffer(3)]],
    constant uint& tape_layer [[buffer(4)]],
    uint3 position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint value_dimension = position.y;
    const uint value_head = position.z;
    const uint key_head = value_head >> 1u;
    const ulong state_base =
        (ulong(value_head) * kGdnHeadDimension + value_dimension) *
        kGdnHeadDimension;
    float state[4];
    for (uint i = 0u; i < 4u; ++i) {
        state[i] = state_in[state_base + lane * 4u + i];
    }
    for (uint step = 0u; step < steps; ++step) {
        const ulong tape_row = ulong(tape_layer) * kGdnTapeRows + step;
        const float step_decay =
            tape[kGdnTapeGateOffset +
                 (tape_row * kGdnValueHeads + value_head) * 2u];
        const float delta =
            tape[(tape_row * kGdnValueHeads + value_head) *
                     kGdnHeadDimension +
                 value_dimension];
        for (uint i = 0u; i < 4u; ++i) {
            const uint element = lane * 4u + i;
            state[i] *= step_decay;
            state[i] += tape[kGdnTapeKeyOffset +
                             (tape_row * kGdnKeyHeads + key_head) *
                                 kGdnHeadDimension +
                             element] *
                        delta;
        }
    }
    for (uint i = 0u; i < 4u; ++i) {
        state_out[state_base + lane * 4u + i] = state[i];
    }
}
