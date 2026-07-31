kernel void gdn_project(
    device const bfloat* input [[buffer(0)]], device const uint* qkv_words [[buffer(1)]],
    device const bfloat* qkv_scales [[buffer(2)]], device const bfloat* qkv_biases [[buffer(3)]],
    device const uint* z_words [[buffer(4)]], device const bfloat* z_scales [[buffer(5)]],
    device const bfloat* z_biases [[buffer(6)]], device const uint* b_words [[buffer(7)]],
    device const bfloat* b_scales [[buffer(8)]], device const bfloat* b_biases [[buffer(9)]],
    device const uint* a_words [[buffer(10)]], device const bfloat* a_scales [[buffer(11)]],
    device const bfloat* a_biases [[buffer(12)]], device bfloat* projection [[buffer(13)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kGdnProjectionRows) {
        return;
    }
    device const uint* words;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local_row;
    if (row < kGdnQkvRows) {
        words = qkv_words;
        scales = qkv_scales;
        biases = qkv_biases;
        local_row = row;
    } else if (row < kGdnQkvRows + kGdnZRows) {
        words = z_words;
        scales = z_scales;
        biases = z_biases;
        local_row = row - kGdnQkvRows;
    } else if (row < kGdnBRowOffset + kGdnValueHeads) {
        words = b_words;
        scales = b_scales;
        biases = b_biases;
        local_row = row - kGdnBRowOffset;
    } else {
        words = a_words;
        scales = a_scales;
        biases = a_biases;
        local_row = row - kGdnARowOffset;
    }
    const ulong word_row = ulong(local_row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(local_row) * ulong(kGroupsPerRow);
    const float acc = q4_dot(input, words + word_row, scales + group_row, biases + group_row,
                             kHiddenDimension, lane);
    if (lane == 0u) {
        projection[row] = static_cast<bfloat>(acc);
    }
}

kernel void gdn_prepare(device const bfloat* projection [[buffer(0)]],
                        device const bfloat* conv_state [[buffer(1)]],
                        device const bfloat* conv_weights [[buffer(2)]],
                        device bfloat* qk [[buffer(3)]], device bfloat* value_out [[buffer(4)]],
                        device bfloat* gate_out [[buffer(5)]],
                        device bfloat* new_conv_state [[buffer(6)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    // Exact reference op-boundary semantics: the width-4 convolution output,
    // the stable-branch sigmoid, and the silu product each round to bfloat16
    // as three separate eager operations. The q/k RMS uses the admitted
    // 32-lane by 4-contiguous single-simdgroup reduction with precise rsqrt,
    // and the scales are exact bfloat16 constants (the k scale is the
    // bfloat16 cast of 1/sqrt(128)).
    threadgroup float activation[128];
    threadgroup float inverse_shared[1];
    if (tg < kGdnConvChannels / 128u) {
        const uint channel = tg * 128u + lane;
        const float x0 = float(conv_state[channel]);
        const float x1 = float(conv_state[kGdnConvChannels + channel]);
        const float x2 = float(conv_state[2u * kGdnConvChannels + channel]);
        const float x3 = float(projection[channel]);
        const ulong weight_base = ulong(channel) * 4ul;
        const float convolved = float(conv_weights[weight_base]) * x0 +
                                float(conv_weights[weight_base + 1ul]) * x1 +
                                float(conv_weights[weight_base + 2ul]) * x2 +
                                float(conv_weights[weight_base + 3ul]) * x3;
        const bfloat convolved_b = static_cast<bfloat>(convolved);
        const bfloat exp_b = static_cast<bfloat>(exp(fabs(float(convolved_b))));
        const bfloat sig_low = bfloat(1.0f) / (bfloat(1.0f) + exp_b);
        const bfloat sigmoid_b =
            float(convolved_b) < 0.0f ? sig_low : static_cast<bfloat>(bfloat(1.0f) - sig_low);
        const bfloat activated_b = convolved_b * sigmoid_b;
        new_conv_state[channel] = static_cast<bfloat>(x1);
        new_conv_state[kGdnConvChannels + channel] = static_cast<bfloat>(x2);
        new_conv_state[2u * kGdnConvChannels + channel] = static_cast<bfloat>(x3);
        if (tg < kGdnQkValues / 128u) {
            activation[lane] = float(activated_b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 32u) {
                const uint base = lane * 4u;
                float acc = 0.0f;
                for (uint i = 0; i < 4u; ++i) {
                    const float value = activation[base + i];
                    acc += value * value;
                }
                acc = simd_sum(acc);
                if (lane == 0u) {
                    inverse_shared[0] =
                        precise::rsqrt(acc / float(kGdnHeadDimension) + kRmsNormEpsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bfloat normed = static_cast<bfloat>(activation[lane] * inverse_shared[0]);
            const bfloat scale_b =
                tg < kGdnKeyHeads ? bfloat(kGdnQueryScale) : bfloat(kGdnKeyScale);
            qk[channel] = normed * scale_b;
        } else {
            value_out[channel - kGdnQkValues] = activated_b;
        }
    } else {
        const uint index = (tg - kGdnConvChannels / 128u) * 128u + lane;
        gate_out[index] = projection[kGdnQkvRows + index];
    }
}

kernel void
gdn_recurrence(device const bfloat* qk [[buffer(0)]], device const bfloat* value [[buffer(1)]],
               device const bfloat* projection [[buffer(2)]],
               device const bfloat* a_log [[buffer(3)]], device const bfloat* dt_bias [[buffer(4)]],
               device const float* state_in [[buffer(5)]], device bfloat* output [[buffer(6)]],
               device float* state_out [[buffer(7)]], uint3 position [[thread_position_in_grid]],
               uint lane [[thread_index_in_simdgroup]]) {
    const uint value_dim = position.y;
    const uint value_head = position.z;
    const uint key_head = value_head >> 1u;
    const uint query_base = key_head * kGdnHeadDimension;
    const uint key_base = kGdnKOffset + query_base;
    const ulong state_base = (ulong(value_head) * ulong(kGdnHeadDimension) + ulong(value_dim)) *
                             ulong(kGdnHeadDimension);
    // Exact reference derivation. beta is the stable-branch sigmoid at
    // bfloat16 with per-operation rounding. The decay exponent is
    // softplus(a + dt_bias) as bfloat16 LogAddExp: the exponential rounds to
    // bfloat16, then the Goldberg log1p form takes one final bfloat16
    // rounding; the two outer exponentials run in float32.
    const bfloat a_b = projection[kGdnARowOffset + value_head];
    const bfloat b_b = projection[kGdnBRowOffset + value_head];
    const bfloat shifted = a_b + dt_bias[value_head];
    const bfloat branch_max = float(shifted) > 0.0f ? shifted : bfloat(0.0f);
    const bfloat branch_min = float(shifted) > 0.0f ? bfloat(0.0f) : shifted;
    const bfloat difference = branch_min - branch_max;
    const bfloat exp_b = static_cast<bfloat>(exp(float(difference)));
    const float one_plus = 1.0f + float(exp_b);
    const bfloat log1p_b =
        (one_plus == 1.0f)
            ? exp_b
            : static_cast<bfloat>(float(exp_b) * (log(one_plus) / (one_plus - 1.0f)));
    const bfloat softplus_b = branch_max + log1p_b;
    const float decay = exp(-exp(float(a_log[value_head])) * float(softplus_b));
    const bfloat beta_exp = static_cast<bfloat>(exp(fabs(float(b_b))));
    const bfloat beta_low = bfloat(1.0f) / (bfloat(1.0f) + beta_exp);
    const float beta =
        float(b_b) < 0.0f ? float(beta_low) : float(static_cast<bfloat>(bfloat(1.0f) - beta_low));
    float state[4];
    float key_value = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const uint element = lane * 4u + i;
        state[i] = state_in[state_base + element] * decay;
        key_value += state[i] * float(qk[key_base + element]);
    }
    key_value = simd_sum(key_value);
    const float delta =
        (float(value[value_head * kGdnHeadDimension + value_dim]) - key_value) * beta;
    float out = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const uint element = lane * 4u + i;
        state[i] += float(qk[key_base + element]) * delta;
        state_out[state_base + element] = state[i];
        out += state[i] * float(qk[query_base + element]);
    }
    out = simd_sum(out);
    if (lane == 0u) {
        output[value_head * kGdnHeadDimension + value_dim] = static_cast<bfloat>(out);
    }
}

kernel void gdn_gate_norm(device const bfloat* recurrence_out [[buffer(0)]],
                          device const bfloat* gate [[buffer(1)]],
                          device const bfloat* weight [[buffer(2)]],
                          device bfloat* output [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]]) {
    // Exact reference semantics: weighted head RMS with the double-rounded
    // output contract, then swiglu — the gate's stable-branch sigmoid and the
    // final product run in float32 with one bfloat16 rounding at the end.
    threadgroup float head_values[128];
    threadgroup float inverse_shared[1];
    const uint index = tg * 128u + lane;
    head_values[lane] = float(recurrence_out[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32u) {
        const uint base = lane * 4u;
        float acc = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const float value = head_values[base + i];
            acc += value * value;
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            inverse_shared[0] = precise::rsqrt(acc / float(kGdnHeadDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bfloat normed_b =
        weight[lane] * static_cast<bfloat>(head_values[lane] * inverse_shared[0]);
    const float gate_value = float(gate[index]);
    const float sig_low = 1.0f / (1.0f + exp(fabs(gate_value)));
    const float sigmoid_value = gate_value < 0.0f ? sig_low : 1.0f - sig_low;
    output[index] = static_cast<bfloat>((gate_value * sigmoid_value) * float(normed_b));
}

kernel void gdn_outproj(device const bfloat* input [[buffer(0)]],
                        device const uint* words [[buffer(1)]],
                        device const bfloat* scales [[buffer(2)]],
                        device const bfloat* biases [[buffer(3)]],
                        device bfloat* output [[buffer(4)]], uint tid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kHiddenDimension) {
        return;
    }
    const ulong word_row = ulong(row) * ulong(kGdnValueValues / 8u);
    const ulong group_row = ulong(row) * ulong(kGdnValueValues / kQ4GroupSize);
    const float acc = q4_dot(input, words + word_row, scales + group_row, biases + group_row,
                             kGdnValueValues, lane);
    if (lane == 0u) {
        output[row] = static_cast<bfloat>(acc);
    }
}
