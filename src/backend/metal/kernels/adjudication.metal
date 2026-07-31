// Probe-only oracle-adjudication kernels; never selected by a production plan.

kernel void adjudicate_rsqrt(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    output[i] = precise::rsqrt(input[i]);
}

// Observes one bare 32-lane simd_sum so the intrinsic's association tree is
// adjudicated directly, without any surrounding reduction structure.
kernel void adjudicate_simd_sum(device const float* input [[buffer(0)]],
                                device float* total [[buffer(1)]],
                                uint lane [[thread_position_in_threadgroup]]) {
    const float sum = simd_sum(input[lane]);
    if (lane == 0u) {
        total[0] = sum;
    }
}

// Replicates rms_only's reduction exactly and exposes the raw float total, so
// simd_sum association is adjudicated at full float resolution.
kernel void adjudicate_rms_sum(device const bfloat* input [[buffer(0)]],
                               device float* total [[buffer(1)]],
                               uint thread_in_group [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[32];

    const uint base = thread_in_group * kRmsValuesPerThread;
    float sum = 0.0f;
    for (uint i = 0u; i < kRmsValuesPerThread; ++i) {
        const float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (simdgroup == 0u) {
        stage[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        stage[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        sum = simd_sum(stage[lane]);
        if (lane == 0u) {
            total[0] = sum;
        }
    }
}

kernel void adjudicate_bfloat_multiply(device const bfloat* left [[buffer(0)]],
                                       device const bfloat* right [[buffer(1)]],
                                       device bfloat* output [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
    output[i] = left[i] * right[i];
}

// GDN-boundary observation kernels. Composite scalar semantics are queried
// directly at needed arguments; chain association is observed through the
// real kernels' float outputs.

kernel void adjudicate_bf16_sigmoid(device const bfloat* input [[buffer(0)]],
                                    device bfloat* output [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    const bfloat x = input[i];
    const bfloat exp_b = static_cast<bfloat>(exp(fabs(float(x))));
    const bfloat low = bfloat(1.0f) / (bfloat(1.0f) + exp_b);
    output[i] = float(x) < 0.0f ? low : static_cast<bfloat>(bfloat(1.0f) - low);
}

kernel void adjudicate_f32_sigmoid(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
    const float x = input[i];
    const float low = 1.0f / (1.0f + exp(fabs(x)));
    output[i] = x < 0.0f ? low : 1.0f - low;
}

kernel void adjudicate_gdn_decay(device const bfloat* shifted [[buffer(0)]],
                                 device const bfloat* a_log [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    const bfloat s = shifted[i];
    const bfloat branch_max = float(s) > 0.0f ? s : bfloat(0.0f);
    const bfloat branch_min = float(s) > 0.0f ? bfloat(0.0f) : s;
    const bfloat difference = branch_min - branch_max;
    const bfloat exp_b = static_cast<bfloat>(exp(float(difference)));
    const float one_plus = 1.0f + float(exp_b);
    const bfloat log1p_b =
        (one_plus == 1.0f)
            ? exp_b
            : static_cast<bfloat>(float(exp_b) * (log(one_plus) / (one_plus - 1.0f)));
    const bfloat softplus_b = branch_max + log1p_b;
    output[i] = exp(-exp(float(a_log[i])) * float(softplus_b));
}

kernel void adjudicate_bfloat_add(device const bfloat* left [[buffer(0)]],
                                  device const bfloat* right [[buffer(1)]],
                                  device bfloat* output [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    output[i] = left[i] + right[i];
}

kernel void adjudicate_conv4(device const bfloat* weights [[buffer(0)]],
                             device const float* taps [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    const ulong base = ulong(i) * 4ul;
    output[i] = float(weights[base]) * taps[base] + float(weights[base + 1ul]) * taps[base + 1ul] +
                float(weights[base + 2ul]) * taps[base + 2ul] +
                float(weights[base + 3ul]) * taps[base + 3ul];
}

kernel void
adjudicate_q4_row(device const bfloat* x [[buffer(0)]], device const uint* words [[buffer(1)]],
                  device const bfloat* scales [[buffer(2)]],
                  device const bfloat* biases [[buffer(3)]], device float* output [[buffer(4)]],
                  uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    const ulong word_row = ulong(row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(row) * ulong(kGroupsPerRow);
    const float acc =
        q4_dot(x, words + word_row, scales + group_row, biases + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        output[row] = acc;
    }
}

kernel void
adjudicate_q4_row_v(device const bfloat* x [[buffer(0)]], device const uint* words [[buffer(1)]],
                    device const bfloat* scales [[buffer(2)]],
                    device const bfloat* biases [[buffer(3)]], device float* output [[buffer(4)]],
                    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    const ulong word_row = ulong(row) * ulong(kGdnValueValues / 8u);
    const ulong group_row = ulong(row) * ulong(kGdnValueValues / kQ4GroupSize);
    const float acc =
        q4_dot(x, words + word_row, scales + group_row, biases + group_row, kGdnValueValues, lane);
    if (lane == 0u) {
        output[row] = acc;
    }
}

// Observes the RoPE angle composite exactly as attn_qk_rope computes it.
kernel void adjudicate_rope_trig(device const uint* positions [[buffer(0)]],
                                 device const uint* pairs [[buffer(1)]],
                                 device float* cosines [[buffer(2)]],
                                 device float* sines [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    const float theta =
        float(positions[i]) / pow(kAttnRopeBase, float(pairs[i]) / float(kAttnRopePairs));
    cosines[i] = cos(theta);
    sines[i] = sin(theta);
}

// Observes exp at float resolution for the softmax paths.
kernel void adjudicate_f32_exp(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    output[i] = exp(input[i]);
}

// Observes the sealed 256-slot threadgroup max and sum trees over raw float
// inputs, one vector per threadgroup, exactly as the decode kernels reduce.
kernel void adjudicate_tg_trees(device const float* input [[buffer(0)]],
                                device float* max_out [[buffer(1)]],
                                device float* sum_out [[buffer(2)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    device const float* src = input + tg * 256u;
    red[tid] = src[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 128u; off; off >>= 1u) {
        if (tid < off) {
            red[tid] = max(red[tid], red[tid + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        max_out[tg] = red[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[tid] = src[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 128u; off; off >>= 1u) {
        if (tid < off) {
            red[tid] += red[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        sum_out[tg] = red[0];
    }
}

// MoE-boundary observation kernels: the packed-Q4 dot at its two widths and
// the Q8 dot at the router width expose raw float row results, and the down
// accumulation mirror exposes the raw slot-order total before the bfloat16
// rounding hides the chain.

kernel void
adjudicate_q4p_row(device const bfloat* x [[buffer(0)]], device const uint* words [[buffer(1)]],
                   device const bfloat* scales [[buffer(2)]],
                   device const bfloat* biases [[buffer(3)]], device float* output [[buffer(4)]],
                   uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    const ulong word_row = ulong(row) * ulong(kQuantWordsPerRow);
    const ulong group_row = ulong(row) * ulong(kGroupsPerRow);
    const float acc = q4_dot_packed(x, words + word_row, scales + group_row, biases + group_row,
                                    kHiddenDimension, lane);
    if (lane == 0u) {
        output[row] = acc;
    }
}

kernel void adjudicate_q4p_row_e(device const bfloat* x [[buffer(0)]],
                                 device const uint* words [[buffer(1)]],
                                 device const bfloat* scales [[buffer(2)]],
                                 device const bfloat* biases [[buffer(3)]],
                                 device float* output [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    const ulong word_row = ulong(row) * ulong(kMoeExpertDimension / 8u);
    const ulong group_row = ulong(row) * ulong(kMoeExpertDimension / kQ4GroupSize);
    const float acc = q4_dot_packed(x, words + word_row, scales + group_row, biases + group_row,
                                    kMoeExpertDimension, lane);
    if (lane == 0u) {
        output[row] = acc;
    }
}

kernel void
adjudicate_q8_row(device const bfloat* x [[buffer(0)]], device const uint* words [[buffer(1)]],
                  device const bfloat* scales [[buffer(2)]],
                  device const bfloat* biases [[buffer(3)]], device float* output [[buffer(4)]],
                  uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    const ulong word_row = ulong(row) * ulong(2u * kQuantWordsPerRow);
    const ulong group_row = ulong(row) * ulong(kGroupsPerRow);
    const float acc =
        q8_dot(x, words + word_row, scales + group_row, biases + group_row, kHiddenDimension, lane);
    if (lane == 0u) {
        output[row] = acc;
    }
}

// Mirrors grouped_down_res's ascending slot accumulation exactly and writes
// the raw float total, giving the multiply-add chain float-resolution
// discrimination.
kernel void adjudicate_down_total(
    device const bfloat* hidden_in [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device const float* shared_coefficient [[buffer(3)]],
    device const uint* down_words [[buffer(4)]], device const bfloat* down_scales [[buffer(5)]],
    device const bfloat* down_biases [[buffer(6)]],
    device const uint* shared_down_words [[buffer(7)]],
    device const bfloat* shared_down_scales [[buffer(8)]],
    device const bfloat* shared_down_biases [[buffer(9)]], device float* total_out [[buffer(10)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid >> 5u;
    if (row >= kHiddenDimension) {
        return;
    }
    float total = 0.0f;
    for (uint slot = 0; slot < kMoeActiveExperts + 1u; ++slot) {
        const bool shared = slot == kMoeActiveExperts;
        const uint expert = shared ? 0u : ids[slot];
        if (!shared && expert == 0xffffffffu) {
            continue;
        }
        const ulong word_row = shared ? ulong(row) * ulong(kMoeExpertDimension / 8u)
                                      : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                            ulong(kMoeExpertDimension / 8u);
        const ulong group_row = shared ? ulong(row) * ulong(kMoeExpertDimension / kQ4GroupSize)
                                       : (ulong(expert) * ulong(kHiddenDimension) + ulong(row)) *
                                             ulong(kMoeExpertDimension / kQ4GroupSize);
        const float part = q4_dot_packed(hidden_in + slot * kMoeExpertDimension,
                                         (shared ? shared_down_words : down_words) + word_row,
                                         (shared ? shared_down_scales : down_scales) + group_row,
                                         (shared ? shared_down_biases : down_biases) + group_row,
                                         kMoeExpertDimension, lane);
        if (lane == 0u) {
            total += (shared ? shared_coefficient[0] : coefficients[slot]) * part;
        }
    }
    if (lane == 0u) {
        total_out[row] = total;
    }
}

// Observes f32 division at exact argument pairs for the division-bearing
// composites: softmax normalization, renormalization, sigmoids, and silu.
kernel void adjudicate_f32_divide(device const float* numerators [[buffer(0)]],
                                  device const float* denominators [[buffer(1)]],
                                  device float* quotients [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    quotients[i] = numerators[i] / denominators[i];
}
