// Two-stage simdgroup reduction derived from MLX (Apple Inc., MIT)
// rms_norm.metal, rms_single_row. The bfloat specialization and the
// double-rounding output contract below are Tatara's own.
// See THIRD_PARTY_NOTICES.md and docs/design/PROVENANCE.md.
kernel void rms_only(device const bfloat* input [[buffer(0)]],
                     device const bfloat* weight [[buffer(1)]], device bfloat* output [[buffer(2)]],
                     uint thread_in_group [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float inverse_norm[1];
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
            inverse_norm[0] = precise::rsqrt(sum / float(kHiddenDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = inverse_norm[0];
    for (uint i = 0u; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        // The admitted contract rounds input * inverse to bfloat16 BEFORE the
        // bfloat16 weight multiply; both roundings are load-bearing.
        output[element] = weight[element] * static_cast<bfloat>(float(input[element]) * inverse);
    }
}
