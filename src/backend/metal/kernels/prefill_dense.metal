// Exact multi-row affine-Q4 dot. Weights are retained while up to four
// independent input rows accumulate, but every row keeps q4_dot's 512-value
// chunk order, scalar association, and one final simd_sum.
inline void q4_dot_multi4(device const bfloat* input, uint input_stride, uint first_position,
                          uint position_count, device const uint* words,
                          device const bfloat* scales, device const bfloat* biases, uint width,
                          uint lane, thread float* accumulators) {
    device const ushort* packed = reinterpret_cast<device const ushort*>(words);
    for (uint k = 0u; k < width; k += kQ4DotChunk) {
        const uint word_base = (k >> 2u) + lane * 4u;
        ushort word_values[4];
        for (uint i = 0u; i < 4u; ++i) {
            word_values[i] = packed[word_base + i];
        }
        const uint group = (k >> 6u) + (lane >> 2u);
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        const uint input_base = k + lane * kQ4LaneValues;
        for (uint position = 0u; position < position_count; ++position) {
            device const bfloat* row =
                input + (first_position + position) * input_stride;
            float values[kQ4LaneValues];
            float sum = 0.0f;
            for (uint i = 0u; i < kQ4LaneValues; i += 4u) {
                const float x0 = float(row[input_base + i]);
                const float x1 = float(row[input_base + i + 1u]);
                const float x2 = float(row[input_base + i + 2u]);
                const float x3 = float(row[input_base + i + 3u]);
                sum += x0 + x1 + x2 + x3;
                values[i] = x0;
                values[i + 1u] = x1 / 16.0f;
                values[i + 2u] = x2 / 256.0f;
                values[i + 3u] = x3 / 4096.0f;
            }
            float quantized = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const ushort word = word_values[i];
                quantized += values[i * 4u] * float(word & 0x000fu) +
                             values[i * 4u + 1u] * float(word & 0x00f0u) +
                             values[i * 4u + 2u] * float(word & 0x0f00u) +
                             values[i * 4u + 3u] * float(word & 0xf000u);
            }
            accumulators[position] += scale * quantized + sum * bias;
        }
    }
    for (uint position = 0u; position < position_count; ++position) {
        accumulators[position] = simd_sum(accumulators[position]);
    }
}

kernel void embed_rows_q4(device const uint* quant_words [[buffer(0)]],
                          device const bfloat* scales [[buffer(1)]],
                          device const bfloat* biases [[buffer(2)]],
                          device const uint* tokens [[buffer(3)]],
                          device bfloat* output [[buffer(4)]],
                          uint2 element [[thread_position_in_grid]]) {
    const uint selected = tokens[element.y];
    const uint group = element.x / kQ4GroupSize;
    const float scale = float(scales[selected * kGroupsPerRow + group]);
    const float bias = float(biases[selected * kGroupsPerRow + group]);
    const uint word =
        quant_words[selected * kQuantWordsPerRow +
                    element.x / kQ4ValuesPerWord];
    const uint quantized =
        (word >> (4u * (element.x % kQ4ValuesPerWord))) & 15u;
    output[element.y * kHiddenDimension + element.x] =
        static_cast<bfloat>(float(quantized) * scale + bias);
}

kernel void rms_blk(device const bfloat* input [[buffer(0)]],
                    device const bfloat* weight [[buffer(1)]],
                    device bfloat* output [[buffer(2)]],
                    uint2 group [[threadgroup_position_in_grid]],
                    uint2 thread_position [[thread_position_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float inverse_shared[1];
    threadgroup float partials[kSimdgroupWidth];
    device const bfloat* input_row = input + group.y * kHiddenDimension;
    device bfloat* output_row = output + group.y * kHiddenDimension;
    const uint base = thread_position.x * kRmsValuesPerThread;
    float total = 0.0f;
    for (uint i = 0u; i < kRmsValuesPerThread; ++i) {
        const float value = float(input_row[base + i]);
        total += value * value;
    }
    total = simd_sum(total);
    if (simdgroup == 0u) {
        partials[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        partials[simdgroup] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        total = simd_sum(partials[lane]);
        if (lane == 0u) {
            inverse_shared[0] =
                precise::rsqrt(total / float(kHiddenDimension) + kRmsNormEpsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = inverse_shared[0];
    for (uint i = 0u; i < kRmsValuesPerThread; ++i) {
        const uint element = base + i;
        output_row[element] =
            weight[element] * static_cast<bfloat>(float(input_row[element]) * inverse);
    }
}

kernel void residual_blk(device const bfloat* left [[buffer(0)]],
                         device const bfloat* right [[buffer(1)]],
                         device bfloat* output [[buffer(2)]],
                         uint2 element [[thread_position_in_grid]]) {
    const ulong index = ulong(element.y) * kHiddenDimension + element.x;
    output[index] = static_cast<bfloat>(float(left[index]) + float(right[index]));
}

kernel void gdn_project_blk(
    device const bfloat* input [[buffer(0)]], device const uint* qkv_words [[buffer(1)]],
    device const bfloat* qkv_scales [[buffer(2)]],
    device const bfloat* qkv_biases [[buffer(3)]], device const uint* z_words [[buffer(4)]],
    device const bfloat* z_scales [[buffer(5)]], device const bfloat* z_biases [[buffer(6)]],
    device const uint* b_words [[buffer(7)]], device const bfloat* b_scales [[buffer(8)]],
    device const bfloat* b_biases [[buffer(9)]], device const uint* a_words [[buffer(10)]],
    device const bfloat* a_scales [[buffer(11)]], device const bfloat* a_biases [[buffer(12)]],
    device bfloat* projection [[buffer(13)]], constant uint& block [[buffer(14)]],
    uint thread_position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_position >> 5u;
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
    const ulong word_row = ulong(local_row) * kQuantWordsPerRow;
    const ulong group_row = ulong(local_row) * kGroupsPerRow;
    for (uint position = 0u; position < block; position += kPrefillPositionBatch) {
        const uint count = min(block - position, kPrefillPositionBatch);
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        q4_dot_multi4(input, kHiddenDimension, position, count, words + word_row,
                      scales + group_row, biases + group_row, kHiddenDimension, lane,
                      accumulators);
        if (lane == 0u) {
            for (uint i = 0u; i < count; ++i) {
                projection[(position + i) * kGdnProjectionRows + row] =
                    static_cast<bfloat>(accumulators[i]);
            }
        }
    }
}

kernel void attn_project_blk(
    device const bfloat* input [[buffer(0)]], device const uint* qg_words [[buffer(1)]],
    device const bfloat* qg_scales [[buffer(2)]],
    device const bfloat* qg_biases [[buffer(3)]], device const uint* k_words [[buffer(4)]],
    device const bfloat* k_scales [[buffer(5)]], device const bfloat* k_biases [[buffer(6)]],
    device const uint* v_words [[buffer(7)]], device const bfloat* v_scales [[buffer(8)]],
    device const bfloat* v_biases [[buffer(9)]], device bfloat* projection [[buffer(10)]],
    constant uint& block [[buffer(11)]],
    uint thread_position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_position >> 5u;
    if (row >= kAttnProjectionRows) {
        return;
    }
    device const uint* words;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local_row;
    if (row < kAttnQGateRows) {
        words = qg_words;
        scales = qg_scales;
        biases = qg_biases;
        local_row = row;
    } else if (row < kAttnVRowOffset) {
        words = k_words;
        scales = k_scales;
        biases = k_biases;
        local_row = row - kAttnQGateRows;
    } else {
        words = v_words;
        scales = v_scales;
        biases = v_biases;
        local_row = row - kAttnVRowOffset;
    }
    const ulong word_row = ulong(local_row) * kQuantWordsPerRow;
    const ulong group_row = ulong(local_row) * kGroupsPerRow;
    for (uint position = 0u; position < block; position += kPrefillPositionBatch) {
        const uint count = min(block - position, kPrefillPositionBatch);
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        q4_dot_multi4(input, kHiddenDimension, position, count, words + word_row,
                      scales + group_row, biases + group_row, kHiddenDimension, lane,
                      accumulators);
        if (lane == 0u) {
            for (uint i = 0u; i < count; ++i) {
                projection[(position + i) * kAttnProjectionRows + row] =
                    static_cast<bfloat>(accumulators[i]);
            }
        }
    }
}

kernel void outproj_blk(device const bfloat* input [[buffer(0)]],
                        device const uint* words [[buffer(1)]],
                        device const bfloat* scales [[buffer(2)]],
                        device const bfloat* biases [[buffer(3)]],
                        device bfloat* output [[buffer(4)]],
                        constant uint& block [[buffer(5)]],
                        constant uint& input_width [[buffer(6)]],
                        uint thread_position [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_position >> 5u;
    if (row >= kHiddenDimension) {
        return;
    }
    const ulong word_row = ulong(row) * ulong(input_width / 8u);
    const ulong group_row = ulong(row) * ulong(input_width / kQ4GroupSize);
    for (uint position = 0u; position < block; position += kPrefillPositionBatch) {
        const uint count = min(block - position, kPrefillPositionBatch);
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        q4_dot_multi4(input, input_width, position, count, words + word_row,
                      scales + group_row, biases + group_row, input_width, lane, accumulators);
        if (lane == 0u) {
            for (uint i = 0u; i < count; ++i) {
                output[(position + i) * kHiddenDimension + row] =
                    static_cast<bfloat>(accumulators[i]);
            }
        }
    }
}
