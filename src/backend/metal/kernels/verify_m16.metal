kernel void verify_dense_q4_m16(device const bfloat* activations [[buffer(0)]],
                                device const uint* packed_weights [[buffer(1)]],
                                device const bfloat* scales [[buffer(2)]],
                                device const bfloat* biases [[buffer(3)]],
                                device bfloat* outputs [[buffer(4)]],
                                constant uint& rows [[buffer(5)]],
                                constant uint& reduction [[buffer(6)]],
                                constant uint& columns [[buffer(7)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint3 thread_in_group [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simdgroup_width [[threads_per_simdgroup]],
                                uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != 32u || threadgroup_shape.x != 32u ||
        threadgroup_shape.y == 0u ||
        threadgroup_shape.y > kVerifyM16MaxRows ||
        threadgroup_shape.z != 1u || rows == 0u ||
        rows > kVerifyM16MaxRows || threadgroup_shape.y != rows ||
        reduction == 0u || (reduction & 511u) != 0u || columns == 0u) {
        return;
    }
    // Staged words for one 512-value k-chunk of every owned column:
    // columns_per_group x 64 words = 2 KiB at 8 columns.
    threadgroup uint word_tile[kVerifyM16ColumnsPerGroup * 64u];

    const uint row = thread_in_group.y;
    const uint flat_thread = row * 32u + lane;
    const uint thread_count = rows * 32u;
    const uint column_begin = group.x * kVerifyM16ColumnsPerGroup;
    device const bfloat* activation_row =
        activations + ulong(row) * reduction;

    float accumulators[kVerifyM16ColumnsPerGroup];
    for (uint index = 0; index < kVerifyM16ColumnsPerGroup; ++index) {
        accumulators[index] = 0.0f;
    }
    for (uint k = 0; k < reduction; k += 512u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint element = flat_thread;
             element < kVerifyM16ColumnsPerGroup * 64u;
             element += thread_count) {
            const uint local_column = element >> 6u;
            const uint word_index = element & 63u;
            const uint column = column_begin + local_column;
            word_tile[element] =
                column < columns
                    ? packed_weights[ulong(column) * (reduction / 8u) +
                                     (k >> 3u) + word_index]
                    : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint x0 = k + lane * 16u;
        device const bfloat4* activation_vectors =
            reinterpret_cast<device const bfloat4*>(activation_row + x0);
        const float4 pack_a = float4(activation_vectors[0]);
        const float4 pack_b = float4(activation_vectors[1]);
        const float4 pack_c = float4(activation_vectors[2]);
        const float4 pack_d = float4(activation_vectors[3]);
        const float4 even_a = float4(pack_a.x, pack_a.z, pack_b.x, pack_b.z);
        const float4 odd_a = float4(pack_a.y, pack_a.w, pack_b.y, pack_b.w);
        const float4 even_b = float4(pack_c.x, pack_c.z, pack_d.x, pack_d.z);
        const float4 odd_b = float4(pack_c.y, pack_c.w, pack_d.y, pack_d.w);
        const float sum = dot(even_a + odd_a, float4(1.0f)) +
                          dot(even_b + odd_b, float4(1.0f));

        for (uint local_column = 0;
             local_column < kVerifyM16ColumnsPerGroup; ++local_column) {
            const uint column = column_begin + local_column;
            if (column >= columns) {
                continue;
            }
            const uint group_index = (k >> 6u) + (lane >> 2u);
            const float scale_value =
                float(scales[ulong(column) * (reduction / 64u) + group_index]);
            const float bias_value =
                float(biases[ulong(column) * (reduction / 64u) + group_index]);
            const uint word_a = word_tile[local_column * 64u + lane * 2u];
            const uint word_b = word_tile[local_column * 64u + lane * 2u + 1u];
            const float4 quant_even_a =
                unpack_unorm4x8_to_float(word_a & 0x0f0f0f0fu) * 255.0f;
            const float4 quant_odd_a =
                unpack_unorm4x8_to_float((word_a >> 4u) & 0x0f0f0f0fu) * 255.0f;
            const float4 quant_even_b =
                unpack_unorm4x8_to_float(word_b & 0x0f0f0f0fu) * 255.0f;
            const float4 quant_odd_b =
                unpack_unorm4x8_to_float((word_b >> 4u) & 0x0f0f0f0fu) * 255.0f;
            const float quant = dot(even_a, quant_even_a) +
                                dot(odd_a, quant_odd_a) +
                                dot(even_b, quant_even_b) +
                                dot(odd_b, quant_odd_b);
            accumulators[local_column] +=
                scale_value * quant + sum * bias_value;
        }
    }
    for (uint local_column = 0;
         local_column < kVerifyM16ColumnsPerGroup; ++local_column) {
        const uint column = column_begin + local_column;
        if (column >= columns) {
            continue;
        }
        const float total = simd_sum(accumulators[local_column]);
        if (lane == 0u) {
            outputs[ulong(row) * columns + column] = static_cast<bfloat>(total);
        }
    }
}
