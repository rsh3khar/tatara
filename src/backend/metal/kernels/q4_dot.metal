// Derived from MLX (Apple Inc., MIT) quantized.h, the bits==4 affine path.
// See NOTICE and THIRD_PARTY_LICENSES/.
//
// Shared affine-Q4 group-64 dot product: one simdgroup per output row over
// 512-column blocks. Each lane pre-scales its 16 contiguous input values by
// exact powers of two so masked nibbles recover their place value without a
// shift, accumulating scale * quant + sum * bias per block, with one final
// simd_sum. The arithmetic order is the admitted reference contract.
inline float q4_dot(device const bfloat* x, device const uint* w, device const bfloat* scales,
                    device const bfloat* biases, uint k_size, uint lane) {
    float result = 0.0f;
    device const ushort* words = reinterpret_cast<device const ushort*>(w);
    for (uint k = 0; k < k_size; k += 512u) {
        const uint x0 = k + lane * 16u;
        float xv[16];
        float sum = 0.0f;
        for (uint i = 0; i < 16u; i += 4u) {
            const float a0 = float(x[x0 + i]);
            const float a1 = float(x[x0 + i + 1u]);
            const float a2 = float(x[x0 + i + 2u]);
            const float a3 = float(x[x0 + i + 3u]);
            sum += a0 + a1 + a2 + a3;
            xv[i] = a0;
            xv[i + 1u] = a1 / 16.0f;
            xv[i + 2u] = a2 / 256.0f;
            xv[i + 3u] = a3 / 4096.0f;
        }
        float quant = 0.0f;
        const uint word0 = (k >> 2u) + lane * 4u;
        for (uint i = 0; i < 4u; ++i) {
            const ushort packed = words[word0 + i];
            quant += xv[i * 4u] * float(packed & 0x000fu) +
                     xv[i * 4u + 1u] * float(packed & 0x00f0u) +
                     xv[i * 4u + 2u] * float(packed & 0x0f00u) +
                     xv[i * 4u + 3u] * float(packed & 0xf000u);
        }
        const uint group = (k >> 6u) + (lane >> 2u);
        result += float(scales[group]) * quant + sum * float(biases[group]);
    }
    return simd_sum(result);
}

// Row-batched q4_dot: each 512-column block loads its weight words and
// affine group constants once, then up to kQ4BatchRowsMax batch rows
// accumulate independently against them. Every row's floating-point
// sequence is identical to q4_dot's, so per-row results are bit-identical
// to the serial kernel; only the weight traffic is shared.
constant uint kQ4BatchRowsMax = 16u;
inline void q4_dot_ms(device const bfloat* x0, ulong x_stride, uint row_count,
                      device const uint* w, device const bfloat* scales,
                      device const bfloat* biases, uint k_size, uint lane,
                      thread float* out) {
    float acc[kQ4BatchRowsMax];
    for (uint r = 0u; r < kQ4BatchRowsMax; ++r) {
        acc[r] = 0.0f;
    }
    device const ushort* words = reinterpret_cast<device const ushort*>(w);
    for (uint k = 0; k < k_size; k += 512u) {
        const uint word0 = (k >> 2u) + lane * 4u;
        // The nibble decode is row-independent: hoisted once per block, the
        // per-row products use the identical float multiplicands in the
        // identical order, so per-row results stay bit-identical to q4_dot.
        float wq[16];
        for (uint i = 0; i < 4u; ++i) {
            const ushort word = words[word0 + i];
            wq[i * 4u] = float(word & 0x000fu);
            wq[i * 4u + 1u] = float(word & 0x00f0u);
            wq[i * 4u + 2u] = float(word & 0x0f00u);
            wq[i * 4u + 3u] = float(word & 0xf000u);
        }
        const uint group = (k >> 6u) + (lane >> 2u);
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        const uint x_offset = k + lane * 16u;
        for (uint r = 0u; r < row_count; ++r) {
            device const bfloat* x = x0 + ulong(r) * x_stride;
            float xv[16];
            float sum = 0.0f;
            for (uint i = 0; i < 16u; i += 4u) {
                const float a0 = float(x[x_offset + i]);
                const float a1 = float(x[x_offset + i + 1u]);
                const float a2 = float(x[x_offset + i + 2u]);
                const float a3 = float(x[x_offset + i + 3u]);
                sum += a0 + a1 + a2 + a3;
                xv[i] = a0;
                xv[i + 1u] = a1 / 16.0f;
                xv[i + 2u] = a2 / 256.0f;
                xv[i + 3u] = a3 / 4096.0f;
            }
            float quant = 0.0f;
            for (uint i = 0; i < 4u; ++i) {
                quant += xv[i * 4u] * wq[i * 4u] +
                         xv[i * 4u + 1u] * wq[i * 4u + 1u] +
                         xv[i * 4u + 2u] * wq[i * 4u + 2u] +
                         xv[i * 4u + 3u] * wq[i * 4u + 3u];
            }
            acc[r] += scale * quant + sum * bias;
        }
    }
    for (uint r = 0u; r < row_count; ++r) {
        out[r] = simd_sum(acc[r]);
    }
}

// q4_dot_ms with bfloat4-vectorized input loads: the sixteen scalar x
// reads become four vector loads per row per block. Component extraction,
// conversion, and every accumulation keep q4_dot's exact order, so per-row
// results stay bit-identical; only load width changes.
inline void q4_dot_ms_v2(device const bfloat* x0, ulong x_stride, uint row_count,
                         device const uint* w, device const bfloat* scales,
                         device const bfloat* biases, uint k_size, uint lane,
                         thread float* out) {
    float acc[kQ4BatchRowsMax];
    for (uint r = 0u; r < kQ4BatchRowsMax; ++r) {
        acc[r] = 0.0f;
    }
    device const ushort* words = reinterpret_cast<device const ushort*>(w);
    for (uint k = 0; k < k_size; k += 512u) {
        const uint word0 = (k >> 2u) + lane * 4u;
        float wq[16];
        for (uint i = 0; i < 4u; ++i) {
            const ushort word = words[word0 + i];
            wq[i * 4u] = float(word & 0x000fu);
            wq[i * 4u + 1u] = float(word & 0x00f0u);
            wq[i * 4u + 2u] = float(word & 0x0f00u);
            wq[i * 4u + 3u] = float(word & 0xf000u);
        }
        const uint group = (k >> 6u) + (lane >> 2u);
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        const uint x_offset = k + lane * 16u;
        for (uint r = 0u; r < row_count; ++r) {
            device const bfloat4* xv4 = reinterpret_cast<device const bfloat4*>(
                x0 + ulong(r) * x_stride + x_offset);
            float xv[16];
            float sum = 0.0f;
            for (uint i = 0; i < 4u; ++i) {
                const bfloat4 chunk = xv4[i];
                const float a0 = float(chunk.x);
                const float a1 = float(chunk.y);
                const float a2 = float(chunk.z);
                const float a3 = float(chunk.w);
                sum += a0 + a1 + a2 + a3;
                xv[i * 4u] = a0;
                xv[i * 4u + 1u] = a1 / 16.0f;
                xv[i * 4u + 2u] = a2 / 256.0f;
                xv[i * 4u + 3u] = a3 / 4096.0f;
            }
            float quant = 0.0f;
            for (uint i = 0; i < 4u; ++i) {
                quant += xv[i * 4u] * wq[i * 4u] +
                         xv[i * 4u + 1u] * wq[i * 4u + 1u] +
                         xv[i * 4u + 2u] * wq[i * 4u + 2u] +
                         xv[i * 4u + 3u] * wq[i * 4u + 3u];
            }
            acc[r] += scale * quant + sum * bias;
        }
    }
    for (uint r = 0u; r < row_count; ++r) {
        out[r] = simd_sum(acc[r]);
    }
}
