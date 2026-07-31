// Derived from MLX (Apple Inc., MIT) quantized.h, the bits==4 affine path.
// See THIRD_PARTY_NOTICES.md and docs/design/PROVENANCE.md.
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
