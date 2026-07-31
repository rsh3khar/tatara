// Derived from MLX (Apple Inc., MIT) quantized.h, the bits==8 affine path.
// See THIRD_PARTY_NOTICES.md and docs/design/PROVENANCE.md.
//
// Affine-Q8 group-64 dot for the router: 256-column blocks, eight contiguous
// values per lane, scale * quant + sum * bias, one final simd_sum.
inline float q8_dot(device const bfloat* x, device const uint* w, device const bfloat* scales,
                    device const bfloat* biases, uint k_size, uint lane) {
    float result = 0.0f;
    device const uchar* bytes = reinterpret_cast<device const uchar*>(w);
    for (uint k = 0; k < k_size; k += 256u) {
        const uint x0 = k + lane * 8u;
        float sum = 0.0f;
        float quant = 0.0f;
        for (uint i = 0; i < 8u; ++i) {
            const float value = float(x[x0 + i]);
            sum += value;
            quant += value * float(bytes[x0 + i]);
        }
        const uint group = (k >> 6u) + (lane >> 3u);
        result += float(scales[group]) * quant + sum * float(biases[group]);
    }
    return simd_sum(result);
}
