kernel void embed_row_q4(device const uint* quant_words [[buffer(0)]],
                         device const bfloat* scales [[buffer(1)]],
                         device const bfloat* biases [[buffer(2)]],
                         device const uint* token [[buffer(3)]], device bfloat* row [[buffer(4)]],
                         uint element [[thread_position_in_grid]]) {
    const uint selected = token[0];
    const uint group = element / kQ4GroupSize;
    const float scale = float(scales[selected * kGroupsPerRow + group]);
    const float bias = float(biases[selected * kGroupsPerRow + group]);
    const uint word = quant_words[selected * kQuantWordsPerRow + element / 8u];
    const uint quant = (word >> (4u * (element % 8u))) & 15u;
    row[element] = static_cast<bfloat>(float(quant) * scale + bias);
}
