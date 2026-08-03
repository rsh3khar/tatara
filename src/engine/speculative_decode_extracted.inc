
struct GdnSnapshot {
    std::vector<std::vector<std::byte>> conv;
    std::vector<std::vector<std::byte>> recurrent;
    std::vector<bool> swapped;
};

backend::metal::MetalBuffer& plane_conv(runtime::DecodeLayerState& state) {
    return state.swapped ? state.first_out : state.first;
}
backend::metal::MetalBuffer& plane_recurrent(runtime::DecodeLayerState& state) {
    return state.swapped ? state.second_out : state.second;
}

GdnSnapshot snapshot_gdn(DecodeStep& step, runtime::DecodeStateSlot& slot) {
    GdnSnapshot snapshot;
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        if (step.schedule[layer] != model::qwen36::LayerKind::GatedDelta) {
            continue;
        }
        auto& state = slot.layers[layer];
        std::vector<std::byte> conv(step.geometry.gdn_conv_state_bytes);
        std::vector<std::byte> recurrent(step.geometry.gdn_recurrent_state_bytes);
        std::memcpy(conv.data(), plane_conv(state).contents(), conv.size());
        std::memcpy(recurrent.data(), plane_recurrent(state).contents(),
                    recurrent.size());
        snapshot.conv.push_back(std::move(conv));
        snapshot.recurrent.push_back(std::move(recurrent));
        snapshot.swapped.push_back(state.swapped);
    }
    return snapshot;
}

void restore_gdn(DecodeStep& step, runtime::DecodeStateSlot& slot,
                 const GdnSnapshot& snapshot) {
    std::size_t cursor = 0;
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        if (step.schedule[layer] != model::qwen36::LayerKind::GatedDelta) {
            continue;
        }
        auto& state = slot.layers[layer];
        state.swapped = snapshot.swapped[cursor];
        std::memcpy(plane_conv(state).contents(), snapshot.conv[cursor].data(),
                    snapshot.conv[cursor].size());
        std::memcpy(plane_recurrent(state).contents(),
                    snapshot.recurrent[cursor].data(),
                    snapshot.recurrent[cursor].size());
        ++cursor;
    }
}

struct PipelineResult {
    int exit_code{43};
    PrefillPipelines pipelines;

    explicit operator bool() const {
        return exit_code == 0;
    }
};

PipelineResult resolve_prefill_pipelines(const MetalDevice& device,
                                         const MetalLibrary& library,
                                         bool indirect) {
    PipelineResult result;
    const auto create_pipeline = [&](const MetalFunction& function) {
        return indirect ? create_indirect_compute_pipeline(device, function)
                        : create_compute_pipeline(device, function);
    };
    constexpr std::array<std::string_view, 23> names{
        "embed_rows_q4", "rms_blk", "residual_blk", "gdn_project_blk",
        "gdn_conv_blk", "gdn_gates_blk", "gdn_recurrence",
        "gdn_recurrence_blk", "gdn_recurrence_gates_blk", "gdn_gate_norm_blk",
        "attn_project_blk", "attn_qk_rope_blk", "attention_partial_blk",
        "attention_combine_blk", "attention_streaming_blk", "outproj_blk",
        "router_q8_block", "router_select_block",
        "router_select_block_parallel", "expert_union", "block_upgate",
        "block_down_partial", "block_down_combine",
    };
    MetalComputePipeline* slots[] = {
        &result.pipelines.embed,
        &result.pipelines.rms,
        &result.pipelines.residual,
        &result.pipelines.gdn_project,
        &result.pipelines.gdn_conv,
        &result.pipelines.gdn_gates,
        &result.pipelines.gdn_recurrence_step,
        &result.pipelines.gdn_recurrence_block,
        &result.pipelines.gdn_recurrence_gates,
        &result.pipelines.gdn_gate_norm,
        &result.pipelines.attn_project,
        &result.pipelines.attn_qk_rope,
        &result.pipelines.attention_partial,
        &result.pipelines.attention_combine,
        &result.pipelines.attention_streaming,
        &result.pipelines.out_projection,
        &result.pipelines.router,
        &result.pipelines.router_select_serial,
        &result.pipelines.router_select_parallel,
        &result.pipelines.expert_union,
        &result.pipelines.expert_upgate,
        &result.pipelines.expert_down,
        &result.pipelines.expert_combine,
    };
    static_assert(names.size() == std::size(slots));
    for (std::size_t index = 0; index < names.size(); ++index) {
        auto function = create_function(library, names[index]);
        if (!function) {
            std::fprintf(stderr, "prefill function lookup failed: %.*s\n",
                         int(names[index].size()), names[index].data());
            return result;
        }
        auto pipeline = create_pipeline(*function.function);
        if (!pipeline) {
            std::fprintf(stderr, "prefill pipeline failed: %.*s\n",
                         int(names[index].size()), names[index].data());
            return result;
        }
        *slots[index] = std::move(*pipeline.pipeline);
    }
    const auto optional_pipeline = [&](std::string_view name,
                                       MetalComputePipeline* slot) {
        auto function = create_function(library, name);
        if (!function) {
            std::fprintf(stderr, "prefill function lookup failed: %.*s\n",
                         int(name.size()), name.data());
            return false;
        }
        auto pipeline = create_pipeline(*function.function);
        if (!pipeline) {
            return false;
        }
        *slot = std::move(*pipeline.pipeline);
        return true;
    };
    namespace kernels = tatara::backend::metal::generated;
    const struct {
        std::string_view name;
        MetalComputePipeline* slot;
    } optional[] = {
        {"attention_staged_scores_blk",
         &result.pipelines.attention_staged_scores},
        {"attention_staged_softmax_blk",
         &result.pipelines.attention_staged_softmax},
        {"attention_staged_values_blk",
         &result.pipelines.attention_staged_values},
        {"native_dense_qgemm_q4_bf16_n1",
         &result.pipelines.native_dense_qgemm},
        {"native_routed_qgemm_r1_build_tasks",
         &result.pipelines.native_routed_task_builder},
        {"native_routed_qgemm_r2_fused_upgate_swiglu",
         &result.pipelines.native_routed_upgate},
        {"native_routed_qgemm_r2_down_partial",
         &result.pipelines.native_routed_down},
        {"expert_union_fused_tasks",
         &result.pipelines.expert_union_fused_tasks},
        {kernels::kKernelLibraryMlxSteelDenseKernelName,
         &result.pipelines.native_dense_steel},
        {kernels::kKernelLibraryMlxSteelRoutedUpgateKernelName,
         &result.pipelines.native_routed_steel_upgate},
        {kernels::kKernelLibraryMlxSteelRoutedDownKernelName,
         &result.pipelines.native_routed_steel_down},
        {"gdn_conv_tape_blk", &result.pipelines.gdn_conv_tape},
        {"gdn_recurrence_gates_tape_blk",
         &result.pipelines.gdn_recurrence_tape},
        {"capture_rows_blk", &result.pipelines.capture_rows},
    };
    static_assert(kernels::kKernelLibraryMlxSteelEnabled,
                  "the admitted serve configuration requires the Steel closure");
    for (const auto& entry : optional) {
        if (!optional_pipeline(entry.name, entry.slot)) {
            return result;
        }
    }
    result.exit_code = 0;
    return result;
}


struct DraftEngine {
    std::map<std::string, MetalBuffer> weights;
    std::map<std::string_view, MetalComputePipeline> pipelines;
    std::vector<MetalBuffer> key_caches;
    std::vector<MetalBuffer> value_caches;
    MetalBuffer positions;
    MetalBuffer ctx_rows;      // [<=16, 2048] projected context chunk
    MetalBuffer features_ctx;  // staging for fc input rows
    MetalBuffer x;             // block residual stream [kBlock, 2048]
    MetalBuffer normed;
    MetalBuffer q;
    MetalBuffer attn;
    MetalBuffer branch;
    MetalBuffer gate;
    MetalBuffer up;
    MetalBuffer swiglu;
    MetalBuffer final_hidden;  // [kBlock, 2048] after final norm
    std::uint32_t capacity{0};
    std::uint32_t cached_rows{0};
    std::uint32_t offset{0};
};

struct Encoder {
    MetalComputePass* pass;
    DraftEngine* engine;
    bool failed{false};

    void check(MetalCommandError error) {
        if (!failed && error != MetalCommandError::None) {
            failed = true;
        }
    }
    void pipeline(std::string_view name) {
        check(set_compute_pipeline(*pass, engine->pipelines.at(name)));
    }
    void buffer(const MetalBuffer& target, std::uint64_t offset, std::uint32_t index) {
        check(set_buffer(*pass, target, offset, index));
    }
    void constant(std::uint32_t value, std::uint32_t index) {
        check(set_bytes(*pass, &value, sizeof(value), index));
    }
    void dispatch(std::uint64_t gx, std::uint64_t gy, std::uint64_t tx) {
        check(dispatch_threadgroups(*pass,
                                    MetalSize{.width = gx, .height = gy, .depth = 1},
                                    MetalSize{.width = tx, .height = 1, .depth = 1}));
    }
    void barrier() {
        check(memory_barrier(*pass));
    }

    void dense(const MetalBuffer& input, std::uint64_t input_offset,
               const std::string& weight, const MetalBuffer& output,
               std::uint64_t output_offset, std::uint32_t rows,
               std::uint32_t reduction, std::uint32_t columns) {
        const bool gemm_shape = rows == 16 && (reduction % 8) == 0 &&
                                (columns % kGemmColumnsPerGroup) == 0;
        pipeline(gemm_shape ? "draft_gemm16_bf16" : "draft_dense_bf16");
        buffer(input, input_offset, 0);
        buffer(engine->weights.at(weight), 0, 1);
        buffer(output, output_offset, 2);
        constant(rows, 3);
        constant(reduction, 4);
        constant(columns, 5);
        if (gemm_shape) {
            dispatch(columns / kGemmColumnsPerGroup, 1, kDenseThreads);
        } else {
            dispatch((columns + kDenseRowsPerGroup - 1) / kDenseRowsPerGroup,
                     1, kDenseThreads);
        }
        barrier();
    }
    void rms(const MetalBuffer& input, std::uint64_t offset,
             const std::string& weight, const MetalBuffer& output,
             std::uint64_t output_offset, std::uint32_t rows,
             std::uint32_t dimension) {
        pipeline("draft_rms_rows");
        buffer(input, offset, 0);
        buffer(engine->weights.at(weight), 0, 1);
        buffer(output, output_offset, 2);
        constant(rows, 3);
        constant(dimension, 4);
        dispatch(rows, 1, dimension / 4);
        barrier();
    }
    void rope(const MetalBuffer& vectors, std::uint64_t offset, std::uint32_t rows,
              std::uint32_t heads, std::uint32_t first_position) {
        pipeline("draft_rope128");
        buffer(vectors, offset, 0);
        constant(rows, 1);
        constant(heads, 2);
        constant(first_position, 3);
        check(dispatch_threadgroups(
            *pass, MetalSize{.width = heads, .height = rows, .depth = 1},
            MetalSize{.width = kDraftHeadDimension / 2, .height = 1, .depth = 1}));
        barrier();
    }
};

void append_context_rows(Encoder& encode, DraftEngine& engine,
                         std::uint32_t rows, std::uint32_t base,
                         std::uint32_t first_position) {
    const std::uint64_t kv_row_bytes =
        std::uint64_t{kDraftKeyValueHeads} * kDraftHeadDimension * 2;
    for (std::uint32_t layer = 0; layer < kDraftLayers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        encode.dense(engine.ctx_rows, 0, prefix + "self_attn.k_proj.weight",
                     engine.key_caches[layer], std::uint64_t{base} * kv_row_bytes,
                     rows, kDraftHidden,
                     kDraftKeyValueHeads * kDraftHeadDimension);
        encode.dense(engine.ctx_rows, 0, prefix + "self_attn.v_proj.weight",
                     engine.value_caches[layer], std::uint64_t{base} * kv_row_bytes,
                     rows, kDraftHidden,
                     kDraftKeyValueHeads * kDraftHeadDimension);
        encode.rms(engine.key_caches[layer], std::uint64_t{base} * kv_row_bytes,
                   prefix + "self_attn.k_norm.weight", engine.key_caches[layer],
                   std::uint64_t{base} * kv_row_bytes,
                   rows * kDraftKeyValueHeads, kDraftHeadDimension);
        encode.rope(engine.key_caches[layer], std::uint64_t{base} * kv_row_bytes,
                    rows, kDraftKeyValueHeads, first_position);
    }
    int* position_values = static_cast<int*>(engine.positions.contents());
    for (std::uint32_t row = 0; row < rows; ++row) {
        position_values[base + row] = static_cast<int>(first_position + row);
    }
}

bool find_copy(const std::vector<std::uint32_t>& history, std::uint32_t staged,
               std::uint32_t needed, std::uint32_t* drafted_out) {
    constexpr std::size_t kGram = 6;
    if (history.size() + 1 < kGram) {
        return false;
    }
    std::uint32_t gram[kGram];
    for (std::size_t index = 0; index < kGram - 1; ++index) {
        gram[index] = history[history.size() - (kGram - 1) + index];
    }
    gram[kGram - 1] = staged;
    for (std::size_t end = history.size(); end >= kGram; --end) {
        if (end + needed > history.size()) {
            continue;
        }
        bool match = true;
        for (std::size_t index = 0; index < kGram && match; ++index) {
            match = history[end - kGram + index] == gram[index];
        }
        if (!match) {
            continue;
        }
        for (std::uint32_t index = 0; index < needed; ++index) {
            drafted_out[index] = history[end + index];
        }
        return true;
    }
    return false;
}

