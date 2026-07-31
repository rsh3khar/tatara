#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"

#include <array>
#include <iostream>
#include <string_view>

int main() {
    using namespace tatara::backend::metal;

    constexpr std::array<std::string_view, 61> kFunctions{
        "embed_rows_q4",
        "rms_blk",
        "residual_blk",
        "gdn_project_blk",
        "gdn_conv_blk",
        "gdn_gates_blk",
        "gdn_recurrence",
        "gdn_recurrence_blk",
        "gdn_recurrence_gates_blk",
        "gdn_gate_norm_blk",
        "attn_project_blk",
        "attn_qk_rope_blk",
        "attention_partial_blk",
        "attention_combine_blk",
        "attention_staged_scores_blk",
        "attention_staged_softmax_blk",
        "attention_staged_values_blk",
        "attention_flash_v2_blk",
        "attention_staged_softmax_bf16_blk",
        "tatara_mlx_steel_attn_scores_nt",
        "tatara_mlx_steel_attn_values_nn",
        "tatara_mlx_steel_attn_scores_nt_unaligned",
        "tatara_mlx_steel_attn_values_nn_unaligned",
        "attention_gate_apply_blk",
        "tatara_mlx_steel_attn_scores_nt_unaligned_l",
        "tatara_mlx_steel_attn_values_nn_unaligned_l",
        "gdn_conv_tape_blk",
        "gdn_recurrence_gates_tape_blk",
        "gdn_tape_replay_blk",
        "attention_streaming_blk",
        "attention_decode_scores_gqa4_simdreduce",
        "attention_decode_scores_gqa8",
        "attention_decode_scores_values_gqa8",
        "attention_decode_vector_2pass_part",
        "attention_decode_vector_2pass_combine",
        "attention_decode_values_gqa8_t512",
        "outproj_blk",
        "router_q8_block",
        "router_select_block",
        "router_select_block_parallel",
        "expert_union",
        "expert_union_fused_tasks",
        "block_upgate",
        "block_down_partial",
        "block_down_combine",
        "native_dense_qgemm_q4_bf16_n1",
        "native_routed_qgemm_r1_fused_upgate_swiglu",
        "native_routed_qgemm_r1_gate",
        "native_routed_qgemm_r1_up_swiglu",
        "native_routed_qgemm_r1_down_partial",
        "native_routed_qgemm_r1_build_tasks",
        "native_routed_qgemm_r2_fused_upgate_swiglu",
        "native_routed_qgemm_r2_down_partial",
        "draft_dense_bf16",
        "draft_gemm16_bf16",
        "draft_rms_rows",
        "draft_rope128",
        "draft_block_attention",
        "draft_swiglu",
        "draft_residual_add",
        "verify_dense_q4_m16",
    };

    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        return 1;
    }
    auto library = create_library_with_source(*device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << "kernel library compilation failed:\n" << library.failure_description << '\n';
        return 2;
    }
    for (const std::string_view name : kFunctions) {
        auto function = create_function(*library.library, name);
        if (!function) {
            std::cerr << "function lookup failed: " << name << '\n';
            return 3;
        }
        auto pipeline =
            create_indirect_compute_pipeline(*device.device, *function.function);
        if (!pipeline) {
            std::cerr << "indirect pipeline creation failed: " << name << '\n'
                      << pipeline.failure_description << '\n';
            return 4;
        }
    }
    if constexpr (generated::kKernelLibraryMlxSteelEnabled) {
        constexpr std::array<std::string_view, 4> kSteelFunctions{
            generated::kKernelLibraryMlxSteelKernelName,
            generated::kKernelLibraryMlxSteelDenseKernelName,
            generated::kKernelLibraryMlxSteelRoutedDownKernelName,
            generated::kKernelLibraryMlxSteelRoutedUpgateKernelName,
        };
        for (const std::string_view name : kSteelFunctions) {
            auto function = create_function(*library.library, name);
            if (!function) {
                std::cerr << "function lookup failed: " << name << '\n';
                return 3;
            }
            auto pipeline = create_indirect_compute_pipeline(
                *device.device, *function.function);
            if (!pipeline) {
                std::cerr << "indirect pipeline creation failed: " << name
                          << '\n'
                          << pipeline.failure_description << '\n';
                return 4;
            }
        }
    }
    constexpr std::size_t kOptionalSteelPipelines =
        generated::kKernelLibraryMlxSteelEnabled ? 4U : 0U;
    std::cout << "block prefill library: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  indirect-capable pipelines: "
              << kFunctions.size() + kOptionalSteelPipelines << '\n'
              << "  command buffers submitted: 0\n";
    return 0;
}
