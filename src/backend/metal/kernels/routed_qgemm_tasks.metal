struct NativeRoutedQgemmR1ProducedTask {
    uint expert_index;
    uint absolute_route_list_begin;
    uint row_count;
    uint output_row_begin;
};

static_assert(
    sizeof(NativeRoutedQgemmR1ProducedTask) ==
        kNativeRoutedQgemmR1TaskBytes,
    "R1 producer task ABI must remain four packed uint fields");
static_assert(
    kNativeRoutedQgemmR1TileRows > 0u,
    "R1 producer requires a nonzero task tile");
static_assert(
    kNativeRoutedQgemmR1TileColumns > 0u &&
        kMoeExpertDimension % kNativeRoutedQgemmR1TileColumns == 0u &&
        kHiddenDimension % kNativeRoutedQgemmR1TileColumns == 0u,
    "R1 producer column bounds must exactly tile generated plan dimensions");
static_assert(
    kPrefillPackedSlotBits > 0u &&
        kPrefillPackedSlotBits < 32u &&
        kMoeActiveExperts < (1u << kPrefillPackedSlotBits),
    "R1 producer packed slots must represent every routed slot");

inline uint native_routed_qgemm_r1_produce_tasks(
    device const uint* counts,
    device const uint* route_list,
    device NativeRoutedQgemmR1ProducedTask* tasks,
    uint position_count,
    uint expert_count,
    uint active_expert_count,
    uint route_list_expert_stride,
    uint position_capacity,
    ulong route_list_total_extent,
    uint planned_task_capacity,
    uint column_groups,
    uint packed_slot_bits,
    uint include_shared_expert,
    thread uint& produced_task_count) {
    produced_task_count = 0u;
    if (include_shared_expert > 1u ||
        expert_count !=
            kMoeExperts + include_shared_expert ||
        active_expert_count !=
            kMoeActiveExperts + include_shared_expert ||
        packed_slot_bits != kPrefillPackedSlotBits) {
        return kNativeRoutedQgemmR1TaskStatusCountOutOfRange;
    }
    const uint maximum_column_groups = max(
        kMoeExpertDimension / kNativeRoutedQgemmR1TileColumns,
        kHiddenDimension / kNativeRoutedQgemmR1TileColumns);
    if (column_groups == 0u ||
        column_groups > maximum_column_groups) {
        return kNativeRoutedQgemmR1TaskStatusCountOutOfRange;
    }
    const uint maximum_position_count =
        1u << (32u - packed_slot_bits);
    if (position_count == 0u ||
        position_count > maximum_position_count ||
        position_capacity > maximum_position_count ||
        position_count > position_capacity ||
        position_capacity > route_list_expert_stride) {
        return kNativeRoutedQgemmR1TaskStatusCountOutOfRange;
    }

    const ulong final_segment_begin =
        ulong(expert_count - 1u) * ulong(route_list_expert_stride);
    const ulong required_list_entries =
        final_segment_begin + ulong(position_capacity);
    if (required_list_entries > route_list_total_extent) {
        return kNativeRoutedQgemmR1TaskStatusPackedSlotOutOfRange;
    }

    ulong routed_count = 0ul;
    ulong required_task_count = 0ul;
    for (uint expert = 0u; expert < expert_count; ++expert) {
        const uint count = counts[expert];
        if (count > position_count ||
            count > position_capacity) {
            return kNativeRoutedQgemmR1TaskStatusCountOutOfRange;
        }
        const ulong segment_begin =
            ulong(expert) * ulong(route_list_expert_stride);
        const ulong segment_end =
            segment_begin + ulong(position_capacity);
        const ulong listed_end = segment_begin + ulong(count);
        if (segment_begin > route_list_total_extent ||
            segment_end > route_list_total_extent ||
            listed_end > segment_end ||
            (count != 0u && listed_end > 0x100000000ul)) {
            return kNativeRoutedQgemmR1TaskStatusPackedSlotOutOfRange;
        }
        routed_count += ulong(count);
        required_task_count +=
            (ulong(count) +
             ulong(kNativeRoutedQgemmR1TileRows) - 1ul) /
            ulong(kNativeRoutedQgemmR1TileRows);
    }

    const ulong expected_routed_count =
        ulong(position_count) * ulong(active_expert_count);
    if (routed_count != expected_routed_count) {
        return kNativeRoutedQgemmR1TaskStatusRouteConservationFailure;
    }
    if (planned_task_capacity >
            kNativeRoutedQgemmR1TaskCapacity ||
        required_task_count > ulong(planned_task_capacity)) {
        return kNativeRoutedQgemmR1TaskStatusTaskCapacityExceeded;
    }

    for (uint expert = 0u; expert < expert_count; ++expert) {
        const uint count = counts[expert];
        const ulong segment_begin =
            ulong(expert) * ulong(route_list_expert_stride);
        for (uint local_row = 0u; local_row < count; ++local_row) {
            const uint packed =
                route_list[segment_begin + ulong(local_row)];
            const uint position =
                packed >> packed_slot_bits;
            const uint slot =
                packed & ((1u << packed_slot_bits) - 1u);
            const bool shared = expert == kMoeExperts;
            const bool valid_slot =
                shared ? slot == kMoeActiveExperts
                       : slot < kMoeActiveExperts;
            if (position >= position_count ||
                !valid_slot) {
                return kNativeRoutedQgemmR1TaskStatusPackedSlotOutOfRange;
            }
        }
    }

    uint task_index = 0u;
    for (uint expert = 0u; expert < expert_count; ++expert) {
        const uint count = counts[expert];
        const ulong segment_begin =
            ulong(expert) * ulong(route_list_expert_stride);
        for (uint local_begin = 0u; local_begin < count;
             local_begin += kNativeRoutedQgemmR1TileRows) {
            const uint row_count =
                min(kNativeRoutedQgemmR1TileRows, count - local_begin);
            tasks[task_index++] = {
                expert,
                uint(segment_begin + ulong(local_begin)),
                row_count,
                0u,
            };
        }
    }
    produced_task_count = task_index;
    return kNativeRoutedQgemmR1TaskStatusReady;
}

kernel void native_routed_qgemm_r1_build_tasks(
    device const uint* counts [[buffer(0)]],
    device const uint* route_list [[buffer(1)]],
    device NativeRoutedQgemmR1ProducedTask* tasks [[buffer(2)]],
    device uint* indirect_arguments [[buffer(3)]],
    device uint* status [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& active_expert_count [[buffer(7)]],
    constant uint& route_list_expert_stride [[buffer(8)]],
    constant uint& position_capacity [[buffer(9)]],
    constant ulong& route_list_total_extent [[buffer(10)]],
    constant uint& planned_task_capacity [[buffer(11)]],
    constant uint& column_groups [[buffer(12)]],
    constant uint& packed_slot_bits [[buffer(13)]],
    constant uint& include_shared_expert [[buffer(14)]],
    uint3 thread_position [[thread_position_in_grid]]) {
    const bool producer = all(thread_position == uint3(0u));
    uint result = kNativeRoutedQgemmR1TaskStatusNotProduced;
    uint task_count = 0u;
    if (producer) {
        indirect_arguments[0] = 0u;
        indirect_arguments[1] = 0u;
        indirect_arguments[2] = 0u;
        status[0] = kNativeRoutedQgemmR1TaskStatusNotProduced;
        result = native_routed_qgemm_r1_produce_tasks(
            counts,
            route_list,
            tasks,
            position_count,
            expert_count,
            active_expert_count,
            route_list_expert_stride,
            position_capacity,
            route_list_total_extent,
            planned_task_capacity,
            column_groups,
            packed_slot_bits,
            include_shared_expert,
            task_count);
        if (result == kNativeRoutedQgemmR1TaskStatusReady) {
            indirect_arguments[0] = task_count;
            indirect_arguments[1] = column_groups;
            indirect_arguments[2] = 1u;
        } else {
            status[0] = result;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (producer &&
        result == kNativeRoutedQgemmR1TaskStatusReady) {
        status[0] = kNativeRoutedQgemmR1TaskStatusReady;
    }
}
