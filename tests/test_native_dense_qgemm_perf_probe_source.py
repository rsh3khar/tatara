import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    REPOSITORY_ROOT
    / "tools/native/native_dense_qgemm_perf_probe.cpp"
)
CMAKE_PATH = REPOSITORY_ROOT / "CMakeLists.txt"
PREFILL_DENSE_KERNEL_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/prefill_dense.metal"
)
N1_KERNEL_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/native_dense_qgemm.metal"
)


class ContractViolation(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def extract_braced_span(source: str, marker: str) -> tuple[int, int]:
    marker_begin = source.index(marker)
    body_begin = source.index("{", marker_begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return body_begin, index + 1
    raise ContractViolation(f"unterminated braced block after {marker!r}")


def extract_braced_block(source: str, marker: str) -> str:
    begin, end = extract_braced_span(source, marker)
    return source[begin:end]


def replace_once_in_braced_block(
    source: str, marker: str, before: str, after: str
) -> str:
    begin, end = extract_braced_span(source, marker)
    block = source[begin:end]
    require(
        before in block,
        f"mutation target {before!r} absent from {marker!r}",
    )
    return (
        source[:begin]
        + block.replace(before, after, 1)
        + source[end:]
    )


def swap_once_in_braced_block(
    source: str, marker: str, first: str, second: str
) -> str:
    begin, end = extract_braced_span(source, marker)
    block = source[begin:end]
    require(
        block.count(first) == 1 and block.count(second) == 1,
        f"swap targets are not unique in {marker!r}",
    )
    placeholder = "__TATARA_SOURCE_CONTRACT_SWAP__"
    require(
        placeholder not in block,
        "source contains the mutation swap placeholder",
    )
    swapped = block.replace(first, placeholder, 1)
    swapped = swapped.replace(second, first, 1)
    swapped = swapped.replace(placeholder, second, 1)
    return source[:begin] + swapped + source[end:]


def extract_calls(source: str, name: str) -> tuple[list[str], list[int]]:
    calls: list[str] = []
    positions: list[int] = []
    cursor = 0
    marker = f"{name}("
    while True:
        begin = source.find(marker, cursor)
        if begin < 0:
            return calls, positions
        open_parenthesis = begin + len(name)
        depth = 0
        for index in range(open_parenthesis, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(source[begin : index + 1])
                    positions.append(begin)
                    cursor = index + 1
                    break
        else:
            raise ContractViolation(f"unterminated {name} call")


def extract_kernel_signature(source: str, function_name: str) -> str:
    marker = f"kernel void {function_name}("
    begin = source.index(marker)
    end = source.index("{", begin)
    return source[begin:end]


def replace_once_in_kernel_signature(
    source: str, function_name: str, before: str, after: str
) -> str:
    signature = extract_kernel_signature(source, function_name)
    require(
        before in signature,
        f"mutation target {before!r} absent from {function_name!r}",
    )
    return source.replace(
        signature, signature.replace(before, after, 1), 1
    )


def require_exact_calls(
    source: str, name: str, expected: tuple[str, ...], domain: str
) -> None:
    calls, _ = extract_calls(source, name)
    observed = tuple(compact(call) for call in calls)
    require(
        observed == expected,
        f"{domain} {name} ABI changed: {observed!r}",
    )


def require_call_category_order(
    source: str, names: tuple[str, ...], domain: str
) -> None:
    previous_maximum = -1
    for name in names:
        calls, positions = extract_calls(source, name)
        require(calls, f"{domain} omits {name}")
        require(
            min(positions) > previous_maximum,
            f"{domain} does not order every {name} after the prior "
            "call category",
        )
        previous_maximum = max(positions)


def require_order(source: str, markers: tuple[str, ...], domain: str) -> None:
    positions = []
    for marker in markers:
        position = source.find(marker)
        require(position >= 0, f"{domain} omits {marker!r}")
        positions.append(position)
    require(
        positions == sorted(positions),
        f"{domain} reorders {markers!r}",
    )


def parse_schedule(source: str, name: str) -> str:
    match = re.search(
        rf"constexpr\s+std::string_view\s+{name}\s*=\s*"
        rf'"([AB]+)"\s*;',
        source,
        re.DOTALL,
    )
    require(match is not None, f"{name} is absent or not a literal")
    return match.group(1)


def validate_modes_and_cpu_boundary(source: str) -> None:
    main = extract_braced_block(source, "int main(")
    require(
        "--cpu-only|--compile-only|--gpu" in source,
        "usage does not enumerate the three explicit modes",
    )
    require(
        compact(
            'mode != "--cpu-only" && mode != "--compile-only" && '
            'mode != "--gpu"'
        )
        in compact(main),
        "mode parser admits an implicit action",
    )
    cpu_begin, cpu_end = extract_braced_span(
        main, 'if (mode == "--cpu-only")'
    )
    cpu_branch = main[cpu_begin:cpu_end]
    for forbidden in (
        "create_system_device(",
        "create_command_queue(",
        "create_library_with_source(",
        "create_compute_pipeline(",
        "create_command_buffer(",
    ):
        require(
            forbidden not in cpu_branch,
            f"CPU-only branch contains {forbidden}",
        )
    require(
        main.index("create_system_device()") > cpu_end,
        "device creation precedes the CPU-only return",
    )
    require(
        "command buffers submitted: 0" in cpu_branch,
        "CPU-only result does not state zero submissions",
    )

    compile_begin, compile_end = extract_braced_span(
        main, 'if (mode == "--compile-only")'
    )
    compile_branch = main[compile_begin:compile_end]
    require(
        "PASS_COMPILE_ONLY" in compile_branch
        and "kPipelineCount" in compile_branch
        and "kKernelLibraryMlxSteelEnabled" in compile_branch
        and "? 14U" in compile_branch
        and ": 4U" in compile_branch
        and "command buffers submitted: 0" in compile_branch
        and compile_branch.count("return 0;") == 1,
        "compile-only does not report the complete zero-submission pipeline set",
    )
    require(
        main.index("create_pipelines(") < compile_begin,
        "compile-only returns before creating all pipelines",
    )
    require(
        main.index("execute_case(") > compile_end,
        "GPU cases are reachable from compile-only",
    )


def validate_plan_cases_and_resources(source: str) -> None:
    for include in (
        '#include "tatara/generated/model_plan.h"',
        '#include "tatara/runtime/prefill_geometry.h"',
    ):
        require(include in source, f"missing authority include {include}")
    geometry = compact(
        source[
            source.index("constexpr PrefillPolicy kGeometryPolicy")
            : source.index("enum class OperationKind")
        ]
    )
    require(
        "make_prefill_geometry("
        "tatara::model::qwen36::generated::kModelPlan,"
        "kGeometryPolicy)" in geometry,
        "geometry is not derived from the generated model plan",
    )
    for forbidden_dimension in ("12352", "9216", "4096"):
        require(
            forbidden_dimension not in extract_braced_block(
                source, "constexpr std::array<OperationSpec, 3> make_operations("
            ),
            f"operation topology hardcodes {forbidden_dimension}",
        )

    require(
        re.search(
            r"constexpr\s+std::array<std::uint32_t,\s*3>"
            r"\s+kWorkloadRows\s*\{",
            source,
        )
        is not None,
        "workload does not contain exactly three chunk rows",
    )
    require(
        "3925U - kGeometryPolicy.first_chunk_rows -" in source,
        "the tail is not derived from the 3925-row workload",
    )
    require(
        "kWorkloadRows[0] + kWorkloadRows[1] + "
        "kWorkloadRows[2] == 3925U" in source,
        "workload partition does not prove its sum",
    )
    require(
        "constexpr std::size_t kCaseCount = 9;" in source,
        "case count is not frozen to nine",
    )
    cases = extract_braced_block(source, "make_cases()")
    require(
        "for (const OperationSpec& operation : kOperations)" in cases
        and "for (const std::uint32_t rows : kWorkloadRows)" in cases,
        "cases are not the 3x3 operation/chunk product",
    )
    require(
        "std::array<CaseSpec, kCaseCount> kCases = make_cases()" in source,
        "the executable case table does not come from make_cases",
    )

    operations = extract_braced_block(source, "make_operations()")
    region_counts = [
        int(value)
        for value in re.findall(r"\.region_count\s*=\s*(\d+)", operations)
    ]
    require(
        region_counts == [4, 3, 1],
        f"operation regions changed from 4/3/1: {region_counts!r}",
    )
    for field in (
        "kGeometry.gdn_projection_rows",
        "kGeometry.attention_projection_rows",
        "kGeometry.hidden",
        "kGeometry.gdn_value_values",
        "kGeometry.attention_vector_values",
    ):
        require(field in source, f"model-derived shape omits {field}")

    budget = extract_braced_block(source, "bool make_resource_budget(")
    for calculation in (
        "budget.activation_bytes",
        "budget.weight_bytes",
        "budget.output_bytes",
        "budget.gpu_bytes",
        "budget.with_host_snapshot_bytes",
        "checked_multiply(",
        "checked_add(",
        "2U * kGuardElements",
    ):
        require(calculation in budget, f"resource budget omits {calculation}")


def validate_kernel_abis(
    prefill_kernel: str, n1_kernel: str
) -> None:
    signatures = {
        "gdn_project_blk": (
            extract_kernel_signature(
                prefill_kernel, "gdn_project_blk"
            ),
            15,
            (
                "deviceconstbfloat*input[[buffer(0)]]",
                "deviceconstuint*qkv_words[[buffer(1)]]",
                "deviceconstbfloat*qkv_scales[[buffer(2)]]",
                "deviceconstbfloat*qkv_biases[[buffer(3)]]",
                "deviceconstuint*z_words[[buffer(4)]]",
                "deviceconstbfloat*z_scales[[buffer(5)]]",
                "deviceconstbfloat*z_biases[[buffer(6)]]",
                "deviceconstuint*b_words[[buffer(7)]]",
                "deviceconstbfloat*b_scales[[buffer(8)]]",
                "deviceconstbfloat*b_biases[[buffer(9)]]",
                "deviceconstuint*a_words[[buffer(10)]]",
                "deviceconstbfloat*a_scales[[buffer(11)]]",
                "deviceconstbfloat*a_biases[[buffer(12)]]",
                "devicebfloat*projection[[buffer(13)]]",
                "constantuint&block[[buffer(14)]]",
            ),
        ),
        "attn_project_blk": (
            extract_kernel_signature(
                prefill_kernel, "attn_project_blk"
            ),
            12,
            (
                "deviceconstbfloat*input[[buffer(0)]]",
                "deviceconstuint*qg_words[[buffer(1)]]",
                "deviceconstbfloat*qg_scales[[buffer(2)]]",
                "deviceconstbfloat*qg_biases[[buffer(3)]]",
                "deviceconstuint*k_words[[buffer(4)]]",
                "deviceconstbfloat*k_scales[[buffer(5)]]",
                "deviceconstbfloat*k_biases[[buffer(6)]]",
                "deviceconstuint*v_words[[buffer(7)]]",
                "deviceconstbfloat*v_scales[[buffer(8)]]",
                "deviceconstbfloat*v_biases[[buffer(9)]]",
                "devicebfloat*projection[[buffer(10)]]",
                "constantuint&block[[buffer(11)]]",
            ),
        ),
        "outproj_blk": (
            extract_kernel_signature(
                prefill_kernel, "outproj_blk"
            ),
            7,
            (
                "deviceconstbfloat*input[[buffer(0)]]",
                "deviceconstuint*words[[buffer(1)]]",
                "deviceconstbfloat*scales[[buffer(2)]]",
                "deviceconstbfloat*biases[[buffer(3)]]",
                "devicebfloat*output[[buffer(4)]]",
                "constantuint&block[[buffer(5)]]",
                "constantuint&input_width[[buffer(6)]]",
            ),
        ),
        "native_dense_qgemm_q4_bf16_n1": (
            extract_kernel_signature(
                n1_kernel, "native_dense_qgemm_q4_bf16_n1"
            ),
            12,
            (
                "deviceconstbfloat*activations[[buffer(0)]]",
                "deviceconstuint*packed_weights[[buffer(1)]]",
                "deviceconstbfloat*scales[[buffer(2)]]",
                "deviceconstbfloat*biases[[buffer(3)]]",
                "devicebfloat*output[[buffer(4)]]",
                "constantuint&input_rows[[buffer(5)]]",
                "constantuint&output_columns[[buffer(6)]]",
                "constantuint&reduction_columns[[buffer(7)]]",
                (
                    "constantulong&activation_row_stride_elements"
                    "[[buffer(8)]]"
                ),
                (
                    "constantulong&packed_weight_row_stride_words"
                    "[[buffer(9)]]"
                ),
                (
                    "constantulong&parameter_row_stride_elements"
                    "[[buffer(10)]]"
                ),
                (
                    "constantulong&output_row_stride_elements"
                    "[[buffer(11)]]"
                ),
            ),
        ),
    }
    for name, (signature, count, fragments) in signatures.items():
        require(
            re.findall(r"\[\[buffer\((\d+)\)\]\]", signature)
            == [str(index) for index in range(count)],
            f"{name} buffer indices are not contiguous 0..{count - 1}",
        )
        compact_signature = compact(signature)
        for fragment in fragments:
            require(
                fragment in compact_signature,
                f"{name} ABI omits {fragment}",
            )


def validate_encoders(
    source: str, prefill_kernel: str, n1_kernel: str
) -> None:
    validate_kernel_abis(prefill_kernel, n1_kernel)
    exact = extract_braced_block(source, "int encode_exact(")
    n1 = extract_braced_block(source, "int encode_n1(")
    require(
        len(extract_calls(exact, "dispatch_threadgroups")[0]) == 3,
        "exact encoder must contain one dispatch for each operation arm",
    )
    gdn_bind_loop = extract_braced_block(
        exact, "for (std::uint32_t index = 0; index < 4; ++index)"
    )
    attention_bind_loop = extract_braced_block(
        exact, "for (std::uint32_t index = 0; index < 3; ++index)"
    )
    require(
        "dispatch_threadgroups(" not in gdn_bind_loop
        and "dispatch_threadgroups(" not in attention_bind_loop,
        "exact bundled projection dispatches once per region",
    )
    gdn = extract_braced_block(
        exact,
        "if (spec.operation.kind == OperationKind::GdnInput)",
    )
    attention = extract_braced_block(
        exact,
        "if (spec.operation.kind == OperationKind::AttentionInput)",
    )
    _, attention_end = extract_braced_span(
        exact,
        "if (spec.operation.kind == OperationKind::AttentionInput)",
    )
    output = exact[attention_end:]
    require_exact_calls(
        gdn,
        "set_compute_pipeline",
        ("set_compute_pipeline(pass,pipelines.exact_gdn)",),
        "exact GDN",
    )
    require_exact_calls(
        gdn,
        "set_buffer",
        (
            "set_buffer(pass,resources.activations,0,0)",
            "set_buffer(pass,region.packed,0,base)",
            "set_buffer(pass,region.scales,0,base+1U)",
            "set_buffer(pass,region.biases,0,base+2U)",
            "set_buffer(pass,resources.output,output_offset,13)",
        ),
        "exact GDN",
    )
    require_exact_calls(
        gdn,
        "set_bytes",
        ("set_bytes(pass,&spec.rows,sizeof(spec.rows),14)",),
        "exact GDN",
    )
    require_exact_calls(
        gdn,
        "dispatch_threadgroups",
        (
            "dispatch_threadgroups(pass,"
            "{.width=ceil_div(spec.operation.columns,"
            "kExactBundleThreads/kSimdgroupThreads),"
            ".height=1,.depth=1,},"
            "{.width=kExactBundleThreads,.height=1,.depth=1,})",
        ),
        "exact GDN",
    )
    require_call_category_order(
        gdn,
        (
            "set_compute_pipeline",
            "set_buffer",
            "set_bytes",
            "dispatch_threadgroups",
        ),
        "exact GDN",
    )
    require(
        "conststd::uint32_tbase=1U+3U*index;" in compact(gdn)
        and "index<4" in compact(gdn),
        "exact GDN regions do not bind buffers 1..12",
    )
    require_exact_calls(
        attention,
        "set_compute_pipeline",
        ("set_compute_pipeline(pass,pipelines.exact_attention)",),
        "exact attention",
    )
    require_exact_calls(
        attention,
        "set_buffer",
        (
            "set_buffer(pass,resources.activations,0,0)",
            "set_buffer(pass,region.packed,0,base)",
            "set_buffer(pass,region.scales,0,base+1U)",
            "set_buffer(pass,region.biases,0,base+2U)",
            "set_buffer(pass,resources.output,output_offset,10)",
        ),
        "exact attention",
    )
    require_exact_calls(
        attention,
        "set_bytes",
        ("set_bytes(pass,&spec.rows,sizeof(spec.rows),11)",),
        "exact attention",
    )
    require_exact_calls(
        attention,
        "dispatch_threadgroups",
        (
            "dispatch_threadgroups(pass,"
            "{.width=ceil_div(spec.operation.columns,"
            "kExactBundleThreads/kSimdgroupThreads),"
            ".height=1,.depth=1,},"
            "{.width=kExactBundleThreads,.height=1,.depth=1,})",
        ),
        "exact attention",
    )
    require_call_category_order(
        attention,
        (
            "set_compute_pipeline",
            "set_buffer",
            "set_bytes",
            "dispatch_threadgroups",
        ),
        "exact attention",
    )
    require(
        "conststd::uint32_tbase=1U+3U*index;"
        in compact(attention)
        and "index<3" in compact(attention),
        "exact attention regions do not bind buffers 1..9",
    )
    require_exact_calls(
        output,
        "set_compute_pipeline",
        ("set_compute_pipeline(pass,pipelines.exact_output)",),
        "exact output",
    )
    require_exact_calls(
        output,
        "set_buffer",
        (
            "set_buffer(pass,resources.activations,0,0)",
            "set_buffer(pass,region.packed,0,1)",
            "set_buffer(pass,region.scales,0,2)",
            "set_buffer(pass,region.biases,0,3)",
            "set_buffer(pass,resources.output,output_offset,4)",
        ),
        "exact output",
    )
    require_exact_calls(
        output,
        "set_bytes",
        (
            "set_bytes(pass,&spec.rows,sizeof(spec.rows),5)",
            (
                "set_bytes(pass,&spec.operation.reduction,"
                "sizeof(spec.operation.reduction),6)"
            ),
        ),
        "exact output",
    )
    require_exact_calls(
        output,
        "dispatch_threadgroups",
        (
            "dispatch_threadgroups(pass,"
            "{.width=ceil_div(spec.operation.columns,"
            "kExactOutputThreads/kSimdgroupThreads),"
            ".height=1,.depth=1,},"
            "{.width=kExactOutputThreads,.height=1,.depth=1,})",
        ),
        "exact output",
    )
    require_call_category_order(
        output,
        (
            "set_compute_pipeline",
            "set_buffer",
            "set_bytes",
            "dispatch_threadgroups",
        ),
        "exact output",
    )
    require(
        "conststd::uint64_toutput_offset="
        "kGuardElements*sizeof(std::uint16_t);" in compact(exact),
        "exact outputs do not share the guarded body offset",
    )
    require(
        "structCaseSpec{OperationSpecoperation;"
        "std::uint32_trows;}" in compact(source)
        and "std::uint32_tcolumns;" in compact(
            extract_braced_block(source, "struct OperationSpec")
        )
        and "std::uint32_treduction;" in compact(
            extract_braced_block(source, "struct OperationSpec")
        ),
        "exact uint constants are not sourced from uint32 fields",
    )

    region_loop = extract_braced_block(
        n1, "for (std::uint32_t index = 0;"
    )
    compact_loop = compact(region_loop)
    require_exact_calls(
        n1,
        "set_compute_pipeline",
        ("set_compute_pipeline(pass,pipelines.n1)",),
        "N1",
    )
    require(
        "index<spec.operation.region_count" in compact(n1),
        "N1 is not dispatched once per product tensor region",
    )
    require(
        len(extract_calls(region_loop, "dispatch_threadgroups")[0]) == 1,
        "N1 region loop does not contain exactly one dispatch",
    )
    require(
        "(kGuardElements+region.column_begin)*"
        "sizeof(std::uint16_t)" in compact_loop,
        "N1 output pointer omits the guarded region-column offset",
    )
    require(
        "conststd::uint64_toutput_stride="
        "spec.operation.columns;" in compact(n1),
        "N1 output stride is not the full combined projection width",
    )
    for formula in (
        (
            "conststd::uint64_tactivation_stride="
            "spec.operation.reduction;"
        ),
        (
            "conststd::uint64_tpacked_stride_words="
            "spec.operation.reduction/kQ4ValuesPerWord;"
        ),
        (
            "conststd::uint64_tparameter_stride="
            "spec.operation.reduction/kGroupSize;"
        ),
        (
            "conststd::uint64_toutput_stride="
            "spec.operation.columns;"
        ),
    ):
        require(
            formula in compact(n1),
            f"N1 stride formula changed: {formula}",
        )
    require(
        "set_bytes(pass,&output_stride,"
        "sizeof(output_stride),11)" in compact_loop,
        "N1 ABI does not bind the combined output stride at buffer 11",
    )
    require(
        "set_buffer(pass,resources.output,output_offset,4)"
        in compact_loop,
        "N1 ABI does not bind the offset output at buffer 4",
    )
    require_exact_calls(
        region_loop,
        "set_buffer",
        (
            "set_buffer(pass,resources.activations,0,0)",
            "set_buffer(pass,region.packed,0,1)",
            "set_buffer(pass,region.scales,0,2)",
            "set_buffer(pass,region.biases,0,3)",
            "set_buffer(pass,resources.output,output_offset,4)",
        ),
        "N1",
    )
    require_exact_calls(
        region_loop,
        "set_bytes",
        (
            "set_bytes(pass,&spec.rows,sizeof(spec.rows),5)",
            (
                "set_bytes(pass,&region.columns,"
                "sizeof(region.columns),6)"
            ),
            (
                "set_bytes(pass,&spec.operation.reduction,"
                "sizeof(spec.operation.reduction),7)"
            ),
            (
                "set_bytes(pass,&activation_stride,"
                "sizeof(activation_stride),8)"
            ),
            (
                "set_bytes(pass,&packed_stride_words,"
                "sizeof(packed_stride_words),9)"
            ),
            (
                "set_bytes(pass,&parameter_stride,"
                "sizeof(parameter_stride),10)"
            ),
            (
                "set_bytes(pass,&output_stride,"
                "sizeof(output_stride),11)"
            ),
        ),
        "N1",
    )
    require_exact_calls(
        region_loop,
        "dispatch_threadgroups",
        (
            "dispatch_threadgroups(pass,"
            "{.width=ceil_div(region.columns,kN1TileColumns),"
            ".height=ceil_div(spec.rows,kN1TileRows),"
            ".depth=1,},"
            "{.width=kN1Threads,.height=1,.depth=1,})",
        ),
        "N1",
    )
    require_call_category_order(
        n1,
        (
            "set_compute_pipeline",
            "set_buffer",
            "set_bytes",
            "dispatch_threadgroups",
        ),
        "N1",
    )
    for stride in (
        "activation_stride",
        "packed_stride_words",
        "parameter_stride",
        "output_stride",
        "output_offset",
    ):
        require(
            f"conststd::uint64_t{stride}=" in compact(n1),
            f"N1 {stride} is not represented as uint64",
        )

    forbidden_hot_operations = (
        "create_shared_buffer(",
        ".resize(",
        ".reserve(",
        "std::vector",
        "std::cout",
        "std::cerr",
        "printf(",
        "fprintf(",
        "new ",
        "malloc(",
    )
    run_arm = extract_braced_block(source, "RunResult run_arm(")
    for name, block in (
        ("encode_exact", exact),
        ("encode_n1", n1),
        ("run_arm", run_arm),
    ):
        for forbidden in forbidden_hot_operations:
            require(
                forbidden not in block,
                f"{name} contains hot-path operation {forbidden}",
            )


def validate_schedule_and_run_boundary(source: str) -> None:
    warmup = parse_schedule(source, "kWarmupSchedule")
    measured = parse_schedule(source, "kMeasuredSchedule")
    require(
        len(warmup) == 64
        and warmup.count("A") == 32
        and warmup.count("B") == 32,
        "warm-up schedule is not balanced 32/32",
    )
    require(
        len(measured) == 34
        and measured.count("A") == 17
        and measured.count("B") == 17
        and measured[0] == "A"
        and measured[-1] == "A",
        "measured schedule is not bracketed and balanced 17/17",
    )
    require(
        "static_assert(kCommandBuffersPerCase == 100);" in source,
        "per-case command-buffer bound is not 100",
    )
    require(
        "static_assert(kTotalCommandBuffers == 900);" in source,
        "total command-buffer bound is not 900",
    )

    run_arm = extract_braced_block(source, "RunResult run_arm(")
    ordered_calls = (
        "create_command_buffer",
        "begin_compute_pass",
        "end_compute_pass",
        "commit",
        "wait_until_completed_timed",
    )
    positions = []
    for call_name in ordered_calls:
        calls, call_positions = extract_calls(run_arm, call_name)
        require(
            len(calls) == 1,
            f"run_arm has {len(calls)} {call_name} calls",
        )
        positions.append(call_positions[0])
    require(
        positions == sorted(positions),
        "run_arm reorders command-buffer ownership",
    )
    require(
        "wait_until_completed(" not in run_arm,
        "run_arm silently uses the untimed wait",
    )
    require(
        run_arm.count("++submissions;") == 1,
        "run_arm does not count exactly one committed command buffer",
    )
    require_order(
        run_arm,
        (
            "wait_until_completed_timed(",
            "canaries_intact(resources)",
            "completed.timing.gpu_start_seconds",
            "completed.timing.gpu_end_seconds",
            "std::isfinite(gpu_start)",
            "gpu_end > gpu_start",
            "if (require_timing && !valid_timing)",
            "kExitTiming",
        ),
        "run_arm completion/timing",
    )

    execute = extract_braced_block(source, "int execute_case(")
    compact_execute = compact(execute)
    require_order(
        execute,
        (
            "RunResult exact = run_arm(",
            "std::memcpy(",
            "RunResult n1 = run_arm(",
            "staged_oracle_matches(resources)",
            "compare_family(resources, report.numeric)",
            "for (const char scheduled : kWarmupSchedule)",
            "for (const char scheduled : kMeasuredSchedule)",
            "control_ranges_overlap(exact_samples)",
            "report.exact_gpu = summarize(",
        ),
        "execute_case correctness/measurement",
    )
    require(
        "Arm::Exact,true,false,case_submissions" in compact_execute
        and "Arm::N1,true,false,case_submissions" in compact_execute,
        "correctness runs are not the two untimed pre-gate submissions",
    )
    require(
        compact_execute.count(
            "arm,false,true,case_submissions"
        )
        == 2,
        "warm-up and measured runs do not both require timestamps",
    )
    require(
        "case_submissions!=kCommandBuffersPerCase" in compact_execute,
        "execute_case does not enforce 100 submissions",
    )


def validate_total_dispatch_topology(source: str) -> None:
    execute = extract_braced_block(source, "int execute_case(")
    warmup_begin, warmup_end = extract_braced_span(
        execute, "for (const char scheduled : kWarmupSchedule)"
    )
    measured_begin, measured_end = extract_braced_span(
        execute, "for (const char scheduled : kMeasuredSchedule)"
    )
    warmup_loop = execute[warmup_begin:warmup_end]
    measured_loop = execute[measured_begin:measured_end]
    run_calls, run_positions = extract_calls(execute, "run_arm")
    require(
        len(run_calls) == 4,
        "execute_case must contain exactly two correctness and two "
        "schedule-body run_arm call sites",
    )
    require(
        len(extract_calls(warmup_loop, "run_arm")[0]) == 1
        and len(extract_calls(measured_loop, "run_arm")[0]) == 1,
        "each of the two schedules must contain exactly one run_arm site",
    )
    direct_calls = [
        compact(call)
        for call, position in zip(run_calls, run_positions)
        if not (
            warmup_begin <= position < warmup_end
            or measured_begin <= position < measured_end
        )
    ]
    require(
        len(direct_calls) == 2
        and "Arm::Exact,true,false,case_submissions" in direct_calls[0]
        and "Arm::N1,true,false,case_submissions" in direct_calls[1],
        "correctness topology is not exactly one direct Exact then one "
        "direct N1 call",
    )
    require(
        execute.count("for (const char scheduled :") == 2
        and execute.count(
            "for (const char scheduled : kWarmupSchedule)"
        )
        == 1
        and execute.count(
            "for (const char scheduled : kMeasuredSchedule)"
        )
        == 1,
        "execute_case must contain exactly the two frozen schedule loops",
    )

    exact_begin, exact_end = extract_braced_span(source, "int encode_exact(")
    n1_begin, n1_end = extract_braced_span(source, "int encode_n1(")
    steel_begin, steel_end = extract_braced_span(
        source, "int encode_steel("
    )
    fused_begin, fused_end = extract_braced_span(
        source, "int encode_steel_gdn_fused2("
    )
    bm64_begin, bm64_end = extract_braced_span(
        source, "int encode_steel_gdn_bm64("
    )
    bm64_wm2_wn2_begin, bm64_wm2_wn2_end = extract_braced_span(
        source, "int encode_steel_gdn_bm64_wm2_wn2("
    )
    bm64_bk64_begin, bm64_bk64_end = extract_braced_span(
        source, "int encode_steel_gdn_bm64_bk64("
    )
    bm128_begin, bm128_end = extract_braced_span(
        source, "int encode_steel_gdn_bm128("
    )
    bm48_begin, bm48_end = extract_braced_span(
        source, "int encode_steel_gdn_bm48("
    )
    bm96_begin, bm96_end = extract_braced_span(
        source, "int encode_steel_gdn_bm96("
    )
    bn64_begin, bn64_end = extract_braced_span(
        source, "int encode_steel_gdn_bn64("
    )
    bk64_begin, bk64_end = extract_braced_span(
        source, "int encode_steel_gdn_bk64("
    )
    exact = source[exact_begin:exact_end]
    n1 = source[n1_begin:n1_end]
    steel = source[steel_begin:steel_end]
    fused = source[fused_begin:fused_end]
    bm64 = source[bm64_begin:bm64_end]
    bm64_wm2_wn2 = source[
        bm64_wm2_wn2_begin:bm64_wm2_wn2_end
    ]
    bm64_bk64 = source[bm64_bk64_begin:bm64_bk64_end]
    bm128 = source[bm128_begin:bm128_end]
    bm48 = source[bm48_begin:bm48_end]
    bm96 = source[bm96_begin:bm96_end]
    bn64 = source[bn64_begin:bn64_end]
    bk64 = source[bk64_begin:bk64_end]
    outside_encoders = (
        source[:exact_begin]
        + source[exact_end:n1_begin]
        + source[n1_end:steel_begin]
        + source[steel_end:fused_begin]
        + source[fused_end:bm64_begin]
        + source[bm64_end:bm64_wm2_wn2_begin]
        + source[bm64_wm2_wn2_end:bm64_bk64_begin]
        + source[bm64_bk64_end:bm128_begin]
        + source[bm128_end:bm48_begin]
        + source[bm48_end:bm96_begin]
        + source[bm96_end:bn64_begin]
        + source[bn64_end:bk64_begin]
        + source[bk64_end:]
    )
    require(
        len(extract_calls(exact, "dispatch_threadgroups")[0]) == 3,
        "exact encoder dispatch topology is not exactly three sites",
    )
    require(
        len(extract_calls(n1, "dispatch_threadgroups")[0]) == 1,
        "N1 encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(steel, "dispatch_threadgroups")[0]) == 1,
        "Steel encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(fused, "dispatch_threadgroups")[0]) == 1,
        "fused-pair GDN encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(bm64, "dispatch_threadgroups")[0]) == 1,
        "BM64 GDN encoder dispatch topology is not exactly one site",
    )
    require(
        len(
            extract_calls(
                bm64_wm2_wn2, "dispatch_threadgroups"
            )[0]
        )
        == 1,
        "BM64/WM2/WN2 GDN encoder topology is not exactly one site",
    )
    require(
        len(extract_calls(bm64_bk64, "dispatch_threadgroups")[0]) == 1,
        "BM64/BK64 GDN encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(bm128, "dispatch_threadgroups")[0]) == 1,
        "BM128 GDN encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(bm48, "dispatch_threadgroups")[0]) == 2,
        "BM48/tail32 GDN encoder topology is not exactly two sites",
    )
    require(
        len(extract_calls(bm96, "dispatch_threadgroups")[0]) == 2,
        "BM96/tail32 GDN encoder topology is not exactly two sites",
    )
    require(
        len(extract_calls(bn64, "dispatch_threadgroups")[0]) == 1,
        "BN64 GDN encoder dispatch topology is not exactly one site",
    )
    require(
        len(extract_calls(bk64, "dispatch_threadgroups")[0]) == 1,
        "BK64 GDN encoder dispatch topology is not exactly one site",
    )
    require(
        "dispatch_threadgroups(" not in outside_encoders,
        "dispatch exists outside the twelve frozen encoders",
    )
    n1_region_loop = extract_braced_block(
        n1, "for (std::uint32_t index = 0;"
    )
    require(
        len(
            extract_calls(
                n1_region_loop, "dispatch_threadgroups"
            )[0]
        )
        == 1,
        "the sole N1 dispatch is not inside the region loop",
    )

    warmup = parse_schedule(source, "kWarmupSchedule")
    measured = parse_schedule(source, "kMeasuredSchedule")
    operations = extract_braced_block(source, "make_operations()")
    region_counts = [
        int(value)
        for value in re.findall(
            r"\.region_count\s*=\s*(\d+)", operations
        )
    ]
    workload_match = re.search(
        r"constexpr\s+std::array<std::uint32_t,\s*(\d+)>"
        r"\s+kWorkloadRows",
        source,
    )
    require(
        workload_match is not None,
        "cannot derive workload multiplicity",
    )
    workload_count = int(workload_match.group(1))
    exact_runs_per_case = (
        1 + warmup.count("A") + measured.count("A")
    )
    n1_runs_per_case = (
        1 + warmup.count("B") + measured.count("B")
    )
    exact_dispatches = (
        exact_runs_per_case * len(region_counts) * workload_count
    )
    n1_dispatches = (
        n1_runs_per_case * sum(region_counts) * workload_count
    )
    require(
        exact_dispatches == 450
        and n1_dispatches == 1200
        and exact_dispatches + n1_dispatches == 1650,
        "frozen schedules/regions no longer derive 450 Exact + 1200 "
        "N1 = 1650 dispatches",
    )


def validate_execute_case_fail_closed(source: str) -> None:
    execute = extract_braced_block(source, "int execute_case(")
    compact_execute = compact(execute)
    immediate_exits = (
        "if(resource_result!=0){returnresource_result;}",
        (
            "if(!make_resource_budget(spec,report.budget))"
            "{returnkExitExtent;}"
        ),
        "if(exact.exit_code!=0){returnexact.exit_code;}",
        (
            "if(!body_is_finite(resources))"
            "{returnkExitNumericalFamily;}"
        ),
        "if(n1.exit_code!=0){returnn1.exit_code;}",
        (
            "if(!body_is_finite(resources)||"
            "!staged_oracle_matches(resources))"
            "{returnkExitStagedOracle;}"
        ),
        (
            "if(!compare_family(resources,report.numeric))"
            "{returnkExitNumericalFamily;}"
        ),
        "if(warmup.exit_code!=0){returnwarmup.exit_code;}",
        (
            "if(hash_bfloat(output_body(resources))!="
            "report.exact_hash)"
            "{returnkExitExactNondeterminism;}"
        ),
        (
            "if(hash_bfloat(output_body(resources))!="
            "report.n1_hash)"
            "{returnkExitN1Nondeterminism;}"
        ),
        (
            "if(!exact_determinism_checked||"
            "!n1_determinism_checked)"
            "{returnkExitSampleAccounting;}"
        ),
        "if(measured.exit_code!=0){returnmeasured.exit_code;}",
        (
            "if(samples.count>=samples.gpu.size())"
            "{returnkExitSampleAccounting;}"
        ),
        (
            "if(exact_samples.count!=kMeasuredSamplesPerArm||"
            "n1_samples.count!=kMeasuredSamplesPerArm||"
            "case_submissions!=kCommandBuffersPerCase)"
            "{returnkExitSampleAccounting;}"
        ),
        (
            "if(!control_ranges_overlap(exact_samples))"
            "{returnkExitControlDrift;}"
        ),
    )
    for gate in immediate_exits:
        require(
            gate in compact_execute,
            f"execute_case gate is not an immediate typed exit: {gate}",
        )

    warmup_loop = extract_braced_block(
        execute, "for (const char scheduled : kWarmupSchedule)"
    )
    measured_loop = extract_braced_block(
        execute, "for (const char scheduled : kMeasuredSchedule)"
    )
    for name, loop in (
        ("warm-up", warmup_loop),
        ("measured", measured_loop),
    ):
        for output in (
            "std::cout",
            "std::cerr",
            "print_report(",
            "print_distribution(",
            "gpu_p50_speedup",
            "tatara_native_dense_qgemm_perf_case",
        ):
            require(
                output not in loop,
                f"{name} loop emits pre-gate sample/performance output",
            )
    forbidden_output_apis = (
        "std::cerr",
        "std::clog",
        "printf(",
        "fprintf(",
        "puts(",
        "fputs(",
        "putchar(",
        "print_report(",
        "print_distribution(",
    )
    for output_api in forbidden_output_apis:
        require(
            output_api not in execute,
            f"execute_case contains forbidden output API {output_api}",
        )
    require(
        execute.count("std::cout") == 1
        and "gpu_p50_speedup" not in execute
        and "tatara_native_dense_qgemm_perf_case" not in execute,
        "execute_case output is not the sole progress emission",
    )
    progress_output = (
        'std::cout<<"tatara_native_dense_qgemm_perf_progress"'
        '<<"operation="<<spec.operation.name'
        '<<"rows="<<spec.rows'
        '<<"submissions="<<case_submissions'
        "<<'\\n'<<std::flush;"
    )
    require(
        progress_output in compact_execute,
        "execute_case progress output structure changed",
    )
    first_output = execute.find("std::cout")
    require(
        first_output > execute.index(
            "if (!control_ranges_overlap(exact_samples))"
        )
        and first_output > execute.index(
            "report.n1_wall = summarize(n1_samples.wall)"
        )
        and first_output > execute.index(
            "total_submissions += case_submissions"
        ),
        "execute_case output occurs before every correctness, sample, "
        "drift, and summary gate passes",
    )


def validate_reporting_and_cmake(source: str, cmake: str) -> None:
    main = extract_braced_block(source, "int main(")
    standard = main[
        main.index("std::array<CaseReport, kCaseCount> reports{};") :
    ]
    case_loop_begin, case_loop_end = extract_braced_span(
        standard, "for (std::size_t index = 0;"
    )
    case_loop = standard[case_loop_begin:case_loop_end]
    require(
        "print_report(" not in case_loop
        and "tatara_native_dense_qgemm_perf_case" not in case_loop
        and "PASS_GPU" not in case_loop,
        "final evidence is printed before every case passes",
    )
    report_loop_begin, report_loop_end = extract_braced_span(
        standard, "for (const CaseReport& report : reports)"
    )
    report_loop = standard[report_loop_begin:report_loop_end]
    require(
        "print_report(report)" in report_loop,
        "final case reports are not emitted from the complete report array",
    )
    require(
        case_loop_end < report_loop_begin < standard.index("PASS_GPU"),
        "final reports do not follow the complete case loop",
    )
    require(
        source.count("evidence class: component-only") == 12
        and source.count("synthetic-resident projection evidence;") == 2
        and source.count(
            "synthetic-resident same-binary N1/Steel evidence;"
        )
        == 1
        and source.count(
            "synthetic-resident same-binary Steel control/"
        )
        == 9
        and source.count(
            "not model, engine, serving, TTFT or"
        )
        == 2
        and source.count("tokens-per-second evidence") == 12,
        "CPU/GPU result labels do not carry the exact component-only nonclaim",
    )
    require("tok/s" not in source, "probe emits a tok/s label")
    require(
        "tokens_per_second" not in source,
        "probe computes an engine-like token rate",
    )

    compact_cmake = compact(cmake)
    require(
        "add_executable(tatara_native_dense_qgemm_perf_probe"
        "tools/native/native_dense_qgemm_perf_probe.cpp)"
        in compact_cmake,
        "CMake omits the direct performance-probe target",
    )
    require(
        "add_dependencies(tatara_native_dense_qgemm_perf_probe"
        "tatara_generated_kernel_library"
        "tatara_generated_model_plan)" in compact_cmake,
        "performance probe is not bound to both generated authorities",
    )
    require(
        "target_link_libraries(tatara_native_dense_qgemm_perf_probe"
        "PRIVATEtatara_metaltatara_model_plan)" in compact_cmake,
        "performance probe has the wrong native dependencies",
    )
    require(
        re.search(
            r"add_test\s*\([^)]*"
            r"tatara_native_dense_qgemm_perf_probe",
            cmake,
            re.DOTALL,
        )
        is None,
        "GPU-capable performance probe is registered in CTest",
    )


def validate_contract(
    source: str, cmake: str, prefill_kernel: str, n1_kernel: str
) -> None:
    validate_modes_and_cpu_boundary(source)
    validate_plan_cases_and_resources(source)
    validate_encoders(source, prefill_kernel, n1_kernel)
    validate_schedule_and_run_boundary(source)
    validate_total_dispatch_topology(source)
    validate_execute_case_fail_closed(source)
    validate_reporting_and_cmake(source, cmake)


class NativeDenseQgemmPerfProbeSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.cmake = CMAKE_PATH.read_text(encoding="utf-8")
        cls.prefill_kernel = PREFILL_DENSE_KERNEL_PATH.read_text(
            encoding="utf-8"
        )
        cls.n1_kernel = N1_KERNEL_PATH.read_text(encoding="utf-8")

    def test_current_source_satisfies_complete_contract(self) -> None:
        validate_contract(
            self.source,
            self.cmake,
            self.prefill_kernel,
            self.n1_kernel,
        )

    def test_dangerous_mutations_are_rejected(self) -> None:
        exact = extract_braced_block(self.source, "int encode_exact(")
        _, attention_end = extract_braced_span(
            exact,
            "if (spec.operation.kind == OperationKind::AttentionInput)",
        )
        exact_output = exact[attention_end:]
        exact_output_buffer = next(
            call
            for call in extract_calls(exact_output, "set_buffer")[0]
            if compact(call)
            == "set_buffer(pass,resources.output,output_offset,4)"
        )
        exact_output_dispatch = extract_calls(
            exact_output, "dispatch_threadgroups"
        )[0][0]
        n1 = extract_braced_block(self.source, "int encode_n1(")
        n1_region_loop = extract_braced_block(
            n1, "for (std::uint32_t index = 0;"
        )
        n1_output_buffer = next(
            call
            for call in extract_calls(n1_region_loop, "set_buffer")[0]
            if compact(call)
            == "set_buffer(pass,resources.output,output_offset,4)"
        )
        n1_dispatch = extract_calls(
            n1_region_loop, "dispatch_threadgroups"
        )[0][0]
        mutations = {
            "device-before-cpu": self.source.replace(
                '    if (mode == "--cpu-only") {',
                "    auto early_device = create_system_device();\n"
                '    if (mode == "--cpu-only") {',
                1,
            ),
            "region-count": self.source.replace(
                ".region_count = 4,", ".region_count = 5,", 1
            ),
            "missing-column-offset": self.source.replace(
                "(kGuardElements + region.column_begin) *",
                "kGuardElements *",
                1,
            ),
            "local-output-stride": self.source.replace(
                "const std::uint64_t output_stride =\n"
                "        spec.operation.columns;",
                "const std::uint64_t output_stride =\n"
                "        region.columns;",
                1,
            ),
            "wrong-gdn-pipeline": replace_once_in_braced_block(
                self.source,
                "int encode_exact(",
                "pass, pipelines.exact_gdn",
                "pass, pipelines.n1",
            ),
            "wrong-n1-pipeline": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                "pass, pipelines.n1",
                "pass, pipelines.exact_output",
            ),
            "packed-stride-plus-one": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                "spec.operation.reduction / kQ4ValuesPerWord;",
                "spec.operation.reduction / kQ4ValuesPerWord + 1U;",
            ),
            "parameter-stride-plus-one": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                "spec.operation.reduction / kGroupSize;",
                "spec.operation.reduction / kGroupSize + 1U;",
            ),
            "gdn-output-buffer-index": (
                replace_once_in_braced_block(
                    self.source,
                    "int encode_exact(",
                    "pass, resources.output, output_offset, 13",
                    "pass, resources.output, output_offset, 12",
                )
            ),
            "gdn-rows-buffer-index": (
                replace_once_in_braced_block(
                    self.source,
                    "int encode_exact(",
                    "pass, &spec.rows, sizeof(spec.rows), 14",
                    "pass, &spec.rows, sizeof(spec.rows), 13",
                )
            ),
            "gdn-grid-width": replace_once_in_braced_block(
                self.source,
                "int encode_exact(",
                "kExactBundleThreads /\n"
                "                            kSimdgroupThreads",
                "kExactBundleThreads",
            ),
            "gdn-threadgroup-width": replace_once_in_braced_block(
                self.source,
                "int encode_exact(",
                ".width = kExactBundleThreads,\n"
                "                    .height = 1",
                ".width = kExactOutputThreads,\n"
                "                    .height = 1",
            ),
            "n1-output-buffer-index": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                "pass, resources.output, output_offset, 4",
                "pass, resources.output, output_offset, 5",
            ),
            "n1-output-stride-buffer-index": (
                replace_once_in_braced_block(
                    self.source,
                    "int encode_n1(",
                    "pass, &output_stride,\n"
                    "                sizeof(output_stride), 11",
                    "pass, &output_stride,\n"
                    "                sizeof(output_stride), 10",
                )
            ),
            "n1-grid-height": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                "spec.rows, kN1TileRows",
                "region.columns, kN1TileRows",
            ),
            "n1-threadgroup-width": replace_once_in_braced_block(
                self.source,
                "int encode_n1(",
                ".width = kN1Threads,",
                ".width = kExactOutputThreads,",
            ),
            "exact-output-dispatch-before-output-bind": (
                swap_once_in_braced_block(
                    self.source,
                    "int encode_exact(",
                    exact_output_buffer,
                    exact_output_dispatch,
                )
            ),
            "n1-dispatch-before-output-bind": (
                swap_once_in_braced_block(
                    self.source,
                    "int encode_n1(",
                    n1_output_buffer,
                    n1_dispatch,
                )
            ),
            "unbalanced-measurement": self.source.replace(
                "ABABBABAABABBABAABABBABAABBAABABBA",
                "BBABBABAABABBABAABABBABAABBAABABBA",
                1,
            ),
            "wrong-command-bound": self.source.replace(
                "static_assert(kCommandBuffersPerCase == 100);",
                "static_assert(kCommandBuffersPerCase == 99);",
                1,
            ),
            "untimed-wait": self.source.replace(
                "wait_until_completed_timed(",
                "wait_until_completed(",
                1,
            ),
            "canary-before-wait": self.source.replace(
                "    auto completed = wait_until_completed_timed(",
                "    if (!canaries_intact(resources)) {\n"
                "        return {.exit_code = kExitCanary};\n"
                "    }\n"
                "    auto completed = wait_until_completed_timed(",
                1,
            ),
            "hot-allocation": self.source.replace(
                "    prepare_output(resources, poison_body);",
                "    std::vector<int> forbidden_allocation;\n"
                "    prepare_output(resources, poison_body);",
                1,
            ),
            "hot-print": self.source.replace(
                "    prepare_output(resources, poison_body);",
                '    std::cout << "forbidden";\n'
                "    prepare_output(resources, poison_body);",
                1,
            ),
            "measurement-before-correctness": self.source.replace(
                "    RunResult exact = run_arm(",
                "    for (const char scheduled : kMeasuredSchedule) {\n"
                "        (void)scheduled;\n"
                "    }\n"
                "    RunResult exact = run_arm(",
                1,
            ),
            "third-correctness-run": replace_once_in_braced_block(
                self.source,
                "int execute_case(",
                "    bool exact_determinism_checked = false;",
                "    RunResult duplicate = run_arm(\n"
                "        queue, pipelines, resources, Arm::Exact, true,"
                " false,\n"
                "        case_submissions);\n"
                "    if (duplicate.exit_code != 0) {\n"
                "        return duplicate.exit_code;\n"
                "    }\n"
                "    bool exact_determinism_checked = false;",
            ),
            "n1-dispatch-after-region-loop": (
                replace_once_in_braced_block(
                    self.source,
                    "int encode_n1(",
                    "    return 0;\n}",
                    "    if (!encoded(dispatch_threadgroups(\n"
                    "            pass,\n"
                    "            {.width = 1, .height = 1, .depth = 1},\n"
                    "            {.width = kN1Threads, .height = 1,"
                    " .depth = 1}))) {\n"
                    "        return kExitEncode;\n"
                    "    }\n"
                    "    return 0;\n}",
                )
            ),
            "wrong-staged-oracle-exit": (
                replace_once_in_braced_block(
                    self.source,
                    "int execute_case(",
                    "        return kExitStagedOracle;",
                    "        return kExitNumericalFamily;",
                )
            ),
            "ignored-control-drift": (
                replace_once_in_braced_block(
                    self.source,
                    "int execute_case(",
                    "    if (!control_ranges_overlap(exact_samples)) {\n"
                    "        return kExitControlDrift;\n"
                    "    }",
                    "    (void)control_ranges_overlap(exact_samples);",
                )
            ),
            "stderr-before-control-drift": (
                replace_once_in_braced_block(
                    self.source,
                    "int execute_case(",
                    "    if (!control_ranges_overlap(exact_samples)) {",
                    '    std::cerr << "sample";\n'
                    "    if (!control_ranges_overlap(exact_samples)) {",
                )
            ),
            "printf-before-control-drift": (
                replace_once_in_braced_block(
                    self.source.replace(
                        "#include <cstring>",
                        "#include <cstdio>\n#include <cstring>",
                        1,
                    ),
                    "int execute_case(",
                    "    if (!control_ranges_overlap(exact_samples)) {",
                    "    std::printf(\"sample=%f\\n\", "
                    "exact_samples.gpu[0]);\n"
                    "    if (!control_ranges_overlap(exact_samples)) {",
                )
            ),
            "sample-print-before-gates": (
                replace_once_in_braced_block(
                    self.source,
                    "int execute_case(",
                    "        if (measured.exit_code != 0) {",
                    "        std::cout << measured.timing.gpu_milliseconds;\n"
                    "        if (measured.exit_code != 0) {",
                )
            ),
            "early-final-report": self.source.replace(
                "        const int result = execute_case(",
                "        print_report(reports[index]);\n"
                "        const int result = execute_case(",
                1,
            ),
        }
        for name, mutated_source in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(mutated_source, self.source)
                with self.assertRaises(ContractViolation):
                    validate_contract(
                        mutated_source,
                        self.cmake,
                        self.prefill_kernel,
                        self.n1_kernel,
                    )

        kernel_mutations = {
            "gdn-kernel-buffer-type": replace_once_in_kernel_signature(
                self.prefill_kernel,
                "gdn_project_blk",
                "device const bfloat* input [[buffer(0)]]",
                "device const float* input [[buffer(0)]]",
            ),
            "n1-kernel-buffer-index": (
                replace_once_in_kernel_signature(
                    self.n1_kernel,
                    "native_dense_qgemm_q4_bf16_n1",
                "[[buffer(11)]]",
                "[[buffer(12)]]",
                )
            ),
        }
        for name, mutated_kernel in kernel_mutations.items():
            with self.subTest(name=name):
                if name.startswith("gdn-"):
                    prefill_kernel = mutated_kernel
                    n1_kernel = self.n1_kernel
                else:
                    prefill_kernel = self.prefill_kernel
                    n1_kernel = mutated_kernel
                self.assertNotEqual(
                    mutated_kernel,
                    (
                        self.prefill_kernel
                        if name.startswith("gdn-")
                        else self.n1_kernel
                    ),
                )
                with self.assertRaises(ContractViolation):
                    validate_contract(
                        self.source,
                        self.cmake,
                        prefill_kernel,
                        n1_kernel,
                    )

        ctest_mutation = (
            self.cmake
            + "\nadd_test(NAME forbidden COMMAND "
            "tatara_native_dense_qgemm_perf_probe --gpu)\n"
        )
        with self.assertRaises(ContractViolation):
            validate_contract(
                self.source,
                ctest_mutation,
                self.prefill_kernel,
                self.n1_kernel,
            )


if __name__ == "__main__":
    unittest.main()
