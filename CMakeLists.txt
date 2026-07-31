cmake_minimum_required(VERSION 3.25)

project(Tatara VERSION 0.1.0 LANGUAGES CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "Tatara currently requires Apple silicon and macOS")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Tatara currently requires an Apple-silicon target")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR "Tatara requires Apple Clang")
endif()

if(NOT CMAKE_OBJCXX_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR "Tatara requires Apple Clang for Objective-C++")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

include(CTest)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
find_library(TATARA_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(TATARA_METAL_FRAMEWORK Metal REQUIRED)
include(cmake/TataraWarnings.cmake)

set(TATARA_MODEL_PACKAGE
    "${CMAKE_CURRENT_SOURCE_DIR}/catalog/model_packages/qwen36-35b-a3b/model.toml"
    CACHE FILEPATH "Tatara model package to compile")
get_filename_component(TATARA_MODEL_PACKAGE_DIR "${TATARA_MODEL_PACKAGE}" DIRECTORY)
set(TATARA_ARTIFACT_MANIFEST "${TATARA_MODEL_PACKAGE_DIR}/artifact.toml")
set(TATARA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(TATARA_GENERATED_PLAN "${TATARA_GENERATED_DIR}/tatara/generated/model_plan.h")
set(TATARA_PREPARED_FIXTURE "${TATARA_GENERATED_DIR}/fixtures/checkpoint.tatara")

add_custom_command(
    OUTPUT "${TATARA_GENERATED_PLAN}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${TATARA_GENERATED_DIR}/tatara/generated"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_model_plan.py"
            --package "${TATARA_MODEL_PACKAGE}"
            --artifact "${TATARA_ARTIFACT_MANIFEST}"
            --output "${TATARA_GENERATED_PLAN}"
    DEPENDS
        "${TATARA_MODEL_PACKAGE}"
        "${TATARA_ARTIFACT_MANIFEST}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/artifact_manifest.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/qwen36_plan_generation.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_model_plan.py"
    VERBATIM)

add_custom_target(tatara_generated_model_plan DEPENDS "${TATARA_GENERATED_PLAN}")

add_custom_command(
    OUTPUT "${TATARA_PREPARED_FIXTURE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${TATARA_GENERATED_DIR}/fixtures"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_prepared_fixture.py"
            --output "${TATARA_PREPARED_FIXTURE}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/prepared_checkpoint.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_prepared_fixture.py"
    VERBATIM)

add_custom_target(tatara_generated_prepared_fixture DEPENDS "${TATARA_PREPARED_FIXTURE}")

set(TATARA_POPULATION_FIXTURE_DIR "${TATARA_GENERATED_DIR}/fixtures/population")
set(TATARA_POPULATION_FIXTURE_RECORD "${TATARA_POPULATION_FIXTURE_DIR}/population.tatara")

add_custom_command(
    OUTPUT "${TATARA_POPULATION_FIXTURE_RECORD}"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_population_fixture.py"
            --output-directory "${TATARA_POPULATION_FIXTURE_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/prepared_checkpoint.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_population_fixture.py"
    VERBATIM)

add_custom_target(
    tatara_generated_population_fixture
    DEPENDS "${TATARA_POPULATION_FIXTURE_RECORD}")

set(TATARA_GENERATED_KERNEL_LIBRARY "${TATARA_GENERATED_DIR}/tatara/generated/kernel_library.h")
set(
    TATARA_MLX_STEEL_SOURCE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx-steel"
    CACHE PATH
    "Root of the vendored Steel GEMM kernel sources")
set(TATARA_MLX_STEEL_GENERATOR_ARGUMENTS "")
set(TATARA_MLX_STEEL_SOURCES "")
if(NOT TATARA_MLX_STEEL_SOURCE_ROOT STREQUAL "")
    if(NOT IS_DIRECTORY "${TATARA_MLX_STEEL_SOURCE_ROOT}")
        message(FATAL_ERROR
                "TATARA_MLX_STEEL_SOURCE_ROOT is not a directory: "
                "${TATARA_MLX_STEEL_SOURCE_ROOT}")
    endif()
    list(APPEND TATARA_MLX_STEEL_GENERATOR_ARGUMENTS
         --mlx-steel-source-root "${TATARA_MLX_STEEL_SOURCE_ROOT}")
    foreach(
        path IN ITEMS
        LICENSE
        mlx/backend/metal/kernels/quantized.h
        mlx/backend/metal/kernels/quantized_utils.h
        mlx/backend/metal/kernels/steel/defines.h
        mlx/backend/metal/kernels/steel/gemm/loader.h
        mlx/backend/metal/kernels/steel/gemm/mma.h
        mlx/backend/metal/kernels/steel/gemm/transforms.h
        mlx/backend/metal/kernels/steel/utils/integral_constant.h
        mlx/backend/metal/kernels/steel/utils/type_traits.h)
        list(APPEND TATARA_MLX_STEEL_SOURCES
             "${TATARA_MLX_STEEL_SOURCE_ROOT}/${path}")
    endforeach()
endif()

# Every kernel file the generator embeds, so editing any of them regenerates
# the library. Must match KERNEL_FILE_NAMES in tools/generate_kernel_library.py.
set(TATARA_KERNEL_SOURCES "")
foreach(
    name IN ITEMS
    adjudication.metal
    attention.metal
    draft.metal
    embed_row_q4.metal
    gdn.metal
    head.metal
    moe.metal
    native_dense_qgemm.metal
    prefill_attention.metal
    prefill_dense.metal
    prefill_gdn.metal
    prefill_moe.metal
    q4_dot.metal
    q4_dot_packed.metal
    q8_dot.metal
    routed_qgemm_tasks.metal
    rms_only.metal
    verify_m16.metal)
    list(APPEND TATARA_KERNEL_SOURCES
         "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels/${name}")
endforeach()

add_custom_command(
    OUTPUT "${TATARA_GENERATED_KERNEL_LIBRARY}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${TATARA_GENERATED_DIR}/tatara/generated"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_library.py"
            --package "${TATARA_MODEL_PACKAGE}"
            --artifact "${TATARA_ARTIFACT_MANIFEST}"
            --kernel-directory "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels"
            ${TATARA_MLX_STEEL_GENERATOR_ARGUMENTS}
            --output "${TATARA_GENERATED_KERNEL_LIBRARY}"
    DEPENDS
        "${TATARA_MODEL_PACKAGE}"
        "${TATARA_ARTIFACT_MANIFEST}"
        ${TATARA_KERNEL_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/artifact_manifest.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tatara/qwen36_plan_generation.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_library.py"
        ${TATARA_MLX_STEEL_SOURCES}
    VERBATIM)

add_custom_target(
    tatara_generated_kernel_library
    DEPENDS "${TATARA_GENERATED_KERNEL_LIBRARY}")

add_library(tatara_model_plan INTERFACE)
add_dependencies(tatara_model_plan tatara_generated_model_plan)
set_source_files_properties("${TATARA_GENERATED_PLAN}" PROPERTIES GENERATED TRUE)
target_sources(tatara_model_plan INTERFACE "${TATARA_GENERATED_PLAN}")
target_include_directories(
    tatara_model_plan
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${TATARA_GENERATED_DIR}")
target_compile_features(tatara_model_plan INTERFACE cxx_std_20)

add_library(
    tatara_checkpoint STATIC
    src/model/checkpoint_layout.cpp
    src/model/image_population.cpp
    src/model/model_image.cpp
    src/model/prepared_checkpoint.cpp
    src/model/sha256.cpp
    src/model/source_shards.cpp)
# Layout planning reads the package's declared tensor exclusions, so the
# generated plan must exist before the model layer compiles.
add_dependencies(tatara_checkpoint tatara_generated_model_plan)
target_include_directories(
    tatara_checkpoint
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${TATARA_GENERATED_DIR}")
target_compile_features(tatara_checkpoint PUBLIC cxx_std_20)
tatara_enable_warnings(tatara_checkpoint)

add_library(tatara_evidence STATIC src/evidence/wire.cpp src/evidence/codecs.cpp)
target_include_directories(tatara_evidence PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_evidence PUBLIC cxx_std_20)
target_link_libraries(tatara_evidence PUBLIC tatara_checkpoint)
tatara_enable_warnings(tatara_evidence)

add_library(tatara_startup STATIC src/runtime/startup_admission.cpp)
target_include_directories(tatara_startup PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_startup PUBLIC cxx_std_20)
target_link_libraries(tatara_startup PUBLIC tatara_model_plan)
tatara_enable_warnings(tatara_startup)

add_library(tatara_owner_protocol STATIC src/runtime/owner_protocol.cpp)
target_include_directories(tatara_owner_protocol PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_owner_protocol PUBLIC cxx_std_20)
tatara_enable_warnings(tatara_owner_protocol)

add_library(
    tatara_engine STATIC
    src/engine/prefix_cache_domain.cpp
    src/engine/prefix_cache_transaction.cpp
    src/engine/speculative_decode.cpp)
target_include_directories(tatara_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_engine PUBLIC cxx_std_20)
target_link_libraries(
    tatara_engine PUBLIC tatara_owner_protocol tatara_runtime tatara_service)
tatara_enable_warnings(tatara_engine)

add_library(tatara_control_records STATIC src/runtime/control_records.cpp)
target_include_directories(tatara_control_records PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_control_records PUBLIC cxx_std_20)
tatara_enable_warnings(tatara_control_records)

add_library(
    tatara_runtime STATIC
    src/runtime/decode_bindings.cpp
    src/runtime/decode_step.cpp
    src/runtime/execution_identity.cpp
    src/runtime/prefix_state_transfer.cpp
    src/runtime/prefill_command_plan.cpp
    src/runtime/prefill_profile_plan.cpp
    src/runtime/prefill_profile_report.cpp
    src/runtime/prefill_profiler.cpp
    src/runtime/prefill_step.cpp
    src/runtime/quantized_gemm.cpp
    src/runtime/serving_capacity.cpp
    src/runtime/state_slot_reset.cpp)
# The runtime reads one generated package fact directly: the naming table
# decode_bindings composes tensor names from. It does NOT read the kernel
# library -- those constants are pinned against the geometry in the geometry
# test, which takes that dependency itself.
add_dependencies(
    tatara_runtime
    tatara_generated_model_plan
    tatara_generated_kernel_library)
target_include_directories(tatara_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_runtime PUBLIC cxx_std_20)
target_link_libraries(tatara_runtime PUBLIC tatara_checkpoint tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_runtime)

add_library(
    tatara_metal STATIC
    src/backend/metal/command_report.cpp
    src/backend/metal/commands.mm
    src/backend/metal/counter_sampling.mm
    src/backend/metal/model_backing.cpp
    src/backend/metal/pipeline.mm
    src/backend/metal/resources.mm
    src/platform/apple/object_owner.cpp)
target_include_directories(
    tatara_metal
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(tatara_metal PUBLIC cxx_std_20)
target_compile_options(
    tatara_metal
    PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
    tatara_metal
    PUBLIC
        tatara_checkpoint
        "${TATARA_FOUNDATION_FRAMEWORK}"
        "${TATARA_METAL_FRAMEWORK}")
tatara_enable_warnings(tatara_metal)

# The native CLI. Deliberately links no Metal: commands that do not touch the
# GPU must not pay for loading it.
add_library(
    tatara_service STATIC
    src/service/completion_request.mm
    src/service/configuration.cpp
    src/service/record_validation.cpp
    src/service/admission.cpp
    src/service/prefix_cache.cpp
    src/service/observability.cpp
    src/service/http_message.cpp
    src/service/router.cpp
    src/service/server.cpp)
target_include_directories(tatara_service PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_service PUBLIC cxx_std_20)
target_compile_options(
    tatara_service
    PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(tatara_service PUBLIC tatara_checkpoint tatara_text)
tatara_enable_warnings(tatara_service)

add_library(tatara_host STATIC src/host/capability.mm)
target_include_directories(tatara_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_host PUBLIC cxx_std_20)
target_link_libraries(
    tatara_host PUBLIC "${TATARA_FOUNDATION_FRAMEWORK}" "${TATARA_METAL_FRAMEWORK}")
tatara_enable_warnings(tatara_host)

add_library(
    tatara_text STATIC
    src/text/chat_template.mm
    src/text/streaming_decoder.cpp
    src/text/tokenizer.mm)
add_dependencies(tatara_text tatara_generated_model_plan)
target_include_directories(tatara_text PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tatara_text PUBLIC cxx_std_20)
target_compile_options(
    tatara_text
    PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
    tatara_text
    PUBLIC tatara_checkpoint tatara_model_plan "${TATARA_FOUNDATION_FRAMEWORK}")
tatara_enable_warnings(tatara_text)

add_executable(
    tatara_cli
    tools/native/tatara_cli.cpp
    tools/native/tatara_benchmark.cpp
    tools/native/tatara_serve.cpp
    tools/native/decode_harness.cpp
    src/draft/dflash_checkpoint.mm)
set_target_properties(tatara_cli PROPERTIES OUTPUT_NAME tatara)
target_include_directories(tatara_cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools/native")
target_link_libraries(
    tatara_cli PRIVATE tatara_engine tatara_service tatara_host tatara_model_plan tatara_runtime)
add_dependencies(tatara_cli tatara_generated_model_plan tatara_generated_kernel_library)
tatara_enable_warnings(tatara_cli)

add_executable(tatara_metal_resource_probe tools/native/metal_resource_probe.cpp)
target_link_libraries(tatara_metal_resource_probe PRIVATE tatara_metal)
tatara_enable_warnings(tatara_metal_resource_probe)

add_executable(tatara_metal_lifecycle_probe tools/native/metal_lifecycle_probe.cpp)
target_link_libraries(tatara_metal_lifecycle_probe PRIVATE tatara_metal)
tatara_enable_warnings(tatara_metal_lifecycle_probe)

add_executable(tatara_model_load_probe tools/native/model_load_probe.cpp)
add_dependencies(tatara_model_load_probe tatara_generated_model_plan)
target_link_libraries(tatara_model_load_probe PRIVATE tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_model_load_probe)

add_executable(tatara_prefill_library_probe tools/native/prefill_library_probe.cpp)
add_dependencies(
    tatara_prefill_library_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_prefill_library_probe
    PRIVATE tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_prefill_library_probe)

add_executable(
    tatara_native_routed_qgemm_probe
    tools/native/native_routed_qgemm_probe.cpp)
add_dependencies(
    tatara_native_routed_qgemm_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_native_routed_qgemm_probe
    PRIVATE tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_native_routed_qgemm_probe)

add_executable(
    tatara_native_dense_qgemm_probe
    tools/native/native_dense_qgemm_probe.cpp)
add_dependencies(
    tatara_native_dense_qgemm_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_native_dense_qgemm_probe
    PRIVATE tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_native_dense_qgemm_probe)

add_executable(
    tatara_native_dense_qgemm_perf_probe
    tools/native/native_dense_qgemm_perf_probe.cpp)
add_dependencies(
    tatara_native_dense_qgemm_perf_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_native_dense_qgemm_perf_probe
    PRIVATE tatara_metal tatara_model_plan)
tatara_enable_warnings(tatara_native_dense_qgemm_perf_probe)

add_library(tatara_kernel_reference STATIC tests/native/kernel_reference.cpp)
target_include_directories(
    tatara_kernel_reference
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests/native")
target_compile_features(tatara_kernel_reference PUBLIC cxx_std_20)
tatara_enable_warnings(tatara_kernel_reference)

add_executable(
    tatara_kernel_fixture_probe
    tools/native/kernel_fixture_probe.cpp
    tools/native/kernel_probe_main.cpp)
add_dependencies(
    tatara_kernel_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_kernel_fixture_probe
    PRIVATE tatara_metal tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_kernel_fixture_probe)

add_executable(
    tatara_attention_fixture_probe
    tools/native/attention_fixture_probe.cpp
    tools/native/attention_probe_main.cpp)
add_dependencies(
    tatara_attention_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_attention_fixture_probe
    PRIVATE tatara_metal tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_attention_fixture_probe)

add_executable(
    tatara_gdn_fixture_probe
    tools/native/gdn_fixture_probe.cpp
    tools/native/gdn_probe_main.cpp)
add_dependencies(
    tatara_gdn_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_gdn_fixture_probe
    PRIVATE tatara_metal tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_gdn_fixture_probe)

add_executable(tatara_token_exam_probe tools/native/token_exam_probe.cpp)
add_dependencies(
    tatara_token_exam_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_token_exam_probe
    PRIVATE tatara_runtime tatara_model_plan)
tatara_enable_warnings(tatara_token_exam_probe)

# Serial teacher-forced prefill (eighth boundary contract). Shares the boot and
# state-record sequence with the other real-weight probes rather than cloning
# it a third time.
add_executable(
    tatara_prefill_probe
    tools/native/prefill_probe.cpp
    tools/native/decode_harness.cpp)
add_dependencies(
    tatara_prefill_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_prefill_probe
    PRIVATE tatara_runtime tatara_model_plan)
tatara_enable_warnings(tatara_prefill_probe)

# Exact generated block-prefill handoff probe. It reuses the same real-weight
# boot owner as the serial teacher-forced oracle and resolves only the
# additional block pipelines from that already-compiled library.
add_executable(
    tatara_block_prefill_probe
    tools/native/block_prefill_probe.cpp
    tools/native/decode_harness.cpp)
add_dependencies(
    tatara_block_prefill_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_block_prefill_probe
    PRIVATE tatara_runtime tatara_model_plan)
tatara_enable_warnings(tatara_block_prefill_probe)

add_executable(
    tatara_decode_perf_probe
    tools/native/decode_perf_probe.cpp
    tools/native/decode_harness.cpp)
add_dependencies(
    tatara_decode_perf_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_decode_perf_probe
    PRIVATE tatara_runtime tatara_model_plan)
tatara_enable_warnings(tatara_decode_perf_probe)

add_executable(
    tatara_composition_fixture_probe
    tools/native/composition_fixture_probe.cpp
    tools/native/composition_probe_main.cpp)
add_dependencies(
    tatara_composition_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_composition_fixture_probe
    PRIVATE tatara_runtime tatara_model_plan)
tatara_enable_warnings(tatara_composition_fixture_probe)

add_executable(
    tatara_head_fixture_probe
    tools/native/head_fixture_probe.cpp
    tools/native/head_probe_main.cpp)
add_dependencies(
    tatara_head_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_head_fixture_probe
    PRIVATE tatara_metal tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_head_fixture_probe)

add_executable(
    tatara_moe_fixture_probe
    tools/native/moe_fixture_probe.cpp
    tools/native/moe_probe_main.cpp)
add_dependencies(
    tatara_moe_fixture_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_moe_fixture_probe
    PRIVATE tatara_metal tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_moe_fixture_probe)

# Safe-math fixture suite: all six family batteries in one process, so the
# semantics bank re-derives in a single launch rather than six. See the seventh
# boundary contract.
add_executable(
    tatara_fixture_suite_probe
    tools/native/fixture_suite_probe.cpp
    tools/native/kernel_fixture_probe.cpp
    tools/native/gdn_fixture_probe.cpp
    tools/native/attention_fixture_probe.cpp
    tools/native/moe_fixture_probe.cpp
    tools/native/head_fixture_probe.cpp
    tools/native/composition_fixture_probe.cpp)
add_dependencies(
    tatara_fixture_suite_probe
    tatara_generated_kernel_library
    tatara_generated_model_plan)
target_link_libraries(
    tatara_fixture_suite_probe
    PRIVATE tatara_runtime tatara_kernel_reference tatara_model_plan)
tatara_enable_warnings(tatara_fixture_suite_probe)

add_executable(
    tatara_prepared_checkpoint_check
    tools/native/prepared_checkpoint_check.cpp)
add_dependencies(tatara_prepared_checkpoint_check tatara_generated_model_plan)
target_link_libraries(
    tatara_prepared_checkpoint_check
    PRIVATE tatara_checkpoint tatara_model_plan)
tatara_enable_warnings(tatara_prepared_checkpoint_check)

if(BUILD_TESTING)
    add_executable(tatara_qwen36_plan_test tests/native/qwen36_plan_test.cpp)
    add_dependencies(tatara_qwen36_plan_test tatara_generated_model_plan)
    target_link_libraries(tatara_qwen36_plan_test PRIVATE tatara_model_plan)
    tatara_enable_warnings(tatara_qwen36_plan_test)
    add_test(NAME tatara.qwen36_plan COMMAND tatara_qwen36_plan_test)

    add_executable(tatara_server_test tests/native/server_test.cpp)
    target_link_libraries(tatara_server_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_server_test)
    add_test(NAME tatara.server COMMAND tatara_server_test)
    set_tests_properties(tatara.server PROPERTIES TIMEOUT 60)

    add_executable(tatara_router_test tests/native/router_test.cpp)
    target_link_libraries(tatara_router_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_router_test)
    add_test(NAME tatara.router COMMAND tatara_router_test)

    add_executable(tatara_http_message_test tests/native/http_message_test.cpp)
    target_link_libraries(tatara_http_message_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_http_message_test)
    add_test(NAME tatara.http_message COMMAND tatara_http_message_test)

    add_executable(tatara_observability_test tests/native/observability_test.cpp)
    target_link_libraries(tatara_observability_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_observability_test)
    add_test(NAME tatara.observability COMMAND tatara_observability_test)

    add_executable(tatara_admission_test tests/native/admission_test.cpp)
    target_link_libraries(tatara_admission_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_admission_test)
    add_test(NAME tatara.admission COMMAND tatara_admission_test)

    add_executable(tatara_prefix_cache_test tests/native/prefix_cache_test.cpp)
    target_link_libraries(tatara_prefix_cache_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_prefix_cache_test)
    add_test(NAME tatara.prefix_cache COMMAND tatara_prefix_cache_test)

    add_executable(
        tatara_prefix_cache_transaction_test
        tests/native/prefix_cache_transaction_test.cpp)
    target_link_libraries(tatara_prefix_cache_transaction_test PRIVATE tatara_engine)
    tatara_enable_warnings(tatara_prefix_cache_transaction_test)
    add_test(
        NAME tatara.prefix_cache_transaction
        COMMAND tatara_prefix_cache_transaction_test)

    add_executable(
        tatara_prefix_cache_domain_test
        tests/native/prefix_cache_domain_test.cpp)
    target_link_libraries(tatara_prefix_cache_domain_test PRIVATE tatara_engine)
    tatara_enable_warnings(tatara_prefix_cache_domain_test)
    add_test(
        NAME tatara.prefix_cache_domain
        COMMAND tatara_prefix_cache_domain_test)

    add_executable(
        tatara_execution_identity_test
        tests/native/execution_identity_test.cpp)
    target_link_libraries(tatara_execution_identity_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_execution_identity_test)
    add_test(
        NAME tatara.execution_identity
        COMMAND tatara_execution_identity_test)

    add_executable(tatara_record_validation_test tests/native/record_validation_test.cpp)
    target_link_libraries(tatara_record_validation_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_record_validation_test)
    add_test(NAME tatara.record_validation COMMAND tatara_record_validation_test)

    add_executable(tatara_host_capability_test tests/native/host_capability_test.cpp)
    target_link_libraries(tatara_host_capability_test PRIVATE tatara_host)
    tatara_enable_warnings(tatara_host_capability_test)
    add_test(NAME tatara.host_capability COMMAND tatara_host_capability_test)

    add_executable(tatara_service_configuration_test tests/native/service_configuration_test.cpp)
    target_link_libraries(tatara_service_configuration_test PRIVATE tatara_service)
    tatara_enable_warnings(tatara_service_configuration_test)
    add_test(NAME tatara.service_configuration COMMAND tatara_service_configuration_test)

    add_executable(tatara_checkpoint_layout_test tests/native/checkpoint_layout_test.cpp)
    target_link_libraries(tatara_checkpoint_layout_test PRIVATE tatara_checkpoint)
    tatara_enable_warnings(tatara_checkpoint_layout_test)
    add_test(NAME tatara.checkpoint_layout COMMAND tatara_checkpoint_layout_test)

    add_executable(tatara_decode_geometry_test tests/native/decode_geometry_test.cpp)
    target_link_libraries(tatara_decode_geometry_test PRIVATE tatara_model_plan)
    add_dependencies(
        tatara_decode_geometry_test
        tatara_generated_kernel_library
        tatara_generated_model_plan)
    tatara_enable_warnings(tatara_decode_geometry_test)
    add_test(NAME tatara.decode_geometry COMMAND tatara_decode_geometry_test)

    add_executable(tatara_prefill_geometry_test tests/native/prefill_geometry_test.cpp)
    target_link_libraries(tatara_prefill_geometry_test PRIVATE tatara_model_plan)
    add_dependencies(tatara_prefill_geometry_test tatara_generated_model_plan)
    tatara_enable_warnings(tatara_prefill_geometry_test)
    add_test(NAME tatara.prefill_geometry COMMAND tatara_prefill_geometry_test)

    add_executable(
        tatara_quantized_gemm_reference_test
        tests/native/quantized_gemm_reference.cpp
        tests/native/quantized_gemm_reference_test.cpp)
    tatara_enable_warnings(tatara_quantized_gemm_reference_test)

    add_executable(
        raw_quantized_affine_identity_test
        tests/native/raw_quantized_affine_identity.cpp
        tests/native/raw_quantized_affine_identity_test.cpp)
    tatara_enable_warnings(raw_quantized_affine_identity_test)

    add_executable(
        tatara_quantized_gemm_shapes_test
        tests/native/quantized_gemm_shapes_test.cpp)
    target_link_libraries(tatara_quantized_gemm_shapes_test PRIVATE tatara_model_plan)
    add_dependencies(tatara_quantized_gemm_shapes_test tatara_generated_model_plan)
    tatara_enable_warnings(tatara_quantized_gemm_shapes_test)

    add_executable(
        tatara_quantized_gemm_contract_test
        tests/native/quantized_gemm_contract_test.cpp)
    target_link_libraries(tatara_quantized_gemm_contract_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_quantized_gemm_contract_test)

    add_executable(
        tatara_prefill_command_plan_test
        tests/native/prefill_command_plan_test.cpp)
    target_link_libraries(tatara_prefill_command_plan_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_prefill_command_plan_test)
    add_test(
        NAME tatara.prefill_command_plan
        COMMAND tatara_prefill_command_plan_test)

    add_executable(
        tatara_prefill_profile_plan_test
        tests/native/prefill_profile_plan_test.cpp)
    target_link_libraries(tatara_prefill_profile_plan_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_prefill_profile_plan_test)

    add_executable(
        tatara_prefill_profile_report_test
        tests/native/prefill_profile_report_test.cpp)
    target_link_libraries(tatara_prefill_profile_report_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_prefill_profile_report_test)

    add_executable(
        tatara_prefill_profiler_contract_test
        tests/native/prefill_profiler_contract_test.cpp)
    target_link_libraries(tatara_prefill_profiler_contract_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_prefill_profiler_contract_test)

    add_executable(tatara_prefill_step_contract_test tests/native/prefill_step_contract_test.cpp)
    target_link_libraries(tatara_prefill_step_contract_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_prefill_step_contract_test)
    add_test(NAME tatara.prefill_step_contract COMMAND tatara_prefill_step_contract_test)

    add_executable(
        tatara_serving_capacity_test
        tests/native/serving_capacity_test.cpp)
    target_link_libraries(
        tatara_serving_capacity_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_serving_capacity_test)
    add_test(
        NAME tatara.serving_capacity
        COMMAND tatara_serving_capacity_test)

    add_executable(tatara_decode_bindings_test tests/native/decode_bindings_test.cpp)
    target_link_libraries(tatara_decode_bindings_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_decode_bindings_test)
    add_test(NAME tatara.decode_bindings COMMAND tatara_decode_bindings_test)

    add_executable(tatara_decode_state_slot_contract_test
                   tests/native/decode_state_slot_contract_test.cpp)
    target_link_libraries(tatara_decode_state_slot_contract_test PRIVATE tatara_runtime)
    tatara_enable_warnings(tatara_decode_state_slot_contract_test)
    add_test(NAME tatara.decode_state_slot_contract COMMAND tatara_decode_state_slot_contract_test)

    add_executable(tatara_prefix_state_transfer_test
                   tests/native/prefix_state_transfer_test.cpp)
    target_link_libraries(tatara_prefix_state_transfer_test PRIVATE tatara_runtime)
    add_dependencies(tatara_prefix_state_transfer_test tatara_generated_model_plan)
    tatara_enable_warnings(tatara_prefix_state_transfer_test)
    add_test(NAME tatara.prefix_state_transfer COMMAND tatara_prefix_state_transfer_test)

    add_executable(tatara_state_slot_reset_test
                   tests/native/state_slot_reset_test.cpp)
    target_link_libraries(tatara_state_slot_reset_test PRIVATE tatara_runtime)
    add_dependencies(tatara_state_slot_reset_test tatara_generated_model_plan)
    tatara_enable_warnings(tatara_state_slot_reset_test)
    add_test(NAME tatara.state_slot_reset COMMAND tatara_state_slot_reset_test)

    add_executable(
        tatara_prepared_checkpoint_test
        tests/native/prepared_checkpoint_test.cpp)
    add_dependencies(tatara_prepared_checkpoint_test tatara_generated_prepared_fixture)
    target_link_libraries(tatara_prepared_checkpoint_test PRIVATE tatara_checkpoint)
    tatara_enable_warnings(tatara_prepared_checkpoint_test)
    add_test(
        NAME tatara.prepared_checkpoint
        COMMAND tatara_prepared_checkpoint_test "${TATARA_PREPARED_FIXTURE}")

    add_executable(tatara_source_shards_test tests/native/source_shards_test.cpp)
    target_link_libraries(tatara_source_shards_test PRIVATE tatara_checkpoint)
    tatara_enable_warnings(tatara_source_shards_test)
    add_test(NAME tatara.source_shards COMMAND tatara_source_shards_test)

    add_executable(
        tatara_apple_object_owner_test
        tests/native/apple_object_owner_test.cpp)
    target_include_directories(
        tatara_apple_object_owner_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(tatara_apple_object_owner_test PRIVATE tatara_metal)
    tatara_enable_warnings(tatara_apple_object_owner_test)
    add_test(NAME tatara.apple_object_owner COMMAND tatara_apple_object_owner_test)

    add_executable(
        tatara_metal_resources_contract_test
        tests/native/metal_resources_contract_test.cpp)
    target_link_libraries(tatara_metal_resources_contract_test PRIVATE tatara_metal)
    tatara_enable_warnings(tatara_metal_resources_contract_test)
    add_test(
        NAME tatara.metal_resources_contract
        COMMAND tatara_metal_resources_contract_test)

    add_executable(
        tatara_metal_lifecycle_contract_test
        tests/native/metal_lifecycle_contract_test.cpp)
    target_link_libraries(tatara_metal_lifecycle_contract_test PRIVATE tatara_metal)
    tatara_enable_warnings(tatara_metal_lifecycle_contract_test)
    add_test(
        NAME tatara.metal_lifecycle_contract
        COMMAND tatara_metal_lifecycle_contract_test)

    add_executable(
        tatara_command_report_test
        tests/native/command_report_test.cpp
        src/backend/metal/command_report.cpp)
    target_include_directories(
        tatara_command_report_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    tatara_enable_warnings(tatara_command_report_test)
    add_test(NAME tatara.command_report COMMAND tatara_command_report_test)

    add_executable(
        tatara_counter_sampling_contract_test
        tests/native/counter_sampling_contract_test.cpp)
    target_link_libraries(tatara_counter_sampling_contract_test PRIVATE tatara_metal)
    tatara_enable_warnings(tatara_counter_sampling_contract_test)

    add_executable(tatara_sha256_test tests/native/sha256_test.cpp)
    target_link_libraries(tatara_sha256_test PRIVATE tatara_checkpoint)
    tatara_enable_warnings(tatara_sha256_test)
    add_test(NAME tatara.sha256 COMMAND tatara_sha256_test)

    add_executable(tatara_evidence_wire_test tests/native/evidence_wire_test.cpp)
    target_link_libraries(tatara_evidence_wire_test PRIVATE tatara_evidence)
    tatara_enable_warnings(tatara_evidence_wire_test)

    add_executable(tatara_evidence_codecs_test tests/native/evidence_codecs_test.cpp)
    target_link_libraries(tatara_evidence_codecs_test PRIVATE tatara_evidence)
    tatara_enable_warnings(tatara_evidence_codecs_test)

    # These source-gate executables remain direct-only until the frozen
    # 31-case evidence registry is superseded by a reviewed manifest version.
    add_executable(tatara_startup_admission_test tests/native/startup_admission_test.cpp)
    target_link_libraries(tatara_startup_admission_test PRIVATE tatara_startup)
    tatara_enable_warnings(tatara_startup_admission_test)

    add_executable(tatara_owner_protocol_test tests/native/owner_protocol_test.cpp)
    target_link_libraries(tatara_owner_protocol_test PRIVATE tatara_owner_protocol)
    tatara_enable_warnings(tatara_owner_protocol_test)

    add_executable(tatara_spsc_ring_test tests/native/spsc_ring_test.cpp)
    target_link_libraries(tatara_spsc_ring_test PRIVATE tatara_control_records)
    tatara_enable_warnings(tatara_spsc_ring_test)

    add_executable(tatara_control_records_test tests/native/control_records_test.cpp)
    target_link_libraries(tatara_control_records_test PRIVATE tatara_control_records)
    tatara_enable_warnings(tatara_control_records_test)

    add_executable(tatara_image_population_test tests/native/image_population_test.cpp)
    add_dependencies(tatara_image_population_test tatara_generated_population_fixture)
    target_link_libraries(tatara_image_population_test PRIVATE tatara_checkpoint)
    tatara_enable_warnings(tatara_image_population_test)
    add_test(
        NAME tatara.image_population
        COMMAND tatara_image_population_test "${TATARA_POPULATION_FIXTURE_DIR}")

    add_executable(
        tatara_model_backing_contract_test
        tests/native/model_backing_contract_test.cpp)
    target_link_libraries(tatara_model_backing_contract_test PRIVATE tatara_metal)
    tatara_enable_warnings(tatara_model_backing_contract_test)
    add_test(
        NAME tatara.model_backing_contract
        COMMAND tatara_model_backing_contract_test)

    add_executable(tatara_kernel_reference_test tests/native/kernel_reference_test.cpp)
    target_link_libraries(tatara_kernel_reference_test PRIVATE tatara_kernel_reference)
    tatara_enable_warnings(tatara_kernel_reference_test)
    add_test(NAME tatara.kernel_reference COMMAND tatara_kernel_reference_test)

    add_executable(tatara_tokenizer_test tests/native/tokenizer_test.mm)
    add_dependencies(tatara_tokenizer_test tatara_generated_model_plan)
    target_compile_options(
        tatara_tokenizer_test
        PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_link_libraries(tatara_tokenizer_test PRIVATE tatara_text tatara_model_plan)
    tatara_enable_warnings(tatara_tokenizer_test)
    add_test(NAME tatara.tokenizer COMMAND tatara_tokenizer_test)

    add_executable(tatara_chat_template_test tests/native/chat_template_test.mm)
    add_dependencies(tatara_chat_template_test tatara_generated_model_plan)
    target_compile_options(
        tatara_chat_template_test
        PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_link_libraries(
        tatara_chat_template_test PRIVATE tatara_text tatara_model_plan)
    tatara_enable_warnings(tatara_chat_template_test)
    add_test(NAME tatara.chat_template COMMAND tatara_chat_template_test)

    add_executable(
        tatara_completion_request_test
        tests/native/completion_request_test.mm)
    add_dependencies(tatara_completion_request_test tatara_generated_model_plan)
    target_compile_options(
        tatara_completion_request_test
        PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_link_libraries(
        tatara_completion_request_test PRIVATE tatara_service tatara_model_plan)
    tatara_enable_warnings(tatara_completion_request_test)
    add_test(
        NAME tatara.completion_request
        COMMAND tatara_completion_request_test)
endif()
