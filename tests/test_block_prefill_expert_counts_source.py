import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPOSITORY_ROOT / "tools/native/block_prefill_probe.cpp"


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def extract_braced_block(source: str, marker: str) -> str:
    marker_begin = source.index(marker)
    body_begin = source.index("{", marker_begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_begin : index + 1]
    raise AssertionError(f"unterminated braced block after {marker!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_order(source: str, markers: tuple[str, ...], domain: str) -> None:
    positions = []
    for marker in markers:
        position = source.find(marker)
        require(position >= 0, f"{domain}: missing {marker!r}")
        positions.append(position)
    require(positions == sorted(positions), f"{domain}: order changed")


def validate_source(source: str) -> None:
    main = extract_braced_block(source, "int main(")
    prepare = extract_braced_block(
        source, "bool prepare_expert_count_capture("
    )
    capture = extract_braced_block(source, "bool capture_expert_counts(")
    complete = extract_braced_block(
        source, "bool expert_count_capture_complete("
    )
    submit = extract_braced_block(
        source, "SubmissionResult submit_prefill_impl("
    )
    writer = extract_braced_block(
        source, "bool write_expert_count_capture("
    )

    require(
        "constexpr int kExitExpertCountCaptureContract = 118;" in source,
        "capture-contract exit is not stable",
    )
    require(
        "constexpr int kExitExpertCountCaptureWrite = 119;" in source,
        "capture-write exit is not stable",
    )
    compact_main = compact(main)
    for required in (
        "argument_count<5||argument_count>8",
        "argument_count>=7&&!parse_fixture_policy(arguments[6],fixture_policy)",
        "expert_count_capture_requested=argument_count==8",
        "std::string_view{arguments[7]}",
        "fixture_policy.serial||fixture_policy.profile",
        "fixture_policy.schedule!=PrefillSchedule::LayerMajor",
    ):
        require(required in compact_main, f"CLI contract omits {required!r}")
    capture_lifecycle = main[
        main.index("expert_count_capture.emplace()") :
    ]
    require_order(
        capture_lifecycle,
        (
            "prepare_expert_count_capture(",
            "submit_prefill(harness, *prefill.step, prefill_ids,",
            "if (block.exit_code != 0)",
            "write_expert_count_capture(",
        ),
        "main capture lifecycle",
    )
    require(
        "block = submit_prefill(harness, *prefill.step, prefill_ids);"
        in main,
        "default uninstrumented submission changed",
    )
    require(
        "return submit_prefill_impl<false>(harness, prefill, ids, nullptr);"
        in source,
        "default submission does not compile capture work out",
    )
    require(
        "return submit_prefill_impl<true>(harness, prefill, ids,"
        in source,
        "capture submission does not select the instrumented specialization",
    )

    compact_prepare = compact(prepare)
    for required in (
        "policy.schedule!=PrefillSchedule::LayerMajor",
        "policy.first_chunk_rows==0",
        "policy.first_chunk_rows>policy.maximum_block_rows",
        "remaining_rows/policy.maximum_block_rows",
        "remaining_rows%policy.maximum_block_rows!=0u?1u:0u",
        "checked_size_add(1u,tail_chunks,chunk_count_size)",
        "checked_size_add(experts,1u,router_rows)",
        "checked_size_multiply(layer_count,chunk_count,record_count)",
        "checked_size_multiply(record_count,router_rows,word_count)",
        "checked_size_multiply(router_rows,sizeof(std::uint32_t),count_bytes)",
        "prefill.expert_counts.size_bytes()!=count_bytes",
        "capture.records.resize(record_count)",
        "capture.counts.resize(word_count)",
        "capture.written.resize(record_count,0u)",
        "catch(...)",
        "chunk==0?policy.first_chunk_rows:policy.maximum_block_rows",
        "remaining<chunk_limit?remaining:chunk_limit",
        "layer*std::size_t{chunk_count}+chunk",
        "record_index*router_rows",
    ):
        require(
            required in compact_prepare,
            f"preallocation contract omits {required!r}",
        )

    compact_capture = compact(capture)
    for required in (
        "encoded.layer_index>=capture.layer_count",
        "encoded.chunk_ordinal>=capture.chunk_count",
        "capture.written[record_index]!=0u",
        "record.layer_index!=encoded.layer_index",
        "record.chunk_ordinal!=encoded.chunk_ordinal",
        "std::memcpy(destination,source,capture.router_rows*sizeof(std::uint32_t))",
        "destination[capture.experts]!=record.chunk_rows",
        "destination[expert]>record.chunk_rows",
        "routed_sum+=destination[expert]",
        "routed_sum!=std::uint64_t{record.chunk_rows}*capture.active_experts",
        "capture.written[record_index]=1u",
    ):
        require(
            required in compact_capture,
            f"capture invariant omits {required!r}",
        )
    for forbidden in (
        "std::vector",
        ".resize(",
        "std::cerr",
        "std::cout",
        "std::ofstream",
    ):
        require(
            forbidden not in capture,
            f"unit capture performs forbidden hot action {forbidden!r}",
        )

    require_order(
        submit,
        (
            "wait_until_completed_timed(",
            "if (!execution)",
            "capture_expert_counts(",
            "commit_prefill_unit(",
        ),
        "post-completion capture seam",
    )
    require(
        compact(submit).count("ifconstexpr(CaptureExpertCounts)") == 3,
        "default path does not compile capture work out",
    )
    compact_complete = compact(complete)
    for required in (
        "capture.records.size()!=capture.written.size()",
        "capture.counts.size()%capture.router_rows!=0",
        "capture.counts.size()/capture.router_rows!=capture.records.size()",
        "written!=1u",
    ):
        require(
            required in compact_complete,
            f"full-coverage gate omits {required!r}",
        )
    require(
        compact(submit).rfind(
            "!expert_count_capture_complete(*expert_count_capture)"
        )
        > compact(submit).rfind(
            "commit_prefill_unit(prefill,*harness.step)"
        ),
        "full coverage is not checked after all unit commits",
    )

    compact_writer = compact(writer)
    require_order(
        writer,
        (
            "expert_count_capture_complete(capture)",
            "std::ofstream output",
            "schema_version",
            "records",
        ),
        "write-after-coverage",
    )
    for required in (
        '\\"schema_version\\":1',
        '\\"prefill_rows\\":',
        '\\"layers\\":',
        '\\"experts\\":',
        '\\"active_experts\\":',
        '\\"schedule\\":\\"layer_major\\"',
        '\\"first_chunk_rows\\":',
        '\\"maximum_block_rows\\":',
        '\\"layer_index\\":',
        '\\"chunk_ordinal\\":',
        '\\"chunk_rows\\":',
        '\\"routed_row_counts\\":[',
        '\\"shared_rows\\":',
        "catch(...)",
    ):
        require(
            required in compact_writer,
            f"schema omits {required!r}",
        )


def replace_once(source: str, before: str, after: str) -> str:
    require(source.count(before) == 1, f"mutation marker not unique: {before!r}")
    return source.replace(before, after, 1)


class BlockPrefillExpertCountsSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")

    def test_current_source_satisfies_capture_contract(self) -> None:
        validate_source(self.source)

    def test_dangerous_mutations_are_rejected(self) -> None:
        mutations = {
            "seventh-argument-only": replace_once(
                self.source,
                "argument_count >= 7 && !parse_fixture_policy",
                "argument_count == 7 && !parse_fixture_policy",
            ),
            "profiled-capture": replace_once(
                self.source,
                "fixture_policy.serial || fixture_policy.profile",
                "fixture_policy.serial && fixture_policy.profile",
            ),
            "default-instrumented": replace_once(
                self.source,
                "return submit_prefill_impl<false>(harness, prefill, ids, nullptr);",
                (
                    "return submit_prefill_impl<true>(harness, prefill, ids,"
                    " nullptr);"
                ),
            ),
            "unchecked-word-count": replace_once(
                self.source,
                "!checked_size_multiply(record_count, router_rows, word_count)",
                "word_count = record_count * router_rows",
            ),
            "tail-chunk-dropped": replace_once(
                self.source,
                "(remaining_rows % policy.maximum_block_rows != 0u ? 1u : 0u)",
                "0u",
            ),
            "first-chunk-uses-maximum": replace_once(
                self.source,
                "chunk == 0 ? policy.first_chunk_rows",
                "chunk == 0 ? policy.maximum_block_rows",
            ),
            "duplicates-accepted": replace_once(
                self.source,
                "capture.written[record_index] != 0u",
                "capture.written[record_index] == 2u",
            ),
            "shared-count-ignored": replace_once(
                self.source,
                "destination[capture.experts] != record.chunk_rows",
                "destination[capture.experts] == record.chunk_rows",
            ),
            "routed-row-bound-reversed": replace_once(
                self.source,
                "destination[expert] > record.chunk_rows",
                "destination[expert] < record.chunk_rows",
            ),
            "routed-sum-reversed": replace_once(
                self.source,
                "routed_sum !=\n        std::uint64_t{record.chunk_rows}",
                "routed_sum ==\n        std::uint64_t{record.chunk_rows}",
            ),
            "record-never-marked": replace_once(
                self.source,
                "capture.written[record_index] = 1u;",
                "capture.written[record_index] = 0u;",
            ),
            "partial-coverage-accepted": replace_once(
                self.source,
                "written != 1u",
                "written > 1u",
            ),
            "write-without-coverage": replace_once(
                self.source,
                "path.empty() || !expert_count_capture_complete(capture)",
                "path.empty()",
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(mutated, self.source)
                with self.assertRaises(AssertionError):
                    validate_source(mutated)


if __name__ == "__main__":
    unittest.main()
