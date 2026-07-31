"""Static safety and identity contract for the 512-thread value candidate."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "src/backend/metal/kernels/attention.metal"
FIXTURE_PATH = ROOT / "tools/native/attention_fixture_probe.cpp"
HARNESS_PATH = ROOT / "tools/native/decode_harness.cpp"
PERF_PATH = ROOT / "tools/native/decode_perf_probe.cpp"


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def kernel_function(source: str, name: str) -> str:
    marker = f"kernel void {name}("
    begin = source.index(marker)
    body_begin = source.index("{", begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : index + 1]
    raise AssertionError(f"unterminated kernel {name}")


class DecodeAttentionValueT512SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = KERNEL_PATH.read_text()
        cls.kernel = kernel_function(
            source, "attention_decode_values_gqa8_t512"
        )
        cls.fixture = FIXTURE_PATH.read_text()
        cls.harness = HARNESS_PATH.read_text()
        cls.perf = PERF_PATH.read_text()

    def test_dispatch_storage_and_indices_are_bounded(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn("threadgroupbfloatvalue_tile[32u*256u];", kernel)
        self.assertEqual(32 * 256 * 2, 16_384)
        self.assertLessEqual(16_384, 32_768)
        self.assertIn(
            "constuinttid=tpos.x+32u*(tpos.y+2u*tpos.z);", kernel
        )
        self.assertIn("constuinthead=kv*8u+tpos.z;", kernel)
        self.assertIn(
            "constuintoutput_dim=tpos.y*32u+lane;", kernel
        )
        self.assertIn(
            compact(".width = 32, .height = 2, .depth = 8"),
            compact(self.fixture),
        )
        self.assertEqual(32 * 2 * 8, 512)
        self.assertEqual(63 + 192, 255)

    def test_each_output_preserves_ascending_key_accumulation(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn(
            "for(uinti=tid;i<elems;i+=512u)", kernel
        )
        self.assertIn(
            "for(uintp=0u;p<vn;++p)", kernel
        )
        self.assertIn(
            "floatacc[4]={0.0f,0.0f,0.0f,0.0f};", kernel
        )
        for output, offset in enumerate((0, 64, 128, 192)):
            suffix = "" if offset == 0 else f"+{offset}u"
            self.assertIn(
                f"acc[{output}]+=wp*float("
                f"value_tile[p*256u+output_dim{suffix}]);",
                kernel,
            )
            destination = (
                "dst[output_dim]"
                if offset == 0
                else f"dst[output_dim+{offset}u]"
            )
            self.assertIn(
                f"{destination}=acc[{output}];", kernel
            )
        self.assertIn("dst[256]=w[256];", kernel)
        self.assertIn("dst[257]=w[257];", kernel)

    def test_component_fixture_requires_exact_control_records(self) -> None:
        fixture = compact(self.fixture)
        self.assertIn('"attention_decode_values_gqa8"', self.fixture)
        self.assertIn('"attention_decode_values_gqa8_t512"', self.fixture)
        self.assertIn(
            "partials2_t512==partials2_device", fixture
        )
        self.assertIn(
            "partials1_t512==partials1_control", fixture
        )
        self.assertIn(
            compact(
                "if (!values_t512_two_part_exact ||"
                " !values_t512_one_part_exact) { return 79; }"
            ),
            fixture,
        )

    def test_shared_harness_keeps_the_current_kernel_as_default(self) -> None:
        harness = compact(self.harness)
        self.assertIn(
            compact(
                "value_kernel == DecodeAttentionValueKernel::Gqa8T512"
            ),
            harness,
        )
        self.assertIn(
            'kernel_name="attention_decode_values_gqa8_t512";', harness
        )
        self.assertGreaterEqual(
            self.harness.count(
                "DecodeAttentionValueKernel::Gqa8T1024"
            ),
            1,
        )
        self.assertIn('"value-t1024"', self.perf)
        self.assertIn('"value-t512"', self.perf)
        self.assertIn(
            "required_capacity > plan.tokenizer.maximum_context",
            self.perf,
        )


if __name__ == "__main__":
    unittest.main()
