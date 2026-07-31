import unittest

from tatara.doctor import GIB, assess, render_human


def report(system="Darwin", machine="arm64", chip="Apple M4 Pro",
           memory=48 * GIB, metal_framework="/Metal.framework",
           metal_compiler=None, clang="/clang"):
    return {
        "host": {
            "system": system,
            "machine": machine,
            "chip": chip,
            "memory_bytes": memory,
        },
        "tools": {
            "metal_framework": metal_framework,
            "metal_compiler": metal_compiler,
            "clang": clang,
        },
    }


class DoctorTests(unittest.TestCase):
    def test_supported_m_series_host(self):
        checks = assess(report())
        self.assertTrue(all(c["pass"] for c in checks if c["required"]))

    def test_intel_mac_is_rejected(self):
        checks = assess(report(machine="x86_64", chip="Intel Core i9"))
        failed = {c["name"] for c in checks if c["required"] and not c["pass"]}
        self.assertEqual(failed, {"Apple silicon", "M-series chip"})

    def test_small_memory_is_advisory(self):
        checks = assess(report(memory=16 * GIB))
        memory = next(c for c in checks if c["name"] == "Qwen3.6-35B target memory")
        self.assertFalse(memory["required"])
        self.assertFalse(memory["pass"])

    def test_human_output_distinguishes_warning(self):
        value = report(memory=16 * GIB)
        value["checks"] = assess(value)
        value["supported_host"] = True
        rendered = render_human(value)
        self.assertIn("Tatara doctor: SUPPORTED", rendered)
        self.assertIn("WARN  Qwen3.6-35B target memory", rendered)


if __name__ == "__main__":
    unittest.main()
