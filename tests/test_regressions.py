"""Offline regression checks; Windows/driver testing remains necessary.

Run with: python3 -m unittest discover -s tests -v
"""
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")


def function_body(signature):
    start = SOURCE.index(signature)
    opening = SOURCE.index("{", start)
    depth = 1
    for i in range(opening + 1, len(SOURCE)):
        if SOURCE[i] == "{":
            depth += 1
        elif SOURCE[i] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[start:i + 1]
    raise AssertionError("unclosed function: " + signature)


class RegressionTests(unittest.TestCase):
    def test_alias_and_submission_hooks_precede_driver_resolution(self):
        table_start = SOURCE.index("const Entry kAlwaysTable[]")
        table = SOURCE[table_start:SOURCE.index("};", table_start)]
        for name in ("vkQueueSubmit", "vkQueueWaitIdle", "vkWaitForFences"):
            self.assertIn(f'VKSTUB("{name}", {name})', table)
        for stem in ("Buffer", "Image", "ImageSparse"):
            name = f"vkGetDevice{stem}MemoryRequirements"
            self.assertRegex(table, rf'VKSTUB\("{name}KHR",\s*{name}\)')
        for loader in ("vkGetInstanceProcAddr", "vkGetDeviceProcAddr"):
            # Definitions appear after the table; inspect their dispatch order there.
            code = SOURCE[SOURCE.rindex(loader + "("):]
            self.assertLess(code.index("stub::Find(pName)"), code.index("g_realG"))

    def test_null_call_diagnostics_are_bounded_and_do_not_replace_recovery(self):
        handler = function_body("LONG CALLBACK CrashHandler(")
        self.assertIn("InterlockedIncrement(&nullCallDetails) <= 20", handler)
        self.assertIn("ReadProcessMemory(GetCurrentProcess()", handler)
        self.assertIn("EXEC@0 caller bytes [return-16, return)", handler)
        self.assertIn("EXEC@0 stack slot [RSP+0x28]", handler)
        self.assertIn("ep->ContextRecord->Rip = ret", handler)
        self.assertIn("ep->ContextRecord->Rax = 0", handler)

    def test_submit_counters_cover_both_paths(self):
        normal = function_body("vkQueueSubmit(VkQueue queue,")
        newer = function_body("vkQueueSubmit2(VkQueue queue,")
        self.assertIn("InterlockedIncrement(&g_submitCount)", normal)
        self.assertEqual(newer.count("InterlockedIncrement(&g_submitCount)"), 2)
        self.assertIn("g_lastSubmitRes", normal)
        self.assertIn("g_lastSubmitRes", newer)

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_actual_buffer_patch_helper(self):
        helper = function_body("static bool PatchSwrtBufferCreateInfo(")
        source = r'''
#include <cassert>
#include <cstdint>
using VkBufferUsageFlags = uint32_t;
struct VkBufferCreateInfo { VkBufferUsageFlags usage; uint64_t size; };
constexpr uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20;
constexpr uint32_t VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x20000;
constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 1;
constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_DST_BIT = 2;
struct Config { bool softwareRT, swrtPatchBuffers; } g_spvCfg;
bool ready;
void LoadSpvConfig() {}
namespace swpx { bool Ready() { return ready; } }
''' + helper + r'''
int main() {
    VkBufferCreateInfo input{0x80000 | 0x40, 1234}, out{};
    ready = true; g_spvCfg = {true, true};
    assert(PatchSwrtBufferCreateInfo(input, out));
    assert(out.usage == (0x40 | 0x20 | 0x20000 | 1 | 2));
    assert(out.size == input.size && input.usage == (0x80000 | 0x40));
    input.usage = 0x100000 | 0x400;
    assert(PatchSwrtBufferCreateInfo(input, out));
    assert(out.usage == (0x20 | 0x20000 | 1 | 2));
    g_spvCfg.softwareRT = false; assert(!PatchSwrtBufferCreateInfo(input, out));
    g_spvCfg = {true, false}; assert(!PatchSwrtBufferCreateInfo(input, out));
    g_spvCfg = {true, true}; ready = false; assert(!PatchSwrtBufferCreateInfo(input, out));
    ready = true; input.usage = 0x40; assert(!PatchSwrtBufferCreateInfo(input, out));
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
        for signature in ("vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pInfo,",
                          "vkGetDeviceBufferMemoryRequirements(VkDevice device, const VkDeviceBufferMemoryRequirements* pInfo,"):
            # Use the definitions, not the prototypes.
            definition = SOURCE[SOURCE.rindex(signature):]
            self.assertIn("PatchSwrtBufferCreateInfo", definition[:2500])


if __name__ == "__main__":
    unittest.main()
