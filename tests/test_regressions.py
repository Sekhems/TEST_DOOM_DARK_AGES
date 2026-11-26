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

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_as_recovery_classifier_and_safety_gates(self):
        source = r'''
#include "rt_recovery_match.h"
#include <cassert>
using namespace rt_recovery;
int main() {
    Evidence e{};
    e.enabled = e.spoof = e.sameDevice = e.gameCallIndirectNull = true;
    e.argumentsReadable = e.outputWritable = e.validPayload = true;
    e.arg2 = 1; e.inputType = 1000150000u; e.outputType = 1000150020u;
    assert(Classify(e) == Target::BuildSizes);
    Evidence x = e; x.enabled = false; assert(Classify(x) == Target::None);
    x = e; x.spoof = false; assert(Classify(x) == Target::None);
    x = e; x.sameDevice = false; assert(Classify(x) == Target::None);
    x = e; x.gameCallIndirectNull = false; assert(Classify(x) == Target::None);
    x = e; x.argumentsReadable = false; assert(Classify(x) == Target::None);
    x = e; x.outputWritable = false; assert(Classify(x) == Target::None);
    x = e; x.validPayload = false; assert(Classify(x) == Target::None);
    x = e; x.arg2 = 0; assert(Classify(x) == Target::None);
    x = e; x.outputType++; assert(Classify(x) == Target::None);
    e.inputType = 1000150017u; e.outputType = 0; e.arg2 = 0x12345678;
    assert(Classify(e) == Target::CreateAS);
    e.outputWritable = false; assert(Classify(e) == Target::None);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
        recovery = function_body("bool TryRedirectMissingASCall(")
        for gate in ("!g_spvCfg.recoverMissingASCalls", "!g_rtSpoof", "c->Rcx !=",
                     "codePage.AllocationBase != GetModuleHandleW(nullptr)",
                     "call[0] != 0xFF", "call[1] != 0x15", "target != 0",
                     "c->R8 != 0", "RecoveryRange(out, sizeof(output), true)"):
            self.assertIn(gate, recovery)
        self.assertIn("&stub::GetASBuildSizes", recovery)
        self.assertIn("&stub::CreateAS", recovery)

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_redirect_logic_with_mocked_windows_memory(self):
        prelude = r'''
#include "rt_recovery_match.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <vector>
using SIZE_T = size_t; using DWORD = uint32_t; using LONG = long;
using VkDevice = void*; using VkAccelerationStructureKHR = void*;
constexpr int VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR = 1;
constexpr int VK_NULL_HANDLE = 0;
struct VkAccelerationStructureBuildGeometryInfoKHR { uint32_t sType, geometryCount; };
struct VkAccelerationStructureBuildSizesInfoKHR { uint32_t sType; uint64_t size; };
struct VkAccelerationStructureCreateInfoKHR { uint32_t sType; uintptr_t buffer; uint64_t size; };
struct CONTEXT { uintptr_t Rcx, Rdx, R8, R9, Rsp, Rip; };
struct MEMORY_BASIC_INFORMATION { void* BaseAddress; void* AllocationBase; size_t RegionSize; DWORD State, Protect, Type; };
constexpr DWORD MEM_COMMIT=1, MEM_IMAGE=2, PAGE_GUARD=4, PAGE_NOACCESS=8,
 PAGE_READWRITE=16, PAGE_WRITECOPY=32, PAGE_EXECUTE_READWRITE=64, PAGE_EXECUTE_WRITECOPY=128,
 PAGE_READONLY=2, PAGE_EXECUTE_READ=3;
struct Region { uintptr_t base; size_t size; DWORD protect; bool image; };
std::vector<Region> regions;
void add(void* p, size_t n, bool image=false, DWORD protect=PAGE_READWRITE) {
    regions.push_back({reinterpret_cast<uintptr_t>(p), n, protect, image});
}
void* GetModuleHandleW(void*) { return reinterpret_cast<void*>(0x1234); }
int VirtualQuery(const void* p, MEMORY_BASIC_INFORMATION* m, size_t) {
    auto x = reinterpret_cast<uintptr_t>(p);
    for (auto r : regions) if (x >= r.base && x-r.base < r.size) {
        *m = {reinterpret_cast<void*>(r.base), r.image ? GetModuleHandleW(nullptr) : nullptr,
              r.size, MEM_COMMIT, r.protect, r.image ? MEM_IMAGE : 0}; return 1;
    }
    return 0;
}
void* GetCurrentProcess() { return nullptr; }
int ReadProcessMemory(void*, const void* p, void* out, size_t n, SIZE_T* got) {
    auto x = reinterpret_cast<uintptr_t>(p);
    for (auto r : regions) if (x >= r.base && x-r.base <= r.size && n <= r.size-(x-r.base)) {
        memcpy(out, p, n); *got=n; return 1;
    }
    *got=0; return 0;
}
LONG InterlockedIncrement(volatile LONG* n) { return ++*n; }
void LogLine(const char*, ...) {}
struct { int recoverMissingASCalls = 1; } g_spvCfg;
int g_spvCfgLoaded = 2, g_rtSpoof = 1;
VkDevice g_lastDevice = reinterpret_cast<void*>(0x3456);
namespace stub { void CreateAS() {} void GetASBuildSizes() {} }
'''
        recovery_range = function_body("bool RecoveryRange(")
        recovery_read = "template <class T> " + function_body("bool RecoveryRead(")
        redirect = function_body("bool TryRedirectMissingASCall(")
        main = r'''
int main() {
    alignas(8) unsigned char code[32]{};
    uintptr_t ret = reinterpret_cast<uintptr_t>(code + 8);
    code[2] = 0xff; code[3] = 0x15; int32_t disp=8; memcpy(code+4, &disp, 4);
    uintptr_t zero=0; memcpy(code+16, &zero, sizeof(zero));
    uintptr_t stack[12]{};
    VkAccelerationStructureBuildGeometryInfoKHR input{1000150000u, 1};
    VkAccelerationStructureBuildSizesInfoKHR output{1000150020u, 0};
    uint32_t prims[1]{1};
    stack[5] = reinterpret_cast<uintptr_t>(&output);
    add(code, sizeof(code), true); add(stack, sizeof(stack));
    add(&input, sizeof(input)); add(&output, sizeof(output)); add(prims, sizeof(prims));
    CONTEXT c{reinterpret_cast<uintptr_t>(g_lastDevice), 1, reinterpret_cast<uintptr_t>(&input),
              reinterpret_cast<uintptr_t>(prims), reinterpret_cast<uintptr_t>(stack), 0};
    assert(TryRedirectMissingASCall(&c, ret));
    assert(c.Rip == reinterpret_cast<uintptr_t>(&stub::GetASBuildSizes));
    g_spvCfg.recoverMissingASCalls=0; c.Rip=0; assert(!TryRedirectMissingASCall(&c, ret));
    assert(c.Rip==0); g_spvCfg.recoverMissingASCalls=1;
    output.sType=0; assert(!TryRedirectMissingASCall(&c, ret)); output.sType=1000150020u;
    c.Rcx=0; assert(!TryRedirectMissingASCall(&c, ret)); c.Rcx=reinterpret_cast<uintptr_t>(g_lastDevice);
    code[2]=0x90; assert(!TryRedirectMissingASCall(&c, ret)); code[2]=0xff;
    uintptr_t nonzero=1; memcpy(code+16, &nonzero, sizeof(nonzero));
    assert(!TryRedirectMissingASCall(&c, ret)); memcpy(code+16, &zero, sizeof(zero));
    VkAccelerationStructureCreateInfoKHR create{1000150017u, 9, 256};
    VkAccelerationStructureKHR handle=nullptr;
    add(&create, sizeof(create)); add(&handle, sizeof(handle));
    c.Rdx=reinterpret_cast<uintptr_t>(&create); c.R8=0; c.R9=reinterpret_cast<uintptr_t>(&handle);
    assert(TryRedirectMissingASCall(&c, ret));
    assert(c.Rip == reinterpret_cast<uintptr_t>(&stub::CreateAS));
    create.size=0; assert(!TryRedirectMissingASCall(&c, ret)); create.size=256;
    c.R8=1; assert(!TryRedirectMissingASCall(&c, ret));
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(prelude + recovery_range + "\n" + recovery_read + "\n" + redirect + "\n" + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

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
