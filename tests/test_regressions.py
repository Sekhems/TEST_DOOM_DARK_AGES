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

    def test_wait_tracing_is_pass_through_and_core_aliases_use_it(self):
        table_start = SOURCE.index("const Entry kAlwaysTable[]")
        table = SOURCE[table_start:SOURCE.index("};", table_start)]
        for name in ("vkWaitSemaphores", "vkWaitSemaphoresKHR"):
            self.assertRegex(table, rf'VKSTUB\("{name}",\s*vkWaitSemaphores\)')
        self.assertRegex(table, r'VKSTUB\("vkDeviceWaitIdle",\s*vkDeviceWaitIdle\)')
        for name, signature, invocation in (
            ("vkWaitForFences", "vkWaitForFences(VkDevice device,", "real(device, fenceCount, pFences, waitAll, timeout)"),
            ("vkWaitSemaphores", "vkWaitSemaphores(VkDevice device,", "real(device, info, timeout)"),
            ("vkQueueWaitIdle", "vkQueueWaitIdle(VkQueue queue)", "real(queue)"),
            ("vkDeviceWaitIdle", "vkDeviceWaitIdle(VkDevice device)", "real(device)"),
        ):
            body = function_body(signature)
            self.assertIn(f'ApiCallTrace trace("{name}"', body)
            self.assertEqual(body.count(invocation), 1)
            self.assertIn("return r;", body)
        newer = function_body("vkQueueSubmit2(VkQueue queue,")
        self.assertEqual(newer.count("TraceSubmit2Begin("), 2)
        self.assertEqual(newer.count("InFlightEnd(slot)"), 2)
        ps = (ROOT / "build_proxy.ps1").read_text(encoding="utf-8")
        for name in ("vkDeviceWaitIdle", "vkWaitSemaphores"):
            self.assertIn("'" + name + "'", ps[ps.index("$localExports = @("):ps.index("$excludedExports =")])

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_wait_wrappers_forward_arguments_and_results(self):
        prelude = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
using VkResult = int; using VkDevice = void*; using VkQueue = void*; using VkFence = void*;
using VkSemaphore = void*; using VkBool32 = unsigned; using LONG = long;
constexpr int VK_SUCCESS=0, VK_TIMEOUT=2, VK_ERROR_INITIALIZATION_FAILED=-3;
constexpr VkFence VK_NULL_HANDLE=nullptr;
constexpr int _TRUNCATE=0;
struct VkSemaphoreWaitInfo { uint32_t flags, semaphoreCount; const VkSemaphore* pSemaphores; const uint64_t* pValues; };
using PFN_vkWaitForFences = VkResult(*)(VkDevice,uint32_t,const VkFence*,VkBool32,uint64_t);
using PFN_vkQueueWaitIdle = VkResult(*)(VkQueue);
using PFN_vkDeviceWaitIdle = VkResult(*)(VkDevice);
using PFN_vkWaitSemaphores = VkResult(*)(VkDevice,const VkSemaphoreWaitInfo*,uint64_t);
VkDevice dev=(void*)0x100; VkQueue queue=(void*)0x200; VkFence fence=(void*)0x300;
VkSemaphore sem=(void*)0x400; uint64_t value=7; VkSemaphoreWaitInfo wi{0,1,&sem,&value};
int calls=0, traced=0, activeTrace=0, sampled=0; uint64_t expectedTimeout=0;
volatile LONG g_fenceTimeouts=0, g_semaphoreTimeouts=0;
VkResult waitFence(VkDevice d,uint32_t n,const VkFence* p,VkBool32 all,uint64_t t) {
 assert(d==dev && n==1 && *p==fence && all==1 && t==expectedTimeout); calls++; return VK_TIMEOUT; }
VkResult waitQueue(VkQueue q) { assert(q==queue); calls++; return 0; }
VkResult waitDevice(VkDevice d) { assert(d==dev); calls++; return 0; }
VkResult waitSem(VkDevice d,const VkSemaphoreWaitInfo* i,uint64_t t) {
 assert(d==dev && i==&wi && t==expectedTimeout); calls++; return VK_TIMEOUT; }
template<typename T> T RealExport(const char* name) {
 if (!strcmp(name,"vkWaitForFences")) return reinterpret_cast<T>(reinterpret_cast<void*>(&waitFence));
 if (!strcmp(name,"vkQueueWaitIdle")) return reinterpret_cast<T>(reinterpret_cast<void*>(&waitQueue));
 if (!strcmp(name,"vkDeviceWaitIdle")) return reinterpret_cast<T>(reinterpret_cast<void*>(&waitDevice));
 if (!strcmp(name,"vkWaitSemaphores")) return reinterpret_cast<T>(reinterpret_cast<void*>(&waitSem));
 return nullptr;
}
struct ApiCallTrace { ApiCallTrace(const char*, const char*, bool enabled=true, VkFence=nullptr) { traced++; if(enabled) activeTrace++; } };
void SampleFencePoll(VkDevice d, VkFence f, VkResult r) { assert(d==dev && f==fence && r==VK_TIMEOUT); sampled++; }
template<typename... A> int _snprintf_s(char*,size_t,int,const char*, A...) { return 0; }
LONG InterlockedIncrement(volatile LONG* p) { return ++*p; }
void LogLine(const char*,...) {}
'''
        wrappers = "\n".join("VkResult " + function_body(signature) for signature in (
            "vkQueueWaitIdle(VkQueue queue)", "vkDeviceWaitIdle(VkDevice device)",
            "vkWaitForFences(VkDevice device,", "vkWaitSemaphores(VkDevice device,"))
        main = r'''
int main() {
 expectedTimeout=123; assert(vkWaitForFences(dev,1,&fence,1,expectedTimeout)==VK_TIMEOUT);
 assert(vkWaitSemaphores(dev,&wi,expectedTimeout)==VK_TIMEOUT);
 expectedTimeout=0; assert(vkWaitForFences(dev,1,&fence,1,0)==VK_TIMEOUT);
 assert(vkWaitSemaphores(dev,&wi,0)==VK_TIMEOUT);
 assert(vkQueueWaitIdle(queue)==VK_SUCCESS && vkDeviceWaitIdle(dev)==VK_SUCCESS);
 assert(calls==6 && traced==6 && activeTrace==4 && sampled==1);
 assert(g_fenceTimeouts==2 && g_semaphoreTimeouts==2);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(prelude + wrappers + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_fence_history_captures_waits_and_is_bounded(self):
        # Compile the actual history implementation against minimal Vulkan/Windows mocks.
        start = SOURCE.index("    struct FenceWaitInfo {")
        end = SOURCE.index("    struct InFlight {", start)
        implementation = SOURCE[start:end]
        prelude = r'''
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <string>
using ULONGLONG = uint64_t; using DWORD=unsigned; using VkFence=void*;
using VkDevice=void*; using VkQueue=void*; using VkSemaphore=void*; using VkResult=int;
using VkPipelineStageFlags=uint32_t; using SRWLOCK=int;
constexpr int SRWLOCK_INIT=0, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO=123;
struct VkBaseInStructure { int sType; const VkBaseInStructure* pNext; };
struct VkTimelineSemaphoreSubmitInfo { int sType; const VkBaseInStructure* pNext; uint32_t waitSemaphoreValueCount; const uint64_t* pWaitSemaphoreValues; uint32_t signalSemaphoreValueCount; const uint64_t* pSignalSemaphoreValues; };
struct VkSubmitInfo { const void* pNext; uint32_t commandBufferCount, waitSemaphoreCount, signalSemaphoreCount;
 const VkSemaphore* pWaitSemaphores; const VkPipelineStageFlags* pWaitDstStageMask; const VkSemaphore* pSignalSemaphores; };
int g_spvCfgLoaded=2; struct { int watchdog=1; } g_spvCfg;
void AcquireSRWLockExclusive(SRWLOCK*) {} void ReleaseSRWLockExclusive(SRWLOCK*) {}
void AcquireSRWLockShared(SRWLOCK*) {} void ReleaseSRWLockShared(SRWLOCK*) {}
ULONGLONG GetTickCount64() { return 123; } DWORD GetCurrentThreadId() { return 12; }
std::string logs;
void LogLine(const char* fmt, ...) { char buf[512]; va_list ap; va_start(ap,fmt); vsnprintf(buf,sizeof(buf),fmt,ap); va_end(ap); logs+=buf; logs+='\n'; }
'''
        main = r'''
int main() {
 auto fence=reinterpret_cast<void*>(0x1234), queue=reinterpret_cast<void*>(0x5678);
 VkSemaphore waits[2]={reinterpret_cast<void*>(0x111),reinterpret_cast<void*>(0x222)};
 VkPipelineStageFlags stages[2]={4,8}; uint64_t values[2]={7,9};
 VkSemaphore signal=reinterpret_cast<void*>(0x333); uint64_t signalValue=12;
 VkTimelineSemaphoreSubmitInfo tl{123,nullptr,2,values,1,&signalValue};
 VkSubmitInfo submits[1]={{&tl,3,2,1,waits,stages,&signal}};
 FenceTraceLife(nullptr,fence,1,1);
 FenceTraceLife(nullptr,fence,2,0);
 FenceTraceSubmit(queue,fence,1,submits);
 FenceTraceResult(queue,fence,0);
 assert(g_fenceHistorySequence == 4);
 const auto& e=g_fenceHistory[2];
 assert(e.kind==4 && e.commands==3 && e.waits==2 && e.signals==1 && e.captured==2);
 assert(e.wait[0].semaphore==0x111 && e.wait[0].hasValue && e.wait[0].value==7 && e.wait[0].stage==4);
 assert(e.wait[1].semaphore==0x222 && e.wait[1].value==9 && e.wait[1].stage==8);
 DumpFenceHistory(fence);
 assert(logs.find("entries=4")!=std::string::npos && logs.find("submit-enter")!=std::string::npos);
 assert(logs.find("value=9")!=std::string::npos);
 SemaphoreTraceSignals(queue,nullptr,1,submits,0);
 assert(g_semaphoreSignalSequence==1);
 const auto& sig=g_semaphoreSignals[0];
 assert(sig.semaphore==0x333 && sig.fence==0 && sig.hasValue && sig.value==12);
 assert(sig.waits==2 && sig.captured==2 && sig.wait[1].value==9);
 logs.clear(); DumpSemaphoreSignals((uintptr_t)signal);
 assert(logs.find("entries=1")!=std::string::npos && logs.find("not GPU completion")!=std::string::npos);
 assert(logs.find("value=12")!=std::string::npos && logs.find("wait[1]")!=std::string::npos);
 logs.clear(); DumpFenceSignalDependencies(fence);
 assert(logs.find("latest_recorded_submit_seq=3")!=std::string::npos);
 VkSemaphore many[70]; for (int i=0;i<70;++i) many[i]=signal;
 VkSubmitInfo huge{nullptr,0,0,70,nullptr,nullptr,many};
 SemaphoreTraceSignals(queue,fence,1,&huge,-4);
 assert(g_semaphoreSignalSequence==65 && g_semaphoreSignals[64].result==-4);
 VkSubmitInfo noSignal{&tl,5,2,0,waits,stages,nullptr};
 SemaphoreTraceSignals(queue,nullptr,1,&noSignal,0);
 assert(g_semaphoreSignalSequence==66 && g_semaphoreSignals[65].semaphore==0);
 assert(g_semaphoreSignals[65].waits==2 && g_semaphoreSignals[65].signals==0 && g_semaphoreSignals[65].wait[0].value==7);
 logs.clear(); DumpRecentSubmitDependencies();
 assert(logs.find("SUBMIT-RECENT entries=48")!=std::string::npos);
 assert(logs.find("cmds=5 waits=2 signals=0")!=std::string::npos && logs.find("wait[1]")!=std::string::npos);
 assert(logs.find("not GPU completion")!=std::string::npos);
 auto other=reinterpret_cast<void*>(0x999);
 for (unsigned i=0;i<8192;++i) FenceTraceLife(nullptr,other,2,0);
 logs.clear(); DumpFenceHistory(fence);
 assert(logs.find("entries=0")!=std::string::npos && logs.find("older events may be overwritten")!=std::string::npos);
 // First poll is sampled, subsequent 4095 are not; the 4097th is sampled.
 SampleFencePoll(nullptr,fence,2);
 assert(g_fencePollSequence==1 && g_fencePollSamples[0].result==2);
 for (unsigned i=0;i<4095;++i) SampleFencePoll(nullptr,fence,2);
 assert(g_fencePollSequence==1);
 SampleFencePoll(nullptr,fence,0);
 assert(g_fencePollSequence==2 && g_fencePollSamples[1].result==0);
 logs.clear(); DumpFencePollSamples(223);
 assert(logs.find("entries=1 samples=2")!=std::string::npos);
 assert(logs.find("age_ms=100")!=std::string::npos && logs.find("result=0")!=std::string::npos);
 assert(logs.find("not proof of cause")!=std::string::npos);
 auto before=g_fenceHistorySequence; g_spvCfg.watchdog=0; FenceTraceLife(nullptr,fence,1,0);
 SampleFencePoll(nullptr,fence,2); SemaphoreTraceSignals(queue,nullptr,1,submits,0);
 assert(g_fenceHistorySequence==before && g_fencePollSequence==2 && g_semaphoreSignalSequence==66);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(prelude + implementation + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_fence_hooks_include_both_submit_paths_and_lifecycle(self):
        table_start = SOURCE.index("const Entry kAlwaysTable[]")
        table = SOURCE[table_start:SOURCE.index("};", table_start)]
        ps = (ROOT / "build_proxy.ps1").read_text(encoding="utf-8")
        local = ps[ps.index("$localExports = @("):ps.index("$excludedExports =")]
        for name in ("vkCreateFence", "vkResetFences", "vkDestroyFence"):
            self.assertRegex(table, rf'VKSTUB\("{name}",\s*{name}\)')
            self.assertIn("'" + name + "'", local)
        for signature in ("vkQueueSubmit(VkQueue queue,", "vkQueueSubmit2(VkQueue queue,"):
            body = function_body(signature)
            self.assertIn("FenceTraceSubmit(", body)
            self.assertIn("FenceTraceResult(", body)
            self.assertIn("SemaphoreTraceSignals(", body)
        self.assertIn("DumpFenceSignalDependencies(e.waitFence)", function_body("DWORD WINAPI WatchdogProc("))
        wait = function_body("vkWaitForFences(VkDevice device,")
        self.assertIn("fenceCount == 1 && pFences ? pFences[0]", wait)
        self.assertIn("real(device, fenceCount, pFences, waitAll, timeout)", wait)
        self.assertIn("timeout == 0 && fenceCount == 1 && pFences", wait)
        self.assertLess(wait.index("real(device, fenceCount, pFences, waitAll, timeout)"), wait.index("SampleFencePoll("))
        watchdog = function_body("DWORD WINAPI WatchdogProc(")
        self.assertIn("DumpFencePollSamples(now)", watchdog)
        self.assertIn("DumpRecentSubmitDependencies()", watchdog)

    @unittest.skipUnless(shutil.which("g++"), "requires a C++ compiler")
    def test_fence_lifecycle_wrappers_forward_driver_results(self):
        prelude = r'''
#include <cstdint>
#include <cstring>
#include <cassert>
using VkResult=int; using VkDevice=void*; using VkFence=void*;
struct VkFenceCreateInfo { uint32_t flags; }; struct VkAllocationCallbacks {};
using PFN_vkCreateFence=VkResult(*)(VkDevice,const VkFenceCreateInfo*,const VkAllocationCallbacks*,VkFence*);
using PFN_vkResetFences=VkResult(*)(VkDevice,uint32_t,const VkFence*);
using PFN_vkDestroyFence=void(*)(VkDevice,VkFence,const VkAllocationCallbacks*);
constexpr VkResult VK_SUCCESS=0, VK_ERROR_INITIALIZATION_FAILED=-3;
VkDevice dev=(void*)0x100; VkFence handle=(void*)0x200; int calls=0, events=0, lastKind=0, nextReset=0;
VkResult create(VkDevice d,const VkFenceCreateInfo* ci,const VkAllocationCallbacks*,VkFence* f) {
 assert(d==dev && ci->flags==1); *f=handle; calls++; return 0; }
VkResult reset(VkDevice d,uint32_t n,const VkFence* f) { assert(d==dev && n==1 && *f==handle); calls++; return nextReset; }
void destroy(VkDevice d,VkFence f,const VkAllocationCallbacks*) { assert(d==dev && f==handle); calls++; }
template <typename T> T RealExport(const char* name) {
 if (!strcmp(name,"vkCreateFence")) return reinterpret_cast<T>(reinterpret_cast<void*>(&create));
 if (!strcmp(name,"vkResetFences")) return reinterpret_cast<T>(reinterpret_cast<void*>(&reset));
 if (!strcmp(name,"vkDestroyFence")) return reinterpret_cast<T>(reinterpret_cast<void*>(&destroy));
 return nullptr;
}
void FenceTraceLife(VkDevice d,VkFence f,int kind,uint32_t flags) {
 assert(d==dev && f==handle); if(kind==1) assert(flags==1); events++; lastKind=kind;
}
void LogLine(const char*,...) {}
'''
        wrappers = "\n".join(prefix + function_body(signature) for prefix, signature in (
            ("VkResult ", "vkCreateFence(VkDevice device, const VkFenceCreateInfo* pInfo,"),
            ("VkResult ", "vkResetFences(VkDevice device, uint32_t count, const VkFence* fences)"),
            ("void ", "vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* allocator)")))
        main = r'''
int main() {
 VkFence out=nullptr; VkFenceCreateInfo ci{1};
 assert(vkCreateFence(dev,&ci,nullptr,&out)==0 && out==handle && events==1 && lastKind==1);
 nextReset=-4; assert(vkResetFences(dev,1,&out)==-4 && events==1);
 nextReset=0; assert(vkResetFences(dev,1,&out)==0 && events==2 && lastKind==2);
 vkDestroyFence(dev,out,nullptr); assert(calls==4 && events==3 && lastKind==3);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = pathlib.Path(tmp) / "test.cpp"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(prelude + wrappers + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
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
