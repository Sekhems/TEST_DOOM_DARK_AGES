"""Offline admission and optimizer-tool tests, not driver or Windows tests."""
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "optimize_heavy_compute.py"
spec = importlib.util.spec_from_file_location("optimize_heavy_compute", SCRIPT)
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


def shader(nops=1, name=b"main", model=5, version=0x10500):
    # Structurally valid for these checks, not a complete executable module.
    name = name + b"\0"
    name += b"\0" * ((-len(name)) % 4)
    words = [0x07230203, version, 0, 10, 0, ((3 + len(name)//4) << 16) | 15, model, 1]
    words.extend(struct.unpack("<%dI" % (len(name)//4), name))
    words.extend([1 << 16] * nops)
    return struct.pack("<%dI" % len(words), *words)


class OptimizedComputeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_actual_cpp_admission(self):
        cpp = r'''
#include "optimized_compute.h"
#include <cassert>
int main() {
    // Compute entry "main", plus one OpNop in baseline.
    uint32_t src[]={0x07230203,0x10500,0,10,0,0x0005000f,5,1,0x6e69616d,0,0x10000};
    uint32_t dst[]={0x07230203,0x10500,0,10,0,0x0005000f,5,1,0x6e69616d,0};
    assert(optcompute::Accept(src,11,dst,10,11));
    assert(!optcompute::Accept(src,11,dst,10,10));
    assert(!optcompute::Accept(src,11,src,11,99));
    assert(!optcompute::Accept(src,11,dst,4,99));
    dst[1]=0x10600; assert(!optcompute::Accept(src,11,dst,10,99)); dst[1]=0x10500;
    dst[6]=4; assert(!optcompute::Accept(src,11,dst,10,99)); dst[6]=5;
    dst[8]=0x6e69616e; assert(!optcompute::Accept(src,11,dst,10,99)); dst[8]=0x6e69616d;
    dst[5]=0x0000000f; assert(!optcompute::Accept(src,11,dst,10,99));
    assert(optcompute::Hash(src,11)==0xHASHull);
}
'''
        data = shader(1)
        cpp = cpp.replace("0xHASH", hex(opt.fnv1a64(data)))
        with tempfile.TemporaryDirectory() as d:
            source, exe = Path(d)/"test.cpp", Path(d)/"test"
            source.write_text(cpp)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_heavy_pipeline_gate_real_block_with_mocks(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("    if (g_spvCfg.heavyComputeWords > 0) {", source.index("vkCreateComputePipelines(VkDevice device, VkPipelineCache"))
        end = source.index("    // Fast-fail heavy compute", start)
        block = source[start:end].replace("vkCreateComputePipelines(device,", "run(device,")
        prelude = r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
using LONG=long; using VkPipeline=void*; using VkDevice=void*; using VkPipelineCache=void*;
using VkAllocationCallbacks=int; using VkResult=int;
constexpr VkResult VK_SUCCESS=0; constexpr VkPipeline VK_NULL_HANDLE=nullptr;
struct Stage { int module; }; struct VkComputePipelineCreateInfo { Stage stage; };
struct SpvInfo { size_t origWords=0; bool optimizedCompute=false; uint64_t optimizedComputeSourceHash=0; std::shared_ptr<std::vector<uint32_t>> code; };
SpvInfo table[5]; int dumped=0, called=0;
struct Config { int heavyComputeWords=61000; } g_spvCfg;
bool GetSpvInfo(int n, SpvInfo& out) { out=table[n]; return true; }
void DumpHeavyComputeOnce(const SpvInfo&) { dumped++; }
LONG InterlockedIncrement(volatile LONG* p) { return ++*p; }
void LogLine(const char*,...) {}
template<class T> T MakeFake() { return reinterpret_cast<T>(0xfeed); }
int run(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
        const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
'''
        main = r'''
    called++;
    for (uint32_t i=0;i<createInfoCount;++i) pPipelines[i]=reinterpret_cast<void*>(0x111);
    return VK_SUCCESS;
}
int main() {
    table[0].origWords=60000; table[0].code=std::make_shared<std::vector<uint32_t>>(60000);
    table[1].origWords=61247; table[1].code=std::make_shared<std::vector<uint32_t>>(61247);
    table[2].origWords=61247; table[2].optimizedCompute=true; table[2].code=std::make_shared<std::vector<uint32_t>>(59000);
    table[3].origWords=61247; table[3].optimizedCompute=true; table[3].code=std::make_shared<std::vector<uint32_t>>(61000);
    VkComputePipelineCreateInfo ci[4]={{{0}},{{1}},{{2}},{{3}}}; VkPipeline out[4]{};
    assert(run(nullptr,nullptr,4,ci,nullptr,out)==0);
    assert(out[0]==(void*)0x111 && out[1]==(void*)0xfeed && out[2]==(void*)0x111 && out[3]==(void*)0xfeed);
    assert(dumped==2 && called==1);
}
'''
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d)/"gate.cpp", Path(d)/"gate"
            cpp.write_text(prelude + block + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_selected_module_and_failed_candidate_retry_real_block(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("    // Up to seven explicitly selected hashes")
        end = source.index("    // The old driver's SPIR-V front end", start)
        block = source[start:end]
        prelude = r'''
#include "optimized_compute.h"
#include "graphics_cap49.h"
#include <cassert>
#include <vector>
using VkResult=int; constexpr int VK_SUCCESS=0, VK_ERROR_UNKNOWN=-13;
void* g_lastDevice=nullptr; volatile long g_graphicsCap49Feature=0;
long InterlockedCompareExchange(volatile long* p, long, long) { return *p; }
struct Config { int heavyComputeWords=11; uint64_t optimizedComputeHash=0, optimizedComputeHash2=0, optimizedComputeHash3=0, optimizedComputeHash4=0, optimizedComputeHash5=0, optimizedComputeHash6=0, optimizedComputeHash7=0, graphicsCap49VertexHash=0, graphicsCap49FragmentHash=0; } g_spvCfg;
struct Info { int model=5; size_t origWords=12; unsigned version=0x10500; std::string caps; } si;
struct CreateInfo { const uint32_t* pCode=nullptr; size_t codeSize=0; };
uint32_t src[]={0x07230203,0x10500,0,10,0,0x0005000f,5,1,0x6e69616d,0,0x10000};
uint32_t dst[]={0x07230203,0x10500,0,10,0,0x0005000f,5,1,0x6e69616d,0};
int calls=0; bool failFirst=false, candidatePresent=true; uint64_t lastKey=0;
bool ReadOptimizedCompute(uint64_t key, std::vector<uint32_t>& out) { lastKey=key; if (!candidatePresent) return false; out.assign(dst,dst+10); return true; }
VkResult real(void*, CreateInfo* ci, void*, void*) {
 calls++; if (failFirst && calls==1) { assert(ci->pCode!=src); return -1; } return VK_SUCCESS;
}
template<class F> bool DriverCall(F f) { f(); return true; }
void LogLine(const char*,...) {} void Crumb(const char*,...) {} void SetJob(const char*,...) {}
void DumpSpirv(const char*,long,const uint32_t*,size_t) {}
void run(int slot, bool fail, bool present) {
 const uint32_t* useCode=src; size_t useWords=11; long n=1; CreateInfo ci{}; CreateInfo info{};
 info.codeSize=44; CreateInfo* pInfo=&info; void* device=nullptr; void* pAllocator=nullptr; void* pShaderModule=nullptr;
 calls=0; lastKey=0; failFirst=fail; candidatePresent=present;
 const uint64_t key=optcompute::Hash(src,11);
 g_spvCfg.optimizedComputeHash=(slot==1 || slot==3 || slot==5) ? key : (slot==4 ? key+1 : 0);
 g_spvCfg.optimizedComputeHash2=(slot==2 || slot==4 || slot==5) ? key : (slot==3 ? key+1 : 0);
 g_spvCfg.optimizedComputeHash3=(slot==6 || slot==7 || slot==8) ? key : (slot==9 ? key+1 : 0);
 if (slot==7) { g_spvCfg.optimizedComputeHash=key+1; g_spvCfg.optimizedComputeHash2=key+2; }
 if (slot==8) { g_spvCfg.optimizedComputeHash=key; g_spvCfg.optimizedComputeHash2=key; }
 g_spvCfg.optimizedComputeHash4=(slot==10 || slot==11 || slot==12) ? key : (slot==13 ? key+1 : 0);
 if (slot==11) { g_spvCfg.optimizedComputeHash=key+1; g_spvCfg.optimizedComputeHash2=key+2; g_spvCfg.optimizedComputeHash3=key+3; }
 if (slot==12) { g_spvCfg.optimizedComputeHash=key; g_spvCfg.optimizedComputeHash2=key; g_spvCfg.optimizedComputeHash3=key; }
 g_spvCfg.optimizedComputeHash5=(slot==14 || slot==15 || slot==16) ? key : (slot==17 ? key+1 : 0);
 if (slot==15) { g_spvCfg.optimizedComputeHash=key+1; g_spvCfg.optimizedComputeHash2=key+2; g_spvCfg.optimizedComputeHash3=key+3; g_spvCfg.optimizedComputeHash4=key+4; }
 if (slot==16) { g_spvCfg.optimizedComputeHash=key; g_spvCfg.optimizedComputeHash2=key; g_spvCfg.optimizedComputeHash3=key; g_spvCfg.optimizedComputeHash4=key; }
 g_spvCfg.optimizedComputeHash6=(slot==18 || slot==19 || slot==20) ? key : (slot==21 ? key+1 : 0);
 if (slot==19) { g_spvCfg.optimizedComputeHash=key+1; g_spvCfg.optimizedComputeHash2=key+2; g_spvCfg.optimizedComputeHash3=key+3; g_spvCfg.optimizedComputeHash4=key+4; g_spvCfg.optimizedComputeHash5=key+5; }
 if (slot==20) { g_spvCfg.optimizedComputeHash=key; g_spvCfg.optimizedComputeHash2=key; g_spvCfg.optimizedComputeHash3=key; g_spvCfg.optimizedComputeHash4=key; g_spvCfg.optimizedComputeHash5=key; }
 g_spvCfg.optimizedComputeHash7=(slot==22 || slot==23 || slot==24) ? key : (slot==25 ? key+1 : 0);
 if (slot==23) { g_spvCfg.optimizedComputeHash=key+1; g_spvCfg.optimizedComputeHash2=key+2; g_spvCfg.optimizedComputeHash3=key+3; g_spvCfg.optimizedComputeHash4=key+4; g_spvCfg.optimizedComputeHash5=key+5; g_spvCfg.optimizedComputeHash6=key+6; }
 if (slot==24) { g_spvCfg.optimizedComputeHash=key; g_spvCfg.optimizedComputeHash2=key; g_spvCfg.optimizedComputeHash3=key; g_spvCfg.optimizedComputeHash4=key; g_spvCfg.optimizedComputeHash5=key; g_spvCfg.optimizedComputeHash6=key; }
'''
        main = r'''
 assert(r==VK_SUCCESS);
 assert(lastKey == ((slot && slot!=9 && slot!=13 && slot!=17 && slot!=21 && slot!=25) ? optcompute::Hash(src,11) : 0));
 if (!slot || slot==9 || slot==13 || slot==17 || slot==21 || slot==25 || !present) { assert(!optimizedAttempt && useCode==src && calls==1); }
 else if (fail) { assert(!optimizedAttempt && useCode==src && calls==2); }
 else { assert(optimizedAttempt && useCode!=src && useWords==10 && calls==1); }
}
int main() { run(0,false,true); run(1,false,false); run(1,false,true); run(1,true,true); run(2,false,false); run(2,false,true); run(2,true,true); run(3,false,true); run(4,false,true); run(5,false,true); run(6,false,false); run(6,false,true); run(6,true,true); run(7,false,true); run(8,false,true); run(9,false,true); run(10,false,false); run(10,false,true); run(10,true,true); run(11,false,true); run(12,false,true); run(13,false,true); run(14,false,false); run(14,false,true); run(14,true,true); run(15,false,true); run(16,false,true); run(17,false,true); run(18,false,false); run(18,false,true); run(18,true,true); run(19,false,true); run(20,false,true); run(21,false,true); run(22,false,false); run(22,false,true); run(22,true,true); run(23,false,true); run(24,false,true); run(25,false,true); }
'''
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d)/"module.cpp", Path(d)/"module"
            cpp.write_text(prelude + block + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_entrypoint_and_hash_validation(self):
        a = shader(2)
        self.assertEqual(opt.entries(a), (0x10500, [b"main"]))
        self.assertEqual(opt.fnv1a64(b""), 0xCBF29CE484222325)
        for bad in (a[:-1], shader(model=4), shader(name=b""), a[:20] + b"\x00\x00\x00\x00"):
            with self.assertRaises(ValueError):
                opt.entries(bad)

    def test_explicit_candidate_runs_validator_twice_and_does_not_overwrite(self):
        if os.name == "nt":
            self.skipTest("uses POSIX executable mocks")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "input.spv"
            source.write_bytes(shader(2))
            calls = d / "calls"
            fake = d / "tool"
            fake.write_text("#!/usr/bin/env python3\n" +
                "import os, pathlib, sys\n" +
                "with open(os.environ['CALLS'],'a') as f: f.write(' '.join(sys.argv[1:])+'\\n')\n" +
                "if '-o' in sys.argv:\n" +
                " src=pathlib.Path(sys.argv[sys.argv.index('-o')-1]); out=pathlib.Path(sys.argv[sys.argv.index('-o')+1]); out.write_bytes(src.read_bytes()[:-4])\n")
            fake.chmod(0o755)
            env = dict(os.environ, CALLS=str(calls), PYTHONDONTWRITEBYTECODE="1")
            cmd = [sys.executable, str(SCRIPT), str(source), "--limit", "12", "--spirv-opt", str(fake), "--spirv-val", str(fake)]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            dest = d / "spv_opt" / (f"spv_{opt.fnv1a64(source.read_bytes()):016x}.spv")
            self.assertEqual(dest.read_bytes(), shader(1))
            self.assertEqual(len(calls.read_text().splitlines()), 3)
            self.assertIn("--preserve-bindings", calls.read_text())
            self.assertIn("--preserve-interface", calls.read_text())
            self.assertNotIn("--scalar-block-layout", calls.read_text())
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(dest.read_bytes(), shader(1))
            dest.unlink()
            # The strict threshold prevents installation, even when validator succeeds.
            cmd[cmd.index("12")] = "11"
            self.assertNotEqual(subprocess.run(cmd, env=env, capture_output=True).returncode, 0)
            self.assertFalse(dest.exists())
            # Explicit scalar layout affects both validators and the optimizer, never by default.
            calls.unlink()
            cmd[cmd.index("11")] = "12"
            result = subprocess.run(cmd + ["--scalar-block-layout"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("requires scalarBlockLayout to be enabled", result.stdout)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(all("--scalar-block-layout" in line for line in lines))

    def test_scalar_layout_log_reports_sanitized_device_features(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("// Record the value actually passed to the driver")
        end = source.index("VkResult r = real(pd, &ci, pAllocator, pDevice)", start)
        block = source[start:end]
        self.assertIn("ci.pNext", block)
        self.assertIn("f->scalarBlockLayout", block)
        self.assertIn("sanitized Vulkan12Features.scalarBlockLayout=%u", block)

    def test_selected_pipeline_outcome_is_reported_after_driver_call(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("vkCreateComputePipelines(VkDevice device, VkPipelineCache")
        end = source.index("// Pass-through wrappers", start)
        block = source[start:end]
        self.assertIn("selected pipeline blocked by quarantine", block)
        self.assertIn("selected pipeline blocked by toxic signature", block)
        driver = block.index("cCallOk = DriverCallTimed")
        outcome = block.index("selected pipeline driver result=%d")
        fake = block.index('vkCreateComputePipelines FAILED: result=%d')
        self.assertLess(driver, outcome)
        self.assertLess(outcome, fake)
        self.assertIn("!si.optimizedCompute", block[driver:outcome])
        self.assertIn("GPU execution not established", block[outcome:fake])
        self.assertIn("optOutcomeCount) <= 40", block[driver:outcome])

    def test_proxy_candidate_is_opt_in_and_guard_stays_active(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        self.assertIn("g_spvCfg.optimizedComputeHash && h == g_spvCfg.optimizedComputeHash", source)
        self.assertIn("g_spvCfg.optimizedComputeHash2 && h == g_spvCfg.optimizedComputeHash2", source)
        self.assertIn("g_spvCfg.optimizedComputeHash3 && h == g_spvCfg.optimizedComputeHash3", source)
        self.assertIn("g_spvCfg.optimizedComputeHash4 && h == g_spvCfg.optimizedComputeHash4", source)
        self.assertIn("g_spvCfg.optimizedComputeHash5 && h == g_spvCfg.optimizedComputeHash5", source)
        self.assertIn("g_spvCfg.optimizedComputeHash6 && h == g_spvCfg.optimizedComputeHash6", source)
        self.assertIn("g_spvCfg.optimizedComputeHash7 && h == g_spvCfg.optimizedComputeHash7", source)
        self.assertIn("ReadOptimizedCompute(selectedComputeHash, optimizedCode)", source)
        self.assertIn("si.optimizedComputeSourceHash = si.optimizedCompute ? selectedComputeHash : 0", source)
        self.assertIn("si.model == 5 && g_spvCfg.heavyComputeWords > 0", source)
        self.assertIn("optcompute::Accept(useCode, useWords", source)
        self.assertIn("si.optimizedCompute = optimizedAttempt && variantCode.empty()", source)
        self.assertIn("hsi.optimizedCompute && hsi.code", source)
        self.assertIn("candidate shader module failed; retrying baseline with original guard", source)
        self.assertIn("DumpHeavyComputeOnce(hsi)", source)
        self.assertIn("CREATE_NEW", source[source.index("void DumpHeavyComputeOnce"):source.index("bool ReadOptimizedCompute")])


if __name__ == "__main__":
    unittest.main()
