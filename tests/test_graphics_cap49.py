"""Bounded graphics capability experiment tests; not Windows or GPU tests."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class GraphicsCapTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_real_helper_on_user_modules_and_negative_gates(self):
        fixture = os.environ.get("GRAPHICS_CAP49_FIXTURE")
        archive = Path(fixture) if fixture else ROOT / "spv_gpfail10_stage1_0.zip"
        if not archive.exists():
            self.skipTest("user archive not available")
        cpp = r'''
#include "graphics_cap49.h"
#include <cassert>
#include <fstream>
#include <vector>
std::vector<uint32_t> read(const char* p) {
 std::ifstream f(p, std::ios::binary|std::ios::ate); auto n=f.tellg(); assert(n>0 && n%4==0);
 std::vector<uint32_t> v(static_cast<size_t>(n)/4); f.seekg(0); f.read((char*)v.data(),n); assert(f); return v;
}
int main(int argc, char** argv) {
 assert(argc==5); auto v=read(argv[1]), f=read(argv[2]); std::vector<uint32_t> out;
 auto vh=optcompute::Hash(v.data(),v.size()); auto fh=optcompute::Hash(f.data(),f.size());
 assert(vh==0x4ff26a9ff4ef23adull && fh==0xe4f5b7cee57c69e4ull);
 assert(!graphicscap49::AddForSelectedModule(v.data(),v.size(),0,0,true,out));
 assert(!graphicscap49::AddForSelectedModule(v.data(),v.size(),0,vh,false,out));
 assert(!graphicscap49::AddForSelectedModule(v.data(),v.size(),4,vh,true,out));
 assert(!graphicscap49::AddForSelectedModule(v.data(),v.size(),5,vh,true,out));
 assert(!graphicscap49::AddForSelectedModule(v.data(),v.size(),0,fh,true,out));
 assert(graphicscap49::AddForSelectedModule(v.data(),v.size(),0,vh,true,out));
 assert(out.size()==v.size()+2 && out[5]==((2u<<16)|17u) && out[6]==49);
 assert(!graphicscap49::AddForSelectedModule(out.data(),out.size(),0,optcompute::Hash(out.data(),out.size()),true,v));
 { std::ofstream o(argv[3],std::ios::binary); o.write((const char*)out.data(),out.size()*4); }
 assert(graphicscap49::AddForSelectedModule(f.data(),f.size(),4,fh,true,out));
 { std::ofstream o(argv[4],std::ios::binary); o.write((const char*)out.data(),out.size()*4); }
}
'''
        with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(archive) as z:
            d = Path(d)
            v, f = d / "vert.spv", d / "frag.spv"
            v.write_bytes(z.read("spv_gpfail10_stage1_0.spv"))
            f.write_bytes(z.read("spv_gpfail10_stage10_0.spv"))
            source, exe = d / "cap.cpp", d / "cap"
            source.write_text(cpp)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(exe)], check=True)
            subprocess.run([str(exe), str(v), str(f), str(d / "vout.spv"), str(d / "fout.spv")], check=True)
            validator = shutil.which("spirv-val")
            if validator:
                for p in (d / "vout.spv", d / "fout.spv"):
                    subprocess.run([validator, "--target-env", "vulkan1.2", "--scalar-block-layout", str(p)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_actual_module_selection_and_failure_fallback_with_mocks(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("    // One explicitly selected vertex and fragment module only.")
        end = source.index("    // The old driver's SPIR-V front end", start)
        block = source[start:end]
        prelude = r"""
#include "graphics_cap49.h"
#include <cassert>
#include <string>
#include <vector>
using VkResult=int; constexpr int VK_SUCCESS=0, VK_ERROR_UNKNOWN=-13;
struct Config { uint64_t graphicsCap49VertexHash=0, graphicsCap49FragmentHash=0, optimizedComputeHash=0; } g_spvCfg;
struct Info { uint32_t model=0, version=0; std::string caps; } si;
void* g_lastDevice=(void*)1; volatile long g_graphicsCap49Feature=1;
long InterlockedCompareExchange(volatile long* p,long,long) { return *p; }
void LogLine(const char*,...) {} void Crumb(const char*,...) {} void SetJob(const char*,...) {}
void DumpSpirv(const char*,long,const uint32_t*,size_t) {}
template<class F> bool DriverCall(F f) { f(); return true; }
struct CreateInfo { const uint32_t* pCode=nullptr; size_t codeSize=0; };
uint32_t src[]={0x07230203,0x10500,0,10,0,(2u<<16)|17,1,(5u<<16)|15,0,1,0x6e69616d,0};
int calls=0; bool fail=false; const uint32_t* first=nullptr;
VkResult real(void*,CreateInfo* ci,void*,void*) { calls++; if(calls==1) first=ci->pCode; return fail && calls==1 ? -1 : 0; }
void run(bool selected,bool enabled,bool otherDevice,bool failFirst) {
 const uint32_t* useCode=src; size_t useWords=12; const uint32_t* baselineCode=src; size_t baselineWords=12;
 bool optimizedAttempt=false; CreateInfo ci{}, info{}; info.codeSize=48; auto* pInfo=&info;
 void* device=otherDevice ? (void*)2 : g_lastDevice; void* pAllocator=nullptr; void* pShaderModule=nullptr;
 long n=1; calls=0; fail=failFirst; first=nullptr; g_graphicsCap49Feature=enabled;
 g_spvCfg.graphicsCap49VertexHash=selected ? optcompute::Hash(src,12) : 0;
"""
        main = r"""
 assert(r==VK_SUCCESS);
 if (!selected || !enabled || otherDevice) { assert(!graphicsAttempt && first==src && calls==1); }
 else if (failFirst) { assert(!graphicsAttempt && first!=src && useCode==src && calls==2); }
 else { assert(graphicsAttempt && first!=src && useWords==14 && calls==1); }
}
int main() { run(false,true,false,false); run(true,false,false,false); run(true,true,true,false);
 run(true,true,false,false); run(true,true,false,true); }
"""
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d) / "mock.cpp", Path(d) / "mock"
            cpp.write_text(prelude + block + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_actual_sanitized_device_feature_gate_with_mocks(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        start = source.index("    // Examine the actual sanitized core features")
        end = source.index("    for (size_t i = unlinked.size()", start)
        block = source[start:end]
        prelude = r"""
#include <cassert>
using VkResult=int; constexpr int VK_SUCCESS=0, VK_FALSE=0, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2=100;
struct Features { int shaderStorageImageExtendedFormats=0; };
struct VkBaseInStructure { int sType; const VkBaseInStructure* pNext; };
struct VkPhysicalDeviceFeatures2 { int sType; const VkBaseInStructure* pNext; Features features; };
struct DeviceInfo { const Features* pEnabledFeatures; const void* pNext; };
volatile long g_graphicsCap49Feature=0;
long InterlockedExchange(volatile long* p,long v) { long old=*p; *p=v; return old; }
void LogLine(const char*,...) {}
void run(DeviceInfo ci, VkResult r) {
"""
        main = r"""
}
int main() {
 Features yes{1}, no{0}; VkPhysicalDeviceFeatures2 f{100,nullptr,yes};
 g_graphicsCap49Feature=0; run({nullptr,&f},0); assert(g_graphicsCap49Feature==1);
 f.features=no; run({nullptr,&f},0); assert(g_graphicsCap49Feature==0);
 run({&yes,nullptr},0); assert(g_graphicsCap49Feature==1);
 run({&yes,&f},0); assert(g_graphicsCap49Feature==0);
 run({nullptr,nullptr},0); assert(g_graphicsCap49Feature==0);
 g_graphicsCap49Feature=1; run({&no,nullptr},-1); assert(g_graphicsCap49Feature==1);
}
"""
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d) / "device.cpp", Path(d) / "device"
            cpp.write_text(prelude + block + main)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_proxy_gates_cap_on_successful_sanitized_device(self):
        source = (ROOT / "dllmain.cpp").read_text(encoding="utf-8")
        self.assertIn('readHash(L"GraphicsCap49VertexHash")', source)
        self.assertIn('readHash(L"GraphicsCap49FragmentHash")', source)
        self.assertIn("device == g_lastDevice && InterlockedCompareExchange(&g_graphicsCap49Feature, 0, 0)", source)
        self.assertIn("if (ci.pEnabledFeatures && hasFeature2) cap49Enabled = false", source)
        self.assertIn("if (r == VK_SUCCESS) InterlockedExchange(&g_graphicsCap49Feature", source)
        self.assertIn("GRAPHICS-CAP49: shader module failed; retrying baseline", source)
        self.assertIn("GRAPHICS-CAP49: pipeline result=%d", source)


if __name__ == "__main__":
    unittest.main()
