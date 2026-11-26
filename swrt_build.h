// swrt_build.h - GPU BVH builder for the software ray tracing path.
// Header-only, no OS dependency. All Vulkan entry points are injected through swrt::Fns so the proxy can use the
// REAL driver functions (never its own hooks) and the test rig can use a normal loader.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <functional>
#include "swrt_spv.h"

namespace swrt {

struct Fns {
    PFN_vkCreateShaderModule       CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule      DestroyShaderModule = nullptr;
    PFN_vkCreatePipelineLayout     CreatePipelineLayout = nullptr;
    PFN_vkCreateComputePipelines   CreateComputePipelines = nullptr;
    PFN_vkCmdBindPipeline          CmdBindPipeline = nullptr;
    PFN_vkCmdPushConstants         CmdPushConstants = nullptr;
    PFN_vkCmdDispatch              CmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier       CmdPipelineBarrier = nullptr;
};

enum Kernel { K_HEADER, K_CLEAR, K_BOUNDS, K_INST, K_MORTON, K_SORT, K_KARRAS, K_REFIT, K_COUNT };

// ---- blob / scratch layout (must match shaders/swrt.glsl) ---------------------------------------------------------
constexpr uint32_t kHeaderBytes = 256, kGeomBytes = 64, kInstBytes = 128, kNodeBytes = 64, kPrimBytes = 8;
inline uint64_t Align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
inline uint32_t Pow2Ceil(uint32_t n) { uint32_t m = 1; while (m < n) m <<= 1; return m; }

struct BlobLayout { uint32_t geomOff, instOff, primOff, nodeOff; uint64_t total; uint32_t inodes; };
inline BlobLayout ComputeBlob(bool tlas, uint32_t primCount, uint32_t geomOrInstCount) {
    BlobLayout L{};
    uint64_t off = kHeaderBytes;
    if (tlas) { L.instOff = (uint32_t)off; off = Align(off + (uint64_t)geomOrInstCount * kInstBytes, 16); }
    else      { L.geomOff = (uint32_t)off; off = Align(off + (uint64_t)geomOrInstCount * kGeomBytes, 16); }
    L.primOff = (uint32_t)off; off = Align(off + (uint64_t)primCount * kPrimBytes, 16);
    L.nodeOff = (uint32_t)off;
    L.inodes = primCount > 1 ? primCount - 1 : 0;
    off += (uint64_t)(L.inodes ? L.inodes : 1) * kNodeBytes;
    L.total = Align(off, 256);
    return L;
}

struct ScratchLayout { uint64_t bounds, keys, ids, box, pre, nodeBox, flags, parentI, parentL, total; uint32_t m; };
inline ScratchLayout ComputeScratch(uint32_t n) {
    ScratchLayout S{};
    uint32_t m = Pow2Ceil(n < 2 ? 2 : n); S.m = m;
    uint64_t o = 0;
    S.bounds = o;  o = Align(o + 32, 16);
    S.keys = o;    o = Align(o + 4ull * m, 16);
    S.ids = o;     o = Align(o + 4ull * m, 16);
    S.box = o;     o = Align(o + 32ull * (n ? n : 1), 16);
    S.pre = o;     o = Align(o + 8ull * (n ? n : 1), 16);
    S.nodeBox = o; o = Align(o + 32ull * (n ? n : 1), 16);
    S.flags = o;   o = Align(o + 4ull * (n ? n : 1), 16);
    S.parentI = o; o = Align(o + 4ull * (n ? n : 1), 16);
    S.parentL = o; o = Align(o + 4ull * (n ? n : 1), 16);
    S.total = Align(o, 256);
    return S;
}

struct GeomDesc {
    uint64_t vertexAddr = 0, indexAddr = 0, transformAddr = 0;
    uint32_t stride = 12, indexType = 0 /*0 none,1 u16,2 u32*/, vertexFmt = 0 /*0 f32x3, 1 f16, 2 snorm16*/;
    uint32_t primCount = 0, flags = 1 /*bit0 opaque*/;
};

class Builder {
public:
    // Optional crash guard around every driver call: returns false if the call crashed (and must not be used further).
    using Guard = std::function<bool(const char* what, const std::function<void()>& call)>;
    using Logger = std::function<void(const char* msg)>;
    static const char* KernelName(int k) {
        static const char* n[K_COUNT] = { "header", "clear", "bounds", "inst", "morton", "sort", "karras", "refit" };
        return n[k];
    }
    uint32_t failedMask() const { return failedMask_; }

    // Creates the 8 compute pipelines. Every kernel is compiled independently (so one bad kernel is reported by name
    // and the others are still tried). Returns false if any kernel failed.
    bool Init(VkDevice dev, const Fns& f, Guard guard = nullptr, Logger log = nullptr) {
        fn_ = f; dev_ = dev;
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 128 };
        VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
        if (fn_.CreatePipelineLayout(dev, &pl, nullptr, &layout_) != VK_SUCCESS) return false;
        const uint32_t* code[K_COUNT] = { swrt_spv_header, swrt_spv_clear, swrt_spv_bounds, swrt_spv_inst,
                                          swrt_spv_morton, swrt_spv_sort, swrt_spv_karras, swrt_spv_refit };
        const size_t   size[K_COUNT] = { sizeof(swrt_spv_header), sizeof(swrt_spv_clear), sizeof(swrt_spv_bounds), sizeof(swrt_spv_inst),
                                          sizeof(swrt_spv_morton), sizeof(swrt_spv_sort), sizeof(swrt_spv_karras), sizeof(swrt_spv_refit) };
        auto run = [&](const char* what, const std::function<void()>& c) -> bool { if (guard) return guard(what, c); c(); return true; };
        auto say = [&](const char* fmt, const char* a, const char* b2, int v) { if (!log) return; char buf[256]; snprintf(buf, sizeof(buf), fmt, a, b2, v); log(buf); };
        failedMask_ = 0;
        for (int k = 0; k < K_COUNT; ++k) {
            VkShaderModuleCreateInfo sm{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            sm.codeSize = size[k]; sm.pCode = code[k];
            VkShaderModule mod = VK_NULL_HANDLE; VkResult r = VK_ERROR_UNKNOWN;
            bool ok = run(KernelName(k), [&] { r = fn_.CreateShaderModule(dev, &sm, nullptr, &mod); });
            if (!ok || r != VK_SUCCESS) { say("SWRT: kernel '%s': vkCreateShaderModule %s (result=%d)", KernelName(k), ok ? "rejected" : "CRASHED the driver", (int)r); failedMask_ |= 1u << k; continue; }
            VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
            ci.layout = layout_;
            r = VK_ERROR_UNKNOWN;
            ok = run(KernelName(k), [&] { r = fn_.CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe_[k]); });
            if (ok) fn_.DestroyShaderModule(dev, mod, nullptr);
            if (!ok || r != VK_SUCCESS) { say("SWRT: kernel '%s': pipeline compile %s (result=%d)", KernelName(k), ok ? "rejected" : "CRASHED the driver", (int)r); failedMask_ |= 1u << k; pipe_[k] = VK_NULL_HANDLE; continue; }
            say("SWRT: kernel '%s' compiled OK%s%d", KernelName(k), "", 0);
        }
        ready_ = (failedMask_ == 0);
        return ready_;
    }
    bool Ready() const { return ready_; }
    VkPipelineLayout Layout() const { return layout_; }

    // Records the full BLAS build. dst = address of the AS blob, scratch = scratch address (size ComputeScratch(N).total).
    void RecordBlas(VkCommandBuffer cb, uint64_t dst, uint64_t scratch, const std::vector<GeomDesc>& geoms) {
        uint32_t n = 0; for (auto& g : geoms) n += g.primCount;
        BlobLayout L = ComputeBlob(false, n, (uint32_t)geoms.size());
        ScratchLayout S = ComputeScratch(n);
        BarrierAll(cb, true);
        Header(cb, dst, scratch, S, L, 1, n, (uint32_t)geoms.size());
        if (n == 0) { BarrierAll(cb, false); return; }
        Clear(cb, scratch + S.flags, L.inodes);
        Barrier(cb);
        uint32_t base = 0, gi = 0;
        for (auto& g : geoms) {
            if (g.primCount) {
                struct { uint64_t vb, ib, xa, blob, box, pre, bounds; uint32_t stride, itype, vfmt, primCount, primBase, geomIdx, gflags, geomOff; } pc
                    { g.vertexAddr, g.indexAddr, g.transformAddr, dst, scratch + S.box, scratch + S.pre, scratch + S.bounds,
                      g.stride, g.indexType, g.vertexFmt, g.primCount, base, gi, g.flags, L.geomOff };
                Dispatch(cb, K_BOUNDS, &pc, sizeof(pc), (g.primCount + 63) / 64);
            }
            base += g.primCount; ++gi;
        }
        Barrier(cb);
        Common(cb, dst, scratch, S, L, n, false);
        BarrierAll(cb, false);
    }

    void RecordTlas(VkCommandBuffer cb, uint64_t dst, uint64_t scratch, uint64_t instAddr, uint32_t count, bool ptrMode) {
        BlobLayout L = ComputeBlob(true, count, count);
        ScratchLayout S = ComputeScratch(count);
        BarrierAll(cb, true);
        Header(cb, dst, scratch, S, L, 2, count, count);
        if (count == 0) { BarrierAll(cb, false); return; }
        Clear(cb, scratch + S.flags, L.inodes);
        Barrier(cb);
        struct { uint64_t inst, blob, box, bounds; uint32_t count, ptrMode, instOff; } pc
            { instAddr, dst, scratch + S.box, scratch + S.bounds, count, ptrMode ? 1u : 0u, L.instOff };
        Dispatch(cb, K_INST, &pc, sizeof(pc), (count + 63) / 64);
        Barrier(cb);
        Common(cb, dst, scratch, S, L, count, true);
        BarrierAll(cb, false);
    }

private:
    void Header(VkCommandBuffer cb, uint64_t dst, uint64_t scratch, const ScratchLayout& S, const BlobLayout& L,
                uint32_t type, uint32_t prims, uint32_t gcount) {
        struct { uint64_t blob, bounds, flags, parentI; uint32_t type, prims, inodes, nodeOff, primOff, geomOff, instOff, geomCnt; } pc
            { dst, scratch + S.bounds, scratch + S.flags, scratch + S.parentI, type, prims, L.inodes, L.nodeOff, L.primOff,
              L.geomOff, L.instOff, gcount };
        Dispatch(cb, K_HEADER, &pc, sizeof(pc), 1);
        Barrier(cb);
    }
    void Clear(VkCommandBuffer cb, uint64_t flags, uint32_t count) {
        if (!count) return;
        struct { uint64_t flags; uint32_t count; } pc{ flags, count };
        Dispatch(cb, K_CLEAR, &pc, sizeof(pc), (count + 63) / 64);
    }
    // morton -> bitonic sort -> karras -> refit
    void Common(VkCommandBuffer cb, uint64_t dst, uint64_t scratch, const ScratchLayout& S, const BlobLayout& L, uint32_t n, bool tlas) {
        uint32_t m = S.m;
        { struct { uint64_t box, bounds, keys, ids; uint32_t n, m; } pc{ scratch + S.box, scratch + S.bounds, scratch + S.keys, scratch + S.ids, n, m };
          Dispatch(cb, K_MORTON, &pc, sizeof(pc), (m + 63) / 64); }
        Barrier(cb);
        for (uint32_t k = 2; k <= m; k <<= 1)
            for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                struct { uint64_t keys, ids; uint32_t m, k, j; } pc{ scratch + S.keys, scratch + S.ids, m, k, j };
                Dispatch(cb, K_SORT, &pc, sizeof(pc), (m + 63) / 64);
                Barrier(cb);
            }
        if (n > 1) {
            struct { uint64_t keys, nodes, parentI, parentL; int32_t n; } pc{ scratch + S.keys, dst + L.nodeOff, scratch + S.parentI, scratch + S.parentL, (int32_t)n };
            Dispatch(cb, K_KARRAS, &pc, sizeof(pc), (n + 63) / 64);
            Barrier(cb);
        }
        struct { uint64_t ids, pre, box, nodes, nodeBox, flags, parentI, parentL, blob; uint32_t n, isTlas, primOff; } pc
            { scratch + S.ids, scratch + S.pre, scratch + S.box, dst + L.nodeOff, scratch + S.nodeBox, scratch + S.flags,
              scratch + S.parentI, scratch + S.parentL, dst, n, tlas ? 1u : 0u, L.primOff };
        Dispatch(cb, K_REFIT, &pc, sizeof(pc), (n + 63) / 64);
    }
    void Dispatch(VkCommandBuffer cb, Kernel k, const void* pc, size_t size, uint32_t groups) {
        fn_.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_[k]);
        fn_.CmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)size, pc);
        fn_.CmdDispatch(cb, groups ? groups : 1, 1, 1);
    }
    void Barrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        fn_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void BarrierAll(VkCommandBuffer cb, bool before) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT };
        if (before) fn_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        else        fn_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    Fns fn_{}; VkDevice dev_ = VK_NULL_HANDLE; VkPipelineLayout layout_ = VK_NULL_HANDLE; VkPipeline pipe_[K_COUNT] = {}; bool ready_ = false; uint32_t failedMask_ = 0;
};

} // namespace swrt
