// swrt_proxy.h - software ray tracing, proxy side. Included from dllmain.cpp (inside the anonymous namespace,
// right before `namespace stub`). Replaces the empty acceleration-structure stubs:
//   * every VkAccelerationStructureKHR is a record backed by the game's own VkBuffer (+offset), the device address
//     of an AS is the real buffer device address, so TLAS instances that reference BLAS addresses keep working
//   * vkCmdBuildAccelerationStructuresKHR is turned into GPU compute work recorded in the game's command buffer
//     (BVH build, see swrt_build.h); the game's compute pipeline + push constants are restored afterwards
//   * acceleration-structure query pools (compacted size) are emulated on the CPU
#pragma once
#include "swrt_build.h"

namespace swpx {

constexpr VkQueryType kQT_AS_COMPACTED = (VkQueryType)1000150000;
constexpr VkQueryType kQT_AS_SERIAL    = (VkQueryType)1000150001;
constexpr VkQueryType kQT_AS_SERIAL_BL = (VkQueryType)1000150002;
constexpr VkQueryType kQT_AS_SIZE      = (VkQueryType)1000386000;
inline bool IsAsQueryType(VkQueryType t) { return t == kQT_AS_COMPACTED || t == kQT_AS_SERIAL || t == kQT_AS_SERIAL_BL || t == kQT_AS_SIZE; }

struct AsRec {
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0, size = 0;
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
    uint64_t addr = 0;      // device address of the blob
    uint64_t used = 0;      // bytes actually used by the last build (what a "compacted size" query returns)
};
struct QPool { VkQueryType type; std::vector<uint64_t> vals; };
struct PushShadow { uint8_t d[128]; uint32_t end = 0; };

struct State {
    SRWLOCK lk = SRWLOCK_INIT;
    std::unordered_map<uint64_t, AsRec*> as;
    std::unordered_map<uint64_t, QPool*> qp;
    std::unordered_map<VkCommandBuffer, PushShadow> push;
    swrt::Builder builder;
    bool inited = false, failed = false;
    VkPipelineLayout restoreLayout = VK_NULL_HANDLE;
    PFN_vkGetBufferDeviceAddress getAddr = nullptr;
    PFN_vkCmdBindPipeline cmdBind = nullptr;
    PFN_vkCmdPushConstants cmdPush = nullptr;
    PFN_vkCmdPipelineBarrier cmdBarrier = nullptr;
    PFN_vkCmdCopyBuffer cmdCopy = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdate = nullptr;
    long builds = 0, tlasBuilds = 0, skipped = 0;
};
inline State& S() { static State s; return s; }

template<class T> inline T RF(VkDevice d, const char* n) { return reinterpret_cast<T>(g_realGdpa(d, n)); }

// Lazily creates the builder pipelines on the real driver. Returns false (permanently) if that fails.
inline bool EnsureInit(VkDevice dev) {
    State& s = S();
    if (!g_spvCfg.softwareRT) return false;          // master switch (VulkanProxy.ini: SoftwareRT)
    if (s.inited) return true;
    if (s.failed || !dev || !g_realGdpa) return false;
    AcquireSRWLockExclusive(&s.lk);
    if (!s.inited && !s.failed) {
        swrt::Fns f;
        f.CreateShaderModule     = RF<PFN_vkCreateShaderModule>(dev, "vkCreateShaderModule");
        f.DestroyShaderModule    = RF<PFN_vkDestroyShaderModule>(dev, "vkDestroyShaderModule");
        f.CreatePipelineLayout   = RF<PFN_vkCreatePipelineLayout>(dev, "vkCreatePipelineLayout");
        f.CreateComputePipelines = RF<PFN_vkCreateComputePipelines>(dev, "vkCreateComputePipelines");
        f.CmdBindPipeline        = RF<PFN_vkCmdBindPipeline>(dev, "vkCmdBindPipeline");
        f.CmdPushConstants       = RF<PFN_vkCmdPushConstants>(dev, "vkCmdPushConstants");
        f.CmdDispatch            = RF<PFN_vkCmdDispatch>(dev, "vkCmdDispatch");
        f.CmdPipelineBarrier     = RF<PFN_vkCmdPipelineBarrier>(dev, "vkCmdPipelineBarrier");
        s.getAddr = RF<PFN_vkGetBufferDeviceAddress>(dev, "vkGetBufferDeviceAddress");
        if (!s.getAddr) s.getAddr = RF<PFN_vkGetBufferDeviceAddress>(dev, "vkGetBufferDeviceAddressKHR");
        s.cmdBind = f.CmdBindPipeline; s.cmdPush = f.CmdPushConstants; s.cmdBarrier = f.CmdPipelineBarrier;
        s.cmdCopy = RF<PFN_vkCmdCopyBuffer>(dev, "vkCmdCopyBuffer");
        s.cmdUpdate = RF<PFN_vkCmdUpdateBuffer>(dev, "vkCmdUpdateBuffer");
        bool ok = f.CreateShaderModule && f.DestroyShaderModule && f.CreatePipelineLayout && f.CreateComputePipelines &&
                  f.CmdBindPipeline && f.CmdPushConstants && f.CmdDispatch && f.CmdPipelineBarrier && s.getAddr && s.cmdCopy && s.cmdUpdate;
        if (!ok) { LogLine("SWRT: missing driver entry points -> software RT disabled"); s.failed = true; }
        else {
            swrt::Builder::Guard guard = [](const char* what, const std::function<void()>& call) -> bool {
                SetJob("SWRT kernel '%s'", what);
                return DriverCall([&]() { call(); }, false);       // ungated: the game thread is already inside our hook
            };
            swrt::Builder::Logger lg = [](const char* m) { LogLine("%s", m); };
            LogLine("SWRT: compiling the BVH builder kernels on the driver ...");
            if (!s.builder.Init(dev, f, guard, lg)) { LogLine("SWRT: BVH builder unavailable (failed kernel mask=0x%X) -> software RT disabled, AS stay fake", s.builder.failedMask()); s.failed = true; }
        }
        if (!s.failed && !s.inited) { s.inited = true; s.restoreLayout = s.builder.Layout(); LogLine("SWRT: BVH builder ready (8 compute kernels)"); }
    }
    ReleaseSRWLockExclusive(&s.lk);
    return s.inited;
}

inline bool Ready() { return S().inited; }

inline AsRec* Lookup(VkAccelerationStructureKHR h) {
    State& s = S(); AsRec* r = nullptr;
    AcquireSRWLockShared(&s.lk);
    auto it = s.as.find(reinterpret_cast<uint64_t>(h)); if (it != s.as.end()) r = it->second;
    ReleaseSRWLockShared(&s.lk);
    return r;
}

// vkCmdPushConstants hook calls this so the game's compute push constants can be re-pushed after our dispatches.
inline void NotePush(VkCommandBuffer cb, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* p) {
    if (!(stages & VK_SHADER_STAGE_COMPUTE_BIT) || !p || offset >= 128) return;
    if (offset + size > 128) size = 128 - offset;
    State& s = S();
    AcquireSRWLockExclusive(&s.lk);
    PushShadow& ps = s.push[cb];
    memcpy(ps.d + offset, p, size);
    if (offset + size > ps.end) ps.end = offset + size;
    ReleaseSRWLockExclusive(&s.lk);
}

inline VkResult CreateAS(VkDevice dev, const VkAccelerationStructureCreateInfoKHR* ci, VkAccelerationStructureKHR* out) {
    if (!ci || !out) return VK_ERROR_INITIALIZATION_FAILED;
    State& s = S();
    if (!EnsureInit(dev)) { *out = MakeFake<VkAccelerationStructureKHR>(); return VK_SUCCESS; }   // old behaviour
    AsRec* r = new AsRec();
    r->buf = ci->buffer; r->off = ci->offset; r->size = ci->size; r->type = ci->type;
    VkBufferDeviceAddressInfo bi{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, ci->buffer };
    uint64_t base = s.getAddr(dev, &bi);
    r->addr = base + ci->offset;
    AcquireSRWLockExclusive(&s.lk);
    s.as[reinterpret_cast<uint64_t>(r)] = r;
    ReleaseSRWLockExclusive(&s.lk);
    *out = reinterpret_cast<VkAccelerationStructureKHR>(r);
    static volatile LONG n = 0; if (InterlockedIncrement(&n) <= 12)
        LogLine("SWRT: CreateAS type=%d size=%llu addr=0x%llx", (int)ci->type, (unsigned long long)ci->size, (unsigned long long)r->addr);
    return VK_SUCCESS;
}
inline void DestroyAS(VkAccelerationStructureKHR h) {
    State& s = S();
    AcquireSRWLockExclusive(&s.lk);
    auto it = s.as.find(reinterpret_cast<uint64_t>(h));
    if (it != s.as.end()) { delete it->second; s.as.erase(it); }
    ReleaseSRWLockExclusive(&s.lk);
}
inline VkDeviceAddress GetAddr(VkAccelerationStructureKHR h) { AsRec* r = Lookup(h); return r ? r->addr : 0; }

inline bool MapVertexFormat(VkFormat f, uint32_t& o) {
    switch (f) {
    case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT: o = 0; return true;
    case VK_FORMAT_R16G16B16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT: o = 1; return true;
    case VK_FORMAT_R16G16B16_SNORM:  case VK_FORMAT_R16G16B16A16_SNORM:  o = 2; return true;
    default: return false; }
}

inline void GetBuildSizes(const VkAccelerationStructureBuildGeometryInfoKHR* bi, const uint32_t* maxPrims,
                          VkAccelerationStructureBuildSizesInfoKHR* out) {
    if (!out) return;
    uint32_t n = 0, g = bi ? bi->geometryCount : 0;
    if (maxPrims) for (uint32_t i = 0; i < g; ++i) n += maxPrims[i];
    bool tlas = bi && bi->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    swrt::BlobLayout L = swrt::ComputeBlob(tlas, n, tlas ? n : g);
    swrt::ScratchLayout Sc = swrt::ComputeScratch(n);
    out->accelerationStructureSize = L.total;
    out->buildScratchSize  = Sc.total + 256;     // slack for the 128-byte scratch alignment the game applies
    out->updateScratchSize = Sc.total + 256;
}

// Saves nothing, restores everything the build clobbered in the game's command buffer.
inline void RestoreGameState(VkCommandBuffer cb) {
    State& s = S();
    VkPipeline prev = VK_NULL_HANDLE; PushShadow ps; bool havePush = false;
    AcquireSRWLockShared(&dynrend::g_lock);
    if (dynrend::g_boundCompute) { auto it = dynrend::g_boundCompute->find(cb); if (it != dynrend::g_boundCompute->end()) prev = it->second; }
    ReleaseSRWLockShared(&dynrend::g_lock);
    AcquireSRWLockShared(&s.lk);
    auto it = s.push.find(cb); if (it != s.push.end() && it->second.end) { ps = it->second; havePush = true; }
    ReleaseSRWLockShared(&s.lk);
    if (prev && !IsFake(prev)) s.cmdBind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, prev);
    if (havePush) s.cmdPush(cb, s.restoreLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (ps.end + 3u) & ~3u, ps.d);
}

inline void CmdBuild(VkCommandBuffer cb, uint32_t infoCount, const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                     const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
    State& s = S();
    if (!infos || !ranges || !EnsureInit(g_lastDevice)) return;
    for (uint32_t i = 0; i < infoCount; ++i) {
        const auto& bi = infos[i];
        AsRec* dst = Lookup(bi.dstAccelerationStructure);
        if (!dst || !bi.scratchData.deviceAddress || !ranges[i]) { InterlockedIncrement(&s.skipped); continue; }
        if (bi.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
            if (bi.geometryCount < 1) continue;
            const VkAccelerationStructureGeometryKHR& G = bi.pGeometries ? bi.pGeometries[0] : *bi.ppGeometries[0];
            if (G.geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR) continue;
            uint32_t cnt = ranges[i][0].primitiveCount;
            uint64_t ia = G.geometry.instances.data.deviceAddress + ranges[i][0].primitiveOffset;
            swrt::BlobLayout L = swrt::ComputeBlob(true, cnt, cnt);
            if (L.total > dst->size) { LogLine("SWRT: TLAS %u instances needs %llu bytes > AS size %llu - skipped", cnt, (unsigned long long)L.total, (unsigned long long)dst->size); continue; }
            s.builder.RecordTlas(cb, dst->addr, bi.scratchData.deviceAddress, ia, cnt, G.geometry.instances.arrayOfPointers != VK_FALSE);
            dst->used = L.total; InterlockedIncrement(&s.tlasBuilds);
        } else {
            std::vector<swrt::GeomDesc> geoms; uint32_t n = 0;
            for (uint32_t g = 0; g < bi.geometryCount; ++g) {
                const VkAccelerationStructureGeometryKHR& G = bi.pGeometries ? bi.pGeometries[g] : *bi.ppGeometries[g];
                const VkAccelerationStructureBuildRangeInfoKHR& R = ranges[i][g];
                swrt::GeomDesc d;
                d.flags = (G.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) ? 1u : 0u;
                bool ok = false;
                if (G.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                    const auto& T = G.geometry.triangles;
                    uint32_t fmt = 0;
                    uint32_t it = (T.indexType == VK_INDEX_TYPE_UINT16) ? 1u : (T.indexType == VK_INDEX_TYPE_UINT32) ? 2u : (T.indexType == (VkIndexType)1000165000) ? 0u : 99u;
                    if (MapVertexFormat(T.vertexFormat, fmt) && it != 99u) {
                        d.vertexFmt = fmt; d.indexType = it; d.stride = (uint32_t)T.vertexStride;
                        d.vertexAddr = T.vertexData.deviceAddress + (uint64_t)R.firstVertex * T.vertexStride + (it == 0u ? R.primitiveOffset : 0u);
                        d.indexAddr = (it != 0u) ? T.indexData.deviceAddress + R.primitiveOffset : 0;
                        d.transformAddr = T.transformData.deviceAddress ? T.transformData.deviceAddress + R.transformOffset : 0;
                        d.primCount = R.primitiveCount; ok = true;
                    } else {
                        static volatile LONG w = 0; if (InterlockedIncrement(&w) <= 6)
                            LogLine("SWRT: unsupported triangle format %d / index type %d - geometry ignored", (int)T.vertexFormat, (int)T.indexType);
                    }
                } else {
                    static volatile LONG w = 0; if (InterlockedIncrement(&w) <= 3) LogLine("SWRT: AABB geometry is not supported - ignored");
                }
                if (!ok) d.primCount = 0;
                n += d.primCount; geoms.push_back(d);
            }
            swrt::BlobLayout L = swrt::ComputeBlob(false, n, (uint32_t)geoms.size());
            if (L.total > dst->size) { LogLine("SWRT: BLAS %u prims needs %llu bytes > AS size %llu - skipped", n, (unsigned long long)L.total, (unsigned long long)dst->size); continue; }
            s.builder.RecordBlas(cb, dst->addr, bi.scratchData.deviceAddress, geoms);
            dst->used = L.total; InterlockedIncrement(&s.builds);
        }
    }
    RestoreGameState(cb);
    static volatile LONG logN = 0; if (InterlockedIncrement(&logN) <= 5)
        LogLine("SWRT: CmdBuildAccelerationStructures x%u recorded (blas=%ld tlas=%ld skipped=%ld)", infoCount, s.builds, s.tlasBuilds, s.skipped);
}

inline void CmdCopy(VkCommandBuffer cb, const VkCopyAccelerationStructureInfoKHR* ci) {
    State& s = S();
    if (!ci || !s.inited) return;
    AsRec* a = Lookup(ci->src); AsRec* b = Lookup(ci->dst);
    if (!a || !b || a == b) return;
    VkDeviceSize bytes = a->used ? a->used : a->size;
    if (bytes > b->size) bytes = b->size;
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
    s.cmdBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy rg{ a->off, b->off, bytes };
    s.cmdCopy(cb, a->buf, b->buf, 1, &rg);
    s.cmdBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    b->used = a->used;
}

// ---- acceleration-structure query pools (CPU emulated) ---------------------------------------------------------------
inline QPool* FindPool(VkQueryPool p) {
    State& s = S(); QPool* r = nullptr;
    AcquireSRWLockShared(&s.lk);
    auto it = s.qp.find(reinterpret_cast<uint64_t>(p)); if (it != s.qp.end()) r = it->second;
    ReleaseSRWLockShared(&s.lk);
    return r;
}
inline bool MaybeCreatePool(const VkQueryPoolCreateInfo* ci, VkQueryPool* out) {
    if (!ci || !out || !g_spvCfg.softwareRT || !IsAsQueryType(ci->queryType)) return false;
    State& s = S();
    QPool* q = new QPool(); q->type = ci->queryType; q->vals.assign(ci->queryCount ? ci->queryCount : 1, 0);
    AcquireSRWLockExclusive(&s.lk);
    s.qp[reinterpret_cast<uint64_t>(q)] = q;
    ReleaseSRWLockExclusive(&s.lk);
    *out = reinterpret_cast<VkQueryPool>(q);
    return true;
}
inline bool MaybeDestroyPool(VkQueryPool p) {
    State& s = S(); bool had = false;
    AcquireSRWLockExclusive(&s.lk);
    auto it = s.qp.find(reinterpret_cast<uint64_t>(p));
    if (it != s.qp.end()) { delete it->second; s.qp.erase(it); had = true; }
    ReleaseSRWLockExclusive(&s.lk);
    return had;
}
inline void CmdWriteProps(uint32_t count, const VkAccelerationStructureKHR* as, VkQueryPool pool, uint32_t first) {
    QPool* q = FindPool(pool); if (!q || !as) return;
    for (uint32_t i = 0; i < count; ++i) {
        AsRec* r = Lookup(as[i]); if (first + i >= q->vals.size()) q->vals.resize(first + i + 1, 0);
        q->vals[first + i] = r ? (r->used ? r->used : r->size) : 0;
    }
}
inline VkResult GetResults(VkQueryPool pool, uint32_t first, uint32_t count, size_t dataSize, void* data, VkDeviceSize stride, VkQueryResultFlags flags) {
    QPool* q = FindPool(pool); if (!q) return VK_ERROR_UNKNOWN;
    const bool b64 = (flags & VK_QUERY_RESULT_64_BIT) != 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t v = (first + i < q->vals.size()) ? q->vals[first + i] : 0;
        uint8_t* p = static_cast<uint8_t*>(data) + (size_t)i * stride;
        if ((size_t)((p - static_cast<uint8_t*>(data)) + (b64 ? 8 : 4)) > dataSize) break;
        if (b64) memcpy(p, &v, 8); else { uint32_t v32 = (uint32_t)v; memcpy(p, &v32, 4); }
    }
    return VK_SUCCESS;
}
inline void CmdCopyResults(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize off, VkDeviceSize stride, VkQueryResultFlags flags) {
    QPool* q = FindPool(pool); State& s = S(); if (!q || !s.cmdUpdate) return;
    const bool b64 = (flags & VK_QUERY_RESULT_64_BIT) != 0; const VkDeviceSize w = b64 ? 8 : 4;
    if (stride < w) stride = w;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t v = (first + i < q->vals.size()) ? q->vals[first + i] : 0; uint32_t v32 = (uint32_t)v;
        s.cmdUpdate(cb, dst, off + i * stride, w, b64 ? (const void*)&v : (const void*)&v32);
    }
}

} // namespace swpx
