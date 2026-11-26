// Vulkan proxy v3.2 (hang-fix) for Windows x64
// v3.2 changes (see CHANGES-v3.2 comments tagged [v3.2]):
//   * compute shaders >= HeavyComputeWords (default 75000 SPIR-V words) are never sent to the AMD driver:
//     in the log every such compile hung LLPC for >40 minutes. They become FAKE pipelines instantly.
//   * compute compile timeout is ComputeTimeoutSec (default 12 s) instead of 60 s; abandoned compile threads are
//     pinned to ONE cpu core so they stop eating the cores the game needs.
//   * pipeline tracking is per bind point (graphics / compute). Before, a FAKE compute pipeline made every
//     following draw on the same command buffer disappear (and a real compute pipeline cleared a fake graphics one).
//   * vkBind*Memory no longer waits on the shader-compile gate (one stuck compile used to freeze memory binding).
//   * VK_EXT_debug_marker functions the game calls without a null check get harmless stubs.
//   * diagnostics: present/submit counters + VkResult in the heartbeat, DEVICE_LOST logging, and a STALL dump
//     (all thread stacks) when nothing is presented for StallDumpSec seconds.
// Vulkan proxy v3.1 (bug-fixed) for Windows x64 (vulkan-1.dll replacement that forwards to the
// real system loader).
//
// Adds to v2:
//   * Emulates VK_KHR_dynamic_rendering by translating vkCmdBeginRendering /
//     vkCmdEndRendering into a synthesized VkRenderPass + vkCmdBeginRenderPass /
//     vkCmdEndRenderPass pair. vkCreateGraphicsPipelines substitutes the
//     renderPass on pipelines that carry VkPipelineRenderingCreateInfo.
//   * Implements maintenance4 (vkGetDevice*MemoryRequirements) by creating a
//     temporary resource and querying its legacy memory requirements.
//   * Tracks VkImageView -> format and VkCommandBuffer -> VkDevice so
//     BeginRendering works even before a pipeline is bound (vkd3d-style order).
//   * No longer strips dynamicRendering / maintenance4 / synchronization2 from
//     the feature chain at device creation; the driver silently ignores these
//     fields because it does not understand Vulkan 1.3 sTypes.
//
// The v2 RT spoofing (features, extensions, stubs) is retained verbatim.

#define WIN32_LEAN_AND_MEAN
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <tlhelp32.h>
#include <vulkan/vulkan.h>
#ifdef _MSC_VER
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cwchar>
#include <functional>
#include <vector>
#include <algorithm>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vk_proxy_exports.generated.h"

namespace { namespace swpx { inline bool EnsureInit(VkDevice dev); } }   // [SWRT] defined in swrt_proxy.h

// Draw/dispatch hooks (defined further below). They skip commands recorded while a FAKE/NULL pipeline is bound.
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize);

// [RTFILL] Forward-декларации модуля заполнения RT-выходов.
namespace rtfill {
    void TrackView(VkImageView view, const VkImageViewCreateInfo* ci);
    void TrackWrites(uint32_t count, const VkWriteDescriptorSet* writes);
    void TrackBind(VkCommandBuffer cb, uint32_t firstSet, uint32_t count, const VkDescriptorSet* sets);
    void TrackBarriers(uint32_t count, const VkImageMemoryBarrier* barriers);
    void OnTraceRays(VkCommandBuffer cb, uint32_t width, uint32_t height);
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCount(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCount(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t);


static_assert(sizeof(void*) == 8, "Build this proxy for x64.");

extern "C" {
    PVOID g_vkSlots[VK_PROXY_EXPORT_COUNT] = {};
}

// ---------------------------------------------------------------------------
// Forward declarations of proxy hooks (defined below, after the helpers).
// These must be visible to stub::Find, so they go before the anonymous
// namespace.
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* pInfo);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* pInfo);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdEndRendering(VkCommandBuffer cb);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdEndRenderingKHR(VkCommandBuffer cb);

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImageView* pView);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* pAllocator);

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pInfo,
    VkCommandBuffer* pBuffers);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkFreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
    const VkCommandBuffer* pBuffers);

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceBufferMemoryRequirements(VkDevice device,
    const VkDeviceBufferMemoryRequirements* pInfo, VkMemoryRequirements2* pMemoryRequirements);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceImageMemoryRequirements(VkDevice device,
    const VkDeviceImageMemoryRequirements* pInfo, VkMemoryRequirements2* pMemoryRequirements);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceImageSparseMemoryRequirements(VkDevice device,
    const VkDeviceImageMemoryRequirements* pInfo, uint32_t* pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2* pSparseMemoryRequirements);
	
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator,
    VkPipeline* pPipelines);
	
extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);
	
	

// ---------------------------------------------------------------------------
// Configuration + helpers
// ---------------------------------------------------------------------------
namespace {
    // Version reported to the game when the real driver reports less.
    const uint32_t kSpoofMinApiMajorMinor = VK_MAKE_API_VERSION(0, 1, 3, 0);

    // Extensions we advertise when the real device lacks them.
    constexpr const char* kFakeExtensions[] = {
        "VK_KHR_deferred_host_operations",
        "VK_KHR_pipeline_library",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_query",
        "VK_KHR_dynamic_rendering"
    };
    constexpr size_t kFakeExtensionCount = sizeof(kFakeExtensions) / sizeof(kFakeExtensions[0]);

    // sType values written as numbers so this compiles with slightly older headers.
    constexpr uint32_t ST_VULKAN11_FEATURES        = 49;
    constexpr uint32_t ST_VULKAN11_PROPERTIES      = 50;
    constexpr uint32_t ST_VULKAN12_FEATURES        = 51;
    constexpr uint32_t ST_FLOAT16_INT8_FEAT        = 1000082000u;
    constexpr uint32_t ST_VULKAN13_FEATURES        = 53;
    constexpr uint32_t ST_DYNAMIC_RENDERING_FEAT   = 1000044003u;
    constexpr uint32_t ST_SYNCHRONIZATION2_FEAT    = 1000314007u;
    constexpr uint32_t ST_MAINTENANCE4_FEAT        = 1000413000u;
    constexpr uint32_t ST_AS_FEATURES              = 1000150013u;
    constexpr uint32_t ST_AS_PROPERTIES            = 1000150014u;
    constexpr uint32_t ST_RTP_FEATURES             = 1000347000u;
    constexpr uint32_t ST_RTP_PROPERTIES           = 1000347001u;
    constexpr uint32_t ST_RAY_QUERY_FEATURES       = 1000348013u;
    constexpr uint32_t ST_RT_MAINTENANCE1_FEATURES = 1000386000u;

    INIT_ONCE g_loaderInitOnce = INIT_ONCE_STATIC_INIT;
    HMODULE g_realLoader = nullptr;
    HMODULE g_thisModule = nullptr;
    PFN_vkGetInstanceProcAddr g_realGipa = nullptr;
    PFN_vkGetDeviceProcAddr g_realGdpa = nullptr;
    VkDevice g_lastDevice = nullptr;   // for resolving driver function names in crash reports
    std::vector<std::pair<uint32_t, uint32_t>> g_createdFamilies;   // (queue family, queue count) from vkCreateDevice
    SRWLOCK g_logLock = SRWLOCK_INIT;
    SRWLOCK g_stateLock = SRWLOCK_INIT;
    std::set<std::string>* g_onceKeys = nullptr;

    // 1 = serve RT stubs / spoof RT (default until a real RT device is created).
    volatile LONG g_rtSpoof = 1;

    // 1 = VK_EXT_extended_dynamic_state was enabled on the real device (core 1.3 vkCmdSet* are routed to it).
    volatile LONG g_edsEnabled = 0;

// [v3.2] frame / submit counters for the watchdog heartbeat and the stall detector.
volatile LONG g_presentCount = 0;
volatile LONG g_submitCount = 0;
volatile LONG g_lastPresentRes = 0;
volatile LONG g_lastSubmitRes = 0;
volatile LONG64 g_lastPresentTick = 0;
volatile LONG64 g_lastSubmitTick = 0;

    // -----------------------------------------------------------------------
    // Logging
    // -----------------------------------------------------------------------
    void GetLogPath(wchar_t* path, size_t capacity) noexcept {
        if (!path || capacity == 0) return;
        path[0] = L'\0';
        DWORD length = g_thisModule ? GetModuleFileNameW(g_thisModule, path, static_cast<DWORD>(capacity)) : 0;
        if (length == 0 || length >= capacity) {
            wcscpy_s(path, capacity, L"VulkanProxy_diagnostic.log");
            return;
        }
        wchar_t* slash = wcsrchr(path, L'\\');
        if (!slash) {
            wcscpy_s(path, capacity, L"VulkanProxy_diagnostic.log");
            return;
        }
        const size_t offset = static_cast<size_t>(slash - path) + 1;
        constexpr wchar_t fileName[] = L"VulkanProxy_diagnostic.log";
        if (offset + (sizeof(fileName) / sizeof(fileName[0])) > capacity) {
            wcscpy_s(path, capacity, L"VulkanProxy_diagnostic.log");
            return;
        }
        memcpy(path + offset, fileName, sizeof(fileName));
    }

    volatile LONG g_logVerbose = 0;   // VulkanProxy.ini [spirv] Verbose=1 disables the noise filter

    // Per-call chatter that makes the log huge and tells us nothing.
    bool IsLogNoise(const char* m) noexcept {
        if (strstr(m, "FAIL") || strstr(m, "fail") || strstr(m, "CAUGHT") || strstr(m, "!!!") ||
            strstr(m, "ERROR") || strstr(m, "error")) return false;
        static const char* const kNoise[] = {
            "vkAllocateDescriptorSets", "vkUpdateDescriptorSets", "vkGetImageMemoryRequirements",
            "vkCreateSampler", "vkCreateGraphicsPipelines ENTER", "  pipeline[", "vkCreatePipelineLayout:",
            "vkCreateDescriptorSetLayout", "vkCreateDescriptorPool", "vkCreateCommandPool", "vkCreateFence",
            "vkCreateRenderPass", "vkCreateImageView", "vkCreateImage", "vkCreateBuffer", "vkAllocateMemory",
            "vkBindImageMemory", "vkBindBufferMemory", "vkCmd", "vkQueueSubmit" };
        for (const char* n : kNoise) if (strncmp(m, n, strlen(n)) == 0) return true;
        return false;
    }

    void LogLine(const char* format, ...) noexcept {
        char message[2048]{};
        va_list args;
        va_start(args, format);
        vsnprintf_s(message, sizeof(message), _TRUNCATE, format, args);
        va_end(args);
        if (!g_logVerbose && IsLogNoise(message)) return;

        char line[2304]{};
        sprintf_s(line, sizeof(line), "[%llu] %s\r\n",
            static_cast<unsigned long long>(GetTickCount64()), message);
        OutputDebugStringA(line);

        wchar_t logPath[2048]{};
        GetLogPath(logPath, sizeof(logPath) / sizeof(logPath[0]));
        AcquireSRWLockExclusive(&g_logLock);
        HANDLE file = CreateFileW(logPath, FILE_APPEND_DATA,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            WriteFile(file, line, static_cast<DWORD>(strlen(line)), &written, nullptr);
            CloseHandle(file);
        }
        ReleaseSRWLockExclusive(&g_logLock);
    }

    // True the first time `key` is seen. Used to keep the log readable.
    bool FirstTime(const std::string& key) {
        AcquireSRWLockExclusive(&g_stateLock);
        if (!g_onceKeys) g_onceKeys = new (std::nothrow) std::set<std::string>();
        bool inserted = g_onceKeys ? g_onceKeys->insert(key).second : false;
        ReleaseSRWLockExclusive(&g_stateLock);
        return inserted;
    }

    // -----------------------------------------------------------------------
    // System loader
    // -----------------------------------------------------------------------
    BOOL CALLBACK LoadSystemVulkanLoader(PINIT_ONCE, PVOID, PVOID*) noexcept {
        wchar_t systemDirectory[2048]{};
        const UINT capacity = static_cast<UINT>(sizeof(systemDirectory) / sizeof(systemDirectory[0]));
        const UINT length = GetSystemDirectoryW(systemDirectory, capacity);
        if (length == 0 || length >= capacity) {
            LogLine("GetSystemDirectoryW failed; Win32 error=%lu", GetLastError());
            return TRUE;
        }
        size_t pathLength = static_cast<size_t>(length);
        if (systemDirectory[pathLength - 1] != L'\\') {
            if (pathLength + 1 >= capacity) return TRUE;
            systemDirectory[pathLength++] = L'\\';
        }
        constexpr wchar_t loaderName[] = L"vulkan-1.dll";
        constexpr size_t loaderNameChars = sizeof(loaderName) / sizeof(loaderName[0]);
        if (pathLength + loaderNameChars > capacity) return TRUE;
        memcpy(systemDirectory + pathLength, loaderName, sizeof(loaderName));

        g_realLoader = LoadLibraryExW(systemDirectory, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g_realLoader) {
            LogLine("Failed to load system Vulkan Loader: %lu", GetLastError());
            return TRUE;
        }
        g_realGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(g_realLoader, "vkGetInstanceProcAddr"));
        g_realGdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            GetProcAddress(g_realLoader, "vkGetDeviceProcAddr"));
        if (!g_realGipa || !g_realGdpa) {
            LogLine("System Loader missing vkGetInstanceProcAddr or vkGetDeviceProcAddr");
            g_realGipa = nullptr;
            g_realGdpa = nullptr;
            return TRUE;
        }
        LogLine("=== VulkanProxy v3 === Loaded system Loader: %ls", systemDirectory);
        return TRUE;
    }

    bool EnsureSystemLoader() noexcept {
        if (!InitOnceExecuteOnce(&g_loaderInitOnce, LoadSystemVulkanLoader, nullptr, nullptr)) return false;
        return g_realLoader && g_realGipa && g_realGdpa;
    }

    // GetProcAddress on a named export is a binary search with string compares; it used to run for EVERY
    // recorded command (draw, bind, barrier ...). All callers pass string literals, so cache by pointer.
    FARPROC RealProc(const char* name) noexcept {
        if (!EnsureSystemLoader()) return nullptr;
        struct Slot { const char* k; FARPROC v; };
        static thread_local Slot cache[512];
        const size_t h = (reinterpret_cast<uintptr_t>(name) >> 3) & 511u;
        if (cache[h].k == name && cache[h].v) return cache[h].v;
        FARPROC p = GetProcAddress(g_realLoader, name);
        if (p) { cache[h].k = name; cache[h].v = p; }
        return p;
    }

    template<typename T>
    T RealExport(const char* name) noexcept {
        return reinterpret_cast<T>(RealProc(name));
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    bool IsFakeExtension(const char* name) noexcept {
        if (!name) return false;
        for (const char* e : kFakeExtensions) if (strcmp(name, e) == 0) return true;
        return false;
    }

    bool IsRayTracingProc(const char* name) noexcept {
        if (!name) return false;
        return strstr(name, "RayTracing") || strstr(name, "AccelerationStructure") ||
               strstr(name, "TraceRays") || strstr(name, "RayQuery") ||
               strstr(name, "DeferredOperation");
    }

    // Real (unmodified) device extension list.
    bool QueryRealExtensions(VkPhysicalDevice pd, const char* layer,
                             std::vector<VkExtensionProperties>& out) {
        out.clear();
        auto real = RealExport<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        if (!real) return false;
        uint32_t count = 0;
        VkResult r = real(pd, layer, &count, nullptr);
        if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count > 8192u) return false;
        out.resize(count);
        r = real(pd, layer, &count, out.data());
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) { out.clear(); return false; }
        out.resize(count);
        return true;
    }

    bool HasExt(const std::vector<VkExtensionProperties>& v, const char* name) {
        for (const auto& e : v) if (strcmp(e.extensionName, name) == 0) return true;
        return false;
    }

    // Does this physical device REALLY support RT (so we must not spoof)?
    bool DeviceHasRealRT(VkPhysicalDevice pd) {
        std::vector<VkExtensionProperties> exts;
        if (!QueryRealExtensions(pd, nullptr, exts)) return false;
        return HasExt(exts, "VK_KHR_acceleration_structure") &&
               HasExt(exts, "VK_KHR_ray_tracing_pipeline");
    }

    const char* KnownSTypeName(uint32_t s) {
        switch (s) {
        case 49: return "Vulkan11Features";
        case 51: return "Vulkan12Features";
        case ST_VULKAN13_FEATURES: return "Vulkan13Features";
        case 1000280000u: return "ShaderIntegerDotProductFeatures";
        case ST_DYNAMIC_RENDERING_FEAT: return "DynamicRenderingFeatures";
        case ST_SYNCHRONIZATION2_FEAT: return "Synchronization2Features";
        case ST_MAINTENANCE4_FEAT: return "Maintenance4Features";
        case ST_AS_FEATURES: return "AccelerationStructureFeaturesKHR";
        case ST_RTP_FEATURES: return "RayTracingPipelineFeaturesKHR";
        case ST_RAY_QUERY_FEATURES: return "RayQueryFeaturesKHR";
        case ST_RT_MAINTENANCE1_FEATURES: return "RayTracingMaintenance1FeaturesKHR";
        case ST_AS_PROPERTIES: return "AccelerationStructurePropertiesKHR";
        case ST_RTP_PROPERTIES: return "RayTracingPipelinePropertiesKHR";
        case 1000059000u: return "Features2";
        case 1000059001u: return "Properties2";
        case 50: return "Vulkan11Properties";
        case 52: return "Vulkan12Properties";
        case ST_FLOAT16_INT8_FEAT: return "ShaderFloat16Int8Features";
        case 1000094000u: return "SubgroupProperties";
        case 1000286000u: return "Robustness2Features";
        default: return "";
        }
    }

    // Log every distinct sType the application puts in a chain, once.
    void LogChainOnce(const char* label, const void* pNext) {
        const VkBaseInStructure* node = reinterpret_cast<const VkBaseInStructure*>(pNext);
        unsigned n = 0;
        while (node && n < 64) {
            const uint32_t s = static_cast<uint32_t>(node->sType);
            char key[96];
            sprintf_s(key, sizeof(key), "chain:%s:%u", label, s);
            if (FirstTime(key)) LogLine("%s requests sType=%u %s", label, s, KnownSTypeName(s));
            node = node->pNext;
            ++n;
        }
    }

    // Fake non-dispatchable handles carry a recognisable tag in the top 16 bits.
    constexpr uint64_t kFakeTag = 0xFA4Eull;
    volatile LONG64 g_fakeCounter = 0;

    template<typename H> H MakeFake() {
        const uint64_t v = (kFakeTag << 48) | static_cast<uint64_t>(InterlockedIncrement64(&g_fakeCounter));
        return reinterpret_cast<H>(static_cast<uintptr_t>(v));
    }
    template<typename H> bool IsFake(H h) {
        return (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h)) >> 48) == kFakeTag;
    }

    // -----------------------------------------------------------------------
    // Spoofing of properties / features
    // -----------------------------------------------------------------------
    void FixupProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p) {
        if (!p) return;
        const uint32_t real = p->apiVersion;
        char key[64];
        sprintf_s(key, sizeof(key), "gpu:%p", static_cast<void*>(pd));
        if (FirstTime(key)) {
            LogLine("GPU: name=\"%s\" vendor=0x%04X device=0x%04X type=%u driverVersion=0x%08X apiVersion(real)=%u.%u.%u",
                p->deviceName, p->vendorID, p->deviceID, static_cast<unsigned>(p->deviceType),
                p->driverVersion, VK_API_VERSION_MAJOR(real), VK_API_VERSION_MINOR(real),
                VK_API_VERSION_PATCH(real));
        }
        const uint32_t realMM = real & ~0xFFFu;
        if (realMM < kSpoofMinApiMajorMinor) {
            p->apiVersion = kSpoofMinApiMajorMinor | (real & 0xFFFu);
            if (FirstTime("spoof:apiVersion"))
                LogLine("SPOOF: apiVersion %u.%u -> 1.3", VK_API_VERSION_MAJOR(real), VK_API_VERSION_MINOR(real));
        }
    }

    void FixupPropertiesChain(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p2) {
        const bool spoofRt = g_rtSpoof && !DeviceHasRealRT(pd);
        auto* node = reinterpret_cast<VkBaseOutStructure*>(p2->pNext);
        unsigned i = 0;
        while (node && i < 64) {
            const uint32_t s = static_cast<uint32_t>(node->sType);
            if (s == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES && spoofRt) {
                auto* sg = reinterpret_cast<VkPhysicalDeviceSubgroupProperties*>(node);
                const VkShaderStageFlags before = sg->supportedStages;
                sg->supportedStages |= 0xF00u;
                if (FirstTime("spoof:subgroup"))
                    LogLine("SPOOF: subgroup supportedStages 0x%X -> 0x%X (size=%u ops=0x%X)",
                        static_cast<unsigned>(before), static_cast<unsigned>(sg->supportedStages),
                        sg->subgroupSize, static_cast<unsigned>(sg->supportedOperations));
            } else if (s == ST_VULKAN11_PROPERTIES && spoofRt) {
                auto* p = reinterpret_cast<VkPhysicalDeviceVulkan11Properties*>(node);
                const VkShaderStageFlags before = p->subgroupSupportedStages;
                p->subgroupSupportedStages |= 0xF00u;
                if (FirstTime("spoof:subgroup11"))
                    LogLine("SPOOF: Vulkan11Properties subgroupSupportedStages 0x%X -> 0x%X",
                        static_cast<unsigned>(before), static_cast<unsigned>(p->subgroupSupportedStages));
            } else if (s == ST_RTP_PROPERTIES && spoofRt) {
                auto* p = reinterpret_cast<VkPhysicalDeviceRayTracingPipelinePropertiesKHR*>(node);
                p->shaderGroupHandleSize = 32;
                p->maxRayRecursionDepth = 31;
                p->maxShaderGroupStride = 4096;
                p->shaderGroupBaseAlignment = 64;
                p->shaderGroupHandleCaptureReplaySize = 32;
                p->maxRayDispatchInvocationCount = 1u << 30;
                p->shaderGroupHandleAlignment = 32;
                p->maxRayHitAttributeSize = 32;
                if (FirstTime("spoof:rtpprops")) LogLine("SPOOF: RayTracingPipelineProperties filled");
            } else if (s == ST_AS_PROPERTIES && spoofRt) {
                auto* p = reinterpret_cast<VkPhysicalDeviceAccelerationStructurePropertiesKHR*>(node);
                p->maxGeometryCount = (1ull << 24) - 1;
                p->maxInstanceCount = (1ull << 24) - 1;
                p->maxPrimitiveCount = (1ull << 29) - 1;
                p->maxPerStageDescriptorAccelerationStructures = 16;
                p->maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 16;
                p->maxDescriptorSetAccelerationStructures = 16;
                p->maxDescriptorSetUpdateAfterBindAccelerationStructures = 16;
                p->minAccelerationStructureScratchOffsetAlignment = 128;
                if (FirstTime("spoof:asprops")) LogLine("SPOOF: AccelerationStructureProperties filled");
            }
            node = node->pNext;
            ++i;
        }
    }

    inline void SetBool(void* node, size_t index) {
        reinterpret_cast<VkBool32*>(reinterpret_cast<uint8_t*>(node) + 16)[index] = VK_TRUE;
    }

    void FixupFeaturesChain(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f2) {
        const bool spoofRt = g_rtSpoof && !DeviceHasRealRT(pd);
        auto* node = reinterpret_cast<VkBaseOutStructure*>(f2->pNext);
        unsigned i = 0;
        while (node && i < 64) {
            const uint32_t s = static_cast<uint32_t>(node->sType);
            if (s == ST_VULKAN12_FEATURES) {
                auto* f = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(node);
                if (FirstTime("real:vk12features"))
                    LogLine("REAL Vulkan12Features: shaderFloat16=%u shaderInt8=%u storageBuffer8BitAccess=%u bufferDeviceAddress=%u descriptorIndexing=%u timelineSemaphore=%u scalarBlockLayout=%u",
                        f->shaderFloat16, f->shaderInt8, f->storageBuffer8BitAccess, f->bufferDeviceAddress,
                        f->descriptorIndexing, f->timelineSemaphore, f->scalarBlockLayout);
                if (!f->shaderFloat16 && FirstTime("spoof:fp16"))
                    LogLine("SPOOF: Vulkan12Features.shaderFloat16 0 -> 1 (will rewrite fp16 shaders to fp32)");
                f->shaderFloat16 = VK_TRUE;
            } else if (s == ST_FLOAT16_INT8_FEAT) {
                auto* f = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(node);
                if (FirstTime("real:f16i8"))
                    LogLine("REAL ShaderFloat16Int8Features: shaderFloat16=%u shaderInt8=%u", f->shaderFloat16, f->shaderInt8);
                f->shaderFloat16 = VK_TRUE;
            } else if (s == ST_VULKAN11_FEATURES) {
                auto* f = reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(node);
                if (FirstTime("real:vk11features"))
                    LogLine("REAL Vulkan11Features: storageBuffer16BitAccess=%u uniformAndStorageBuffer16BitAccess=%u storagePushConstant16=%u storageInputOutput16=%u multiview=%u shaderDrawParameters=%u",
                        f->storageBuffer16BitAccess, f->uniformAndStorageBuffer16BitAccess,
                        f->storagePushConstant16, f->storageInputOutput16, f->multiview, f->shaderDrawParameters);
            } else if (s == ST_VULKAN13_FEATURES) {
                auto* f = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(node);
                if (FirstTime("real:vk13features"))
                    LogLine("REAL Vulkan13Features: dynamicRendering=%u synchronization2=%u maintenance4=%u subgroupSizeControl=%u computeFullSubgroups=%u shaderIntegerDotProduct=%u",
                        f->dynamicRendering, f->synchronization2, f->maintenance4,
                        f->subgroupSizeControl, f->computeFullSubgroups, f->shaderIntegerDotProduct);
                f->dynamicRendering = VK_TRUE;
                f->synchronization2 = VK_TRUE;
                f->maintenance4 = VK_TRUE;
            } else if (s == ST_DYNAMIC_RENDERING_FEAT) {
                if (FirstTime("real:dynrendfeat"))
                    LogLine("REAL DynamicRenderingFeatures: dynamicRendering=%u",
                        reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeatures*>(node)->dynamicRendering);
                SetBool(node, 0);
            } else if (s == ST_SYNCHRONIZATION2_FEAT) {
                SetBool(node, 0);
            } else if (s == ST_MAINTENANCE4_FEAT) {
                SetBool(node, 0);
            } else if (spoofRt && s == ST_AS_FEATURES) {
                SetBool(node, 0); SetBool(node, 2); SetBool(node, 4);
                if (FirstTime("spoof:asfeat")) LogLine("SPOOF: AccelerationStructureFeatures = TRUE");
            } else if (spoofRt && s == ST_RTP_FEATURES) {
                SetBool(node, 0); SetBool(node, 3); SetBool(node, 4);
                if (FirstTime("spoof:rtpfeat")) LogLine("SPOOF: RayTracingPipelineFeatures = TRUE");
            } else if (spoofRt && s == ST_RAY_QUERY_FEATURES) {
                SetBool(node, 0);
                if (FirstTime("spoof:rqfeat")) LogLine("SPOOF: RayQueryFeatures = TRUE");
            } else if (spoofRt && s == ST_RT_MAINTENANCE1_FEATURES) {
                SetBool(node, 0);
                SetBool(node, 1);
            }
            node = node->pNext;
            ++i;
        }
    }

    const char* const kNamesCore[] = { "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray", "independentBlend", "geometryShader", "tessellationShader", "sampleRateShading", "dualSrcBlend", "logicOp", "multiDrawIndirect", "drawIndirectFirstInstance", "depthClamp", "depthBiasClamp", "fillModeNonSolid", "depthBounds", "wideLines", "largePoints", "alphaToOne", "multiViewport", "samplerAnisotropy", "textureCompressionETC2", "textureCompressionASTC_LDR", "textureCompressionBC", "occlusionQueryPrecise", "pipelineStatisticsQuery", "vertexPipelineStoresAndAtomics", "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize", "shaderImageGatherExtended", "shaderStorageImageExtendedFormats", "shaderStorageImageMultisample", "shaderStorageImageReadWithoutFormat", "shaderStorageImageWriteWithoutFormat", "shaderUniformBufferArrayDynamicIndexing", "shaderSampledImageArrayDynamicIndexing", "shaderStorageBufferArrayDynamicIndexing", "shaderStorageImageArrayDynamicIndexing", "shaderClipDistance", "shaderCullDistance", "shaderFloat64", "shaderInt64", "shaderInt16", "shaderResourceResidency", "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer", "sparseResidencyImage2D", "sparseResidencyImage3D", "sparseResidency2Samples", "sparseResidency4Samples", "sparseResidency8Samples", "sparseResidency16Samples", "sparseResidencyAliased", "variableMultisampleRate", "inheritedQueries" };
    const char* const kNames11[] = { "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess", "storagePushConstant16", "storageInputOutput16", "multiview", "multiviewGeometryShader", "multiviewTessellationShader", "variablePointersStorageBuffer", "variablePointers", "protectedMemory", "samplerYcbcrConversion", "shaderDrawParameters" };
    const char* const kNames12[] = { "samplerMirrorClampToEdge", "drawIndirectCount", "storageBuffer8BitAccess", "uniformAndStorageBuffer8BitAccess", "storagePushConstant8", "shaderBufferInt64Atomics", "shaderSharedInt64Atomics", "shaderFloat16", "shaderInt8", "descriptorIndexing", "shaderInputAttachmentArrayDynamicIndexing", "shaderUniformTexelBufferArrayDynamicIndexing", "shaderStorageTexelBufferArrayDynamicIndexing", "shaderUniformBufferArrayNonUniformIndexing", "shaderSampledImageArrayNonUniformIndexing", "shaderStorageBufferArrayNonUniformIndexing", "shaderStorageImageArrayNonUniformIndexing", "shaderInputAttachmentArrayNonUniformIndexing", "shaderUniformTexelBufferArrayNonUniformIndexing", "shaderStorageTexelBufferArrayNonUniformIndexing", "descriptorBindingUniformBufferUpdateAfterBind", "descriptorBindingSampledImageUpdateAfterBind", "descriptorBindingStorageImageUpdateAfterBind", "descriptorBindingStorageBufferUpdateAfterBind", "descriptorBindingUniformTexelBufferUpdateAfterBind", "descriptorBindingStorageTexelBufferUpdateAfterBind", "descriptorBindingUpdateUnusedWhilePending", "descriptorBindingPartiallyBound", "descriptorBindingVariableDescriptorCount", "runtimeDescriptorArray", "samplerFilterMinmax", "scalarBlockLayout", "imagelessFramebuffer", "uniformBufferStandardLayout", "shaderSubgroupExtendedTypes", "separateDepthStencilLayouts", "hostQueryReset", "timelineSemaphore", "bufferDeviceAddress", "bufferDeviceAddressCaptureReplay", "bufferDeviceAddressMultiDevice", "vulkanMemoryModel", "vulkanMemoryModelDeviceScope", "vulkanMemoryModelAvailabilityVisibilityChains", "shaderOutputViewportIndex", "shaderOutputLayer", "subgroupBroadcastDynamicId" };
    const char* const kNames13[] = { "robustImageAccess", "inlineUniformBlock", "descriptorBindingInlineUniformBlockUpdateAfterBind", "pipelineCreationCacheControl", "privateData", "shaderDemoteToHelperInvocation", "shaderTerminateInvocation", "subgroupSizeControl", "computeFullSubgroups", "synchronization2", "textureCompressionASTC_HDR", "shaderZeroInitializeWorkgroupMemory", "dynamicRendering", "shaderIntegerDotProduct", "maintenance4" };

    struct ClampInfo { uint32_t sType; size_t size; const char* name; const char* const* fields; size_t fieldCount; };
#define CL(st, T, nm) { static_cast<uint32_t>(st), sizeof(T), nm, nullptr, 0 }
#define CLN(st, T, nm, arr) { static_cast<uint32_t>(st), sizeof(T), nm, arr, sizeof(arr) / sizeof(arr[0]) }
    const ClampInfo kClampable[] = {
        CLN(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, VkPhysicalDeviceVulkan11Features, "Vulkan11Features", kNames11),
        CLN(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features, "Vulkan12Features", kNames12),
        CLN(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features, "Vulkan13Features", kNames13),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features, "ShaderFloat16Int8"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, VkPhysicalDeviceDynamicRenderingFeatures, "DynamicRendering"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, VkPhysicalDeviceSynchronization2Features, "Synchronization2"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, VkPhysicalDeviceMaintenance4Features, "Maintenance4"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, VkPhysicalDeviceRobustness2FeaturesEXT, "Robustness2(robustBufferAccess2,robustImageAccess2,nullDescriptor)"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, VkPhysicalDeviceExtendedDynamicStateFeaturesEXT, "ExtendedDynamicState"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT, VkPhysicalDeviceDepthClipEnableFeaturesEXT, "DepthClipEnable"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, VkPhysicalDeviceSubgroupSizeControlFeatures, "SubgroupSizeControl(subgroupSizeControl,computeFullSubgroups)"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, VkPhysicalDevice8BitStorageFeatures, "8BitStorage"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, VkPhysicalDevice16BitStorageFeatures, "16BitStorage"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, VkPhysicalDeviceBufferDeviceAddressFeatures, "BufferDeviceAddress"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, VkPhysicalDeviceDescriptorIndexingFeatures, "DescriptorIndexing"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, VkPhysicalDeviceTimelineSemaphoreFeatures, "TimelineSemaphore"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, VkPhysicalDeviceScalarBlockLayoutFeatures, "ScalarBlockLayout"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, VkPhysicalDeviceHostQueryResetFeatures, "HostQueryReset"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures, "ShaderDemoteToHelper"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, VkPhysicalDevicePipelineCreationCacheControlFeatures, "PipelineCreationCacheControl"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, VkPhysicalDevicePrivateDataFeatures, "PrivateData"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES, VkPhysicalDeviceImageRobustnessFeatures, "ImageRobustness"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, VkPhysicalDeviceInlineUniformBlockFeatures, "InlineUniformBlock"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, VkPhysicalDeviceColorWriteEnableFeaturesEXT, "ColorWriteEnable"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT, VkPhysicalDevice4444FormatsFeaturesEXT, "4444Formats"),
        CL(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT, "VertexAttributeDivisor"),
    };
#undef CL
#undef CLN

    struct DropInfo { uint32_t sType; const char* ext; };
    const DropInfo kDropIfExtMissing[] = {
        { static_cast<uint32_t>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT), "VK_EXT_mutable_descriptor_type" },
    };

    void ClampBools(VkBool32* dst, const VkBool32* real, size_t count, const char* label,
                    const char* const* fields, size_t fieldCount) {
        for (size_t k = 0; k < count; ++k) {
            const VkBool32 nv = (dst[k] && real[k]) ? VK_TRUE : VK_FALSE;
            if (dst[k] && !real[k]) {
                if (fields && k < fieldCount) LogLine("    CLEARED %s.%s (driver lacks it)", label, fields[k]);
                else LogLine("    CLEARED %s field#%u (driver lacks it)", label, static_cast<unsigned>(k));
            }
            dst[k] = nv;
        }
    }

    void ClampToReal(VkPhysicalDevice pd, VkBaseOutStructure* node, const ClampInfo& c) {
        auto realFn = RealExport<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2");
        if (!realFn) return;
        std::vector<uint64_t> buf((c.size + 7) / 8, 0);
        auto* r = reinterpret_cast<VkBaseOutStructure*>(buf.data());
        r->sType = node->sType;
        r->pNext = nullptr;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = r;
        realFn(pd, &f2);
        auto* dst = reinterpret_cast<VkBool32*>(reinterpret_cast<uint8_t*>(node) + 16);
        const auto* src = reinterpret_cast<const VkBool32*>(reinterpret_cast<const uint8_t*>(r) + 16);
        ClampBools(dst, src, (c.size - 16) / 4, c.name, c.fields, c.fieldCount);
    }

    void ClampCoreFeatures(VkPhysicalDevice pd, VkPhysicalDeviceFeatures* f, const char* label) {
        auto realFn = RealExport<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures");
        if (!realFn || !f) return;
        VkPhysicalDeviceFeatures real{};
        realFn(pd, &real);
        ClampBools(reinterpret_cast<VkBool32*>(f), reinterpret_cast<const VkBool32*>(&real),
                   sizeof(VkPhysicalDeviceFeatures) / 4, label, kNamesCore,
                   sizeof(kNamesCore) / sizeof(kNamesCore[0]));
    }

    // -----------------------------------------------------------------------
    // Dynamic-rendering -> VkRenderPass emulation
    // -----------------------------------------------------------------------
    namespace dynrend {
        // Vulkan: VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO = 1000044002.
        // 1000044001 is VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; using it here
        // made the proxy miss VkPipelineRenderingCreateInfo and pass a NULL renderPass
        // to the Vulkan 1.2 AMD driver, which cannot create these dynamic-rendering pipelines.
        constexpr uint32_t ST_PIPELINE_RENDERING_CREATE_INFO = 1000044002u;

        struct RPKey {
        std::vector<VkFormat> colorFormats;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        VkFormat stencilFormat = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        uint32_t viewMask = 0;
        std::vector<VkAttachmentLoadOp> colorLoadOps;
        VkAttachmentLoadOp depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkAttachmentLoadOp stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        std::vector<VkImageLayout> colorLayouts;
        VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkImageLayout stencilLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bool operator<(const RPKey& o) const {
            if (colorFormats != o.colorFormats) return colorFormats < o.colorFormats;
            if (depthFormat != o.depthFormat) return depthFormat < o.depthFormat;
            if (stencilFormat != o.stencilFormat) return stencilFormat < o.stencilFormat;
            if (samples != o.samples) return samples < o.samples;
            if (viewMask != o.viewMask) return viewMask < o.viewMask;
            if (colorLoadOps != o.colorLoadOps) return colorLoadOps < o.colorLoadOps;
            if (depthLoadOp != o.depthLoadOp) return depthLoadOp < o.depthLoadOp;
            if (stencilLoadOp != o.stencilLoadOp) return stencilLoadOp < o.stencilLoadOp;
            if (colorLayouts != o.colorLayouts) return colorLayouts < o.colorLayouts;
            if (depthLayout != o.depthLayout) return depthLayout < o.depthLayout;
            return stencilLayout < o.stencilLayout;
        }
    };
    struct PipeInfo {
            VkRenderPass renderPass = VK_NULL_HANDLE;
            VkDevice device = VK_NULL_HANDLE;
            std::vector<VkFormat> colorFormats;
            VkFormat depthFormat = VK_FORMAT_UNDEFINED;
            VkFormat stencilFormat = VK_FORMAT_UNDEFINED;
            VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
            uint32_t viewMask = 0;
        };

        struct ImageViewInfo {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect = 0;
        VkImage image = VK_NULL_HANDLE;
    };
    struct ActiveState {
            VkRenderPass renderPass = VK_NULL_HANDLE;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
        };

        struct FBKey {
            VkRenderPass rp = VK_NULL_HANDLE;
            std::vector<VkImageView> views;
            uint32_t width = 0, height = 0, layers = 0;
            bool operator<(const FBKey& o) const {
                if (rp != o.rp) return rp < o.rp;
                if (views != o.views) return views < o.views;
                if (width != o.width) return width < o.width;
                if (height != o.height) return height < o.height;
                return layers < o.layers;
            }
        };

        SRWLOCK g_lock = SRWLOCK_INIT;
        std::map<RPKey, VkRenderPass>* g_rpCache = nullptr;
        std::map<VkPipeline, PipeInfo>* g_pipes = nullptr;
        std::map<VkCommandBuffer, VkPipeline>* g_boundPipes = nullptr;     // GRAPHICS bind point only ([v3.2])
        std::map<VkCommandBuffer, VkPipeline>* g_boundCompute = nullptr;   // COMPUTE bind point ([v3.2])
        std::map<VkCommandBuffer, VkDevice>* g_cbDevices = nullptr;
        std::map<VkCommandBuffer, ActiveState>* g_active = nullptr;
        std::map<FBKey, VkFramebuffer>* g_fbCache = nullptr;
        std::map<VkImageView, ImageViewInfo>* g_imageViews = nullptr;
    std::map<VkImage, VkImageLayout>* g_imageLayouts = nullptr;
    void TrackImages(uint32_t n, const VkImageMemoryBarrier* b) {
        if (!n || !b) return;
        AcquireSRWLockExclusive(&g_lock);
        if (!g_imageLayouts) g_imageLayouts = new std::map<VkImage, VkImageLayout>();
        for (uint32_t i = 0; i < n; ++i)
            if (b[i].newLayout != VK_IMAGE_LAYOUT_UNDEFINED) (*g_imageLayouts)[b[i].image] = b[i].newLayout;
        ReleaseSRWLockExclusive(&g_lock);
    }
    VkImageLayout LayoutOf(VkImage img, VkImageLayout def) {
        VkImageLayout l = def;
        AcquireSRWLockShared(&g_lock);
        if (g_imageLayouts) { auto it = g_imageLayouts->find(img); if (it != g_imageLayouts->end()) l = it->second; }
        ReleaseSRWLockShared(&g_lock);
        return l;
    }

        VkRenderPass CreateOrGetRenderPass(VkDevice device, const RPKey& key) {
        AcquireSRWLockExclusive(&g_lock);
        if (!g_rpCache) g_rpCache = new std::map<RPKey, VkRenderPass>();
        auto it = g_rpCache->find(key);
        if (it != g_rpCache->end()) { VkRenderPass rp = it->second; ReleaseSRWLockExclusive(&g_lock); return rp; }
        ReleaseSRWLockExclusive(&g_lock);
        std::vector<VkAttachmentDescription> atts;
        std::vector<VkAttachmentReference> colorRefs;
        VkAttachmentReference dsRef{};
        const bool hasDepth = key.depthFormat != VK_FORMAT_UNDEFINED;
        const bool hasStencil = key.stencilFormat != VK_FORMAT_UNDEFINED;
        const bool hasDS = hasDepth || hasStencil;
        for (size_t i = 0; i < key.colorFormats.size(); ++i) {
            VkAttachmentReference ref{};
            if (key.colorFormats[i] == VK_FORMAT_UNDEFINED) {
                ref.attachment = VK_ATTACHMENT_UNUSED;
                ref.layout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorRefs.push_back(ref);
                continue;
            }
            const VkImageLayout lay = (i < key.colorLayouts.size()) ? key.colorLayouts[i] : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentLoadOp lo = (i < key.colorLoadOps.size()) ? key.colorLoadOps[i] : VK_ATTACHMENT_LOAD_OP_LOAD;
            VkAttachmentDescription ad{};
            ad.format = key.colorFormats[i];
            ad.samples = key.samples;
            ad.loadOp = lo;
            ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            ad.initialLayout = (lo == VK_ATTACHMENT_LOAD_OP_LOAD) ? lay : VK_IMAGE_LAYOUT_UNDEFINED;
            ad.finalLayout = lay;
            ref.attachment = static_cast<uint32_t>(atts.size());
            ref.layout = lay;
            atts.push_back(ad);
            colorRefs.push_back(ref);
        }
        if (hasDS) {
            const VkImageLayout lay = hasDepth ? key.depthLayout : key.stencilLayout;
            VkAttachmentDescription ad{};
            ad.format = hasDepth ? key.depthFormat : key.stencilFormat;
            ad.samples = key.samples;
            ad.loadOp = hasDepth ? key.depthLoadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            ad.storeOp = hasDepth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            ad.stencilLoadOp = hasStencil ? key.stencilLoadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            ad.stencilStoreOp = hasStencil ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            // [FIX] contents must be preserved when ANY aspect is loaded (was: all aspects, which threw away a loaded depth
            // buffer whenever the stencil aspect was cleared / not loaded).
            const bool keep = (hasDepth && ad.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) || (hasStencil && ad.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
            ad.initialLayout = keep ? lay : VK_IMAGE_LAYOUT_UNDEFINED;
            ad.finalLayout = lay;
            dsRef.attachment = static_cast<uint32_t>(atts.size());
            dsRef.layout = lay;
            atts.push_back(ad);
        }
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
        sub.pColorAttachments = colorRefs.empty() ? nullptr : colorRefs.data();
        sub.pDepthStencilAttachment = hasDS ? &dsRef : nullptr;
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[0].dstStageMask = deps[0].srcStageMask;
        deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = deps[0].srcStageMask;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        deps[1].srcAccessMask = deps[0].srcAccessMask;
        deps[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        VkRenderPassCreateInfo rpci{};
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.attachmentCount = static_cast<uint32_t>(atts.size());
        rpci.pAttachments = atts.empty() ? nullptr : atts.data();
        rpci.subpassCount = 1;
        rpci.pSubpasses = &sub;
        rpci.dependencyCount = 2;
        rpci.pDependencies = deps;
        // [FIX] viewMask (multiview) was silently ignored -> only one view was rendered.
        uint32_t mvMask = key.viewMask, mvCorr = key.viewMask;
        VkRenderPassMultiviewCreateInfo mv{ VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO };
        if (key.viewMask) {
            mv.subpassCount = 1; mv.pViewMasks = &mvMask;
            mv.correlationMaskCount = 1; mv.pCorrelationMasks = &mvCorr;
            rpci.pNext = &mv;
        }
        auto createRp = RealExport<PFN_vkCreateRenderPass>("vkCreateRenderPass");
        if (!createRp) return VK_NULL_HANDLE;
        VkRenderPass rp = VK_NULL_HANDLE;
        VkResult r = createRp(device, &rpci, nullptr, &rp);
        if (r != VK_SUCCESS) {
            LogLine("dynrend: vkCreateRenderPass FAILED result=%d (colors=%u depth=0x%X stencil=0x%X samples=%u)",
                static_cast<int>(r), static_cast<unsigned>(key.colorFormats.size()),
                key.depthFormat, key.stencilFormat, static_cast<unsigned>(key.samples));
            return VK_NULL_HANDLE;
        }
        AcquireSRWLockExclusive(&g_lock);
        (*g_rpCache)[key] = rp;
        ReleaseSRWLockExclusive(&g_lock);
        LogLine("dynrend: render pass created (colors=%u depth=0x%X stencil=0x%X samples=%u)",
            static_cast<unsigned>(key.colorFormats.size()), key.depthFormat, key.stencilFormat,
            static_cast<unsigned>(key.samples));
        return rp;
    }
    void RegisterPipeline(VkPipeline pipeline, VkDevice device,
                              const VkPipelineRenderingCreateInfo* ri,
                              VkSampleCountFlagBits samples) {
            if (!ri || !pipeline) return;
            PipeInfo pi;
            pi.device = device;
            for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i)
                pi.colorFormats.push_back(ri->pColorAttachmentFormats[i]);
            pi.depthFormat = ri->depthAttachmentFormat;
            pi.stencilFormat = ri->stencilAttachmentFormat;
            pi.samples = samples;
            pi.viewMask = ri->viewMask;

            RPKey k;
            k.colorFormats = pi.colorFormats;
            k.depthFormat = pi.depthFormat;
            k.stencilFormat = pi.stencilFormat;
            k.samples = samples;
            k.viewMask = pi.viewMask;
            for (size_t i = 0; i < pi.colorFormats.size(); ++i)
                k.colorLoadOps.push_back(VK_ATTACHMENT_LOAD_OP_LOAD);
        k.colorLayouts.push_back(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            k.depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            k.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

            pi.renderPass = CreateOrGetRenderPass(device, k);
            if (pi.renderPass == VK_NULL_HANDLE) return;

            AcquireSRWLockExclusive(&g_lock);
            if (!g_pipes) g_pipes = new std::map<VkPipeline, PipeInfo>();
            (*g_pipes)[pipeline] = pi;
            ReleaseSRWLockExclusive(&g_lock);
            LogLine("dynrend: pipeline %p registered (rp=%p)", pipeline, pi.renderPass);
        }

        void UnregisterPipeline(VkPipeline p) {
            AcquireSRWLockExclusive(&g_lock);
            if (g_pipes) g_pipes->erase(p);
            ReleaseSRWLockExclusive(&g_lock);
        }

        bool LookupImageView(VkImageView v, ImageViewInfo* out) {
            AcquireSRWLockShared(&g_lock);
            bool found = false;
            if (g_imageViews) {
                auto it = g_imageViews->find(v);
                if (it != g_imageViews->end()) { *out = it->second; found = true; }
            }
            ReleaseSRWLockShared(&g_lock);
            return found;
        }

        VkDevice LookupCBDevice(VkCommandBuffer cb) {
            AcquireSRWLockShared(&g_lock);
            VkDevice d = VK_NULL_HANDLE;
            if (g_cbDevices) {
                auto it = g_cbDevices->find(cb);
                if (it != g_cbDevices->end()) d = it->second;
            }
            ReleaseSRWLockShared(&g_lock);
            return d;
        }

        bool LookupBoundPipeline(VkCommandBuffer cb, PipeInfo* out) {
            AcquireSRWLockShared(&g_lock);
            bool found = false;
            if (g_boundPipes && g_pipes) {
                auto bit = g_boundPipes->find(cb);
                if (bit != g_boundPipes->end()) {
                    auto pit = g_pipes->find(bit->second);
                    if (pit != g_pipes->end()) { *out = pit->second; found = true; }
                }
            }
            ReleaseSRWLockShared(&g_lock);
            return found;
        }
    }


    // ---------------------------------------------------------------------------
// v4: sync2 -> sync1 translation helpers
// ---------------------------------------------------------------------------
namespace {
    VkPipelineStageFlags ConvertStage2(VkPipelineStageFlags2 s2) {
    if (s2 == 0) return 0;
    constexpr VkPipelineStageFlags kKnown =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags out = static_cast<VkPipelineStageFlags>(s2) & kKnown;
    if (s2 & (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
              VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT))
        out |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (s2 & (VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT))
        out |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    if (s2 & VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT)
        out |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
               VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    if (s2 & (0x00080000ull | 0x00100000ull | 0x00200000ull | 0x00400000ull | 0x02000000ull))
        out |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT; // [FIX] RT/AS/mesh-стадии эмулируются compute/transfer
    if (out == 0) out = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    return out;
}
VkAccessFlags ConvertAccess2(VkAccessFlags2 a2) {
    if (a2 == 0) return 0;
    VkAccessFlags out = static_cast<VkAccessFlags>(a2) & 0x1FFFFu;
    if (a2 & ((1ull << 32) | (1ull << 33))) out |= VK_ACCESS_SHADER_READ_BIT;
    if (a2 & (1ull << 34)) out |= VK_ACCESS_SHADER_WRITE_BIT;
    if (a2 & ~(0x1FFFFull | (7ull << 32))) out |= VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (out == 0) out = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    return out;
}
// [FIX] Vulkan 1.3 / VK_KHR_synchronization2 layouts do not exist on a 1.2 driver. A game that uses
// vkCmdPipelineBarrier2 / dynamic rendering passes ATTACHMENT_OPTIMAL / READ_ONLY_OPTIMAL, which the old driver
// treats as garbage -> black / corrupted render targets. Map them to the legacy equivalents.
VkImageLayout MapLayout2(VkImageLayout l, VkImageAspectFlags aspect) {
    const bool ds = (aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
    switch (static_cast<int>(l)) {
    case 1000314000: return ds ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;       // READ_ONLY_OPTIMAL
    case 1000314001: return ds ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;      // ATTACHMENT_OPTIMAL
    default: return l;
    }
}
}
// ---------------------------------------------------------------------------// v4: sync2 implementations (translate to sync1)
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence) {
    auto real = RealExport<PFN_vkQueueSubmit>("vkQueueSubmit");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (submitCount == 0 || !pSubmits) {
        VkResult sr = real(queue, 0, nullptr, fence);
        InterlockedIncrement(&g_submitCount);
        InterlockedExchange(&g_lastSubmitRes, static_cast<LONG>(sr));
        InterlockedExchange64(&g_lastSubmitTick, static_cast<LONG64>(GetTickCount64()));
        if (sr != VK_SUCCESS) LogLine("vkQueueSubmit2 (empty) FAILED: result=%d", (int)sr);
        return sr;
    }

    std::vector<VkSubmitInfo> si(submitCount);
    // [FIX] The old translation dropped VkSemaphoreSubmitInfo::value. Timeline semaphores then never reached the
    // value the game waits for (vkWaitSemaphores / next submit) -> the game waited forever on a black screen.
    std::vector<VkTimelineSemaphoreSubmitInfo> tl(submitCount);
    std::vector<std::vector<uint64_t>> waitVal(submitCount), sigVal(submitCount);
    std::vector<std::vector<VkSemaphore>> waitSem(submitCount);
    std::vector<std::vector<VkPipelineStageFlags>> waitStages(submitCount);
    std::vector<std::vector<VkSemaphore>> sigSem(submitCount);
    std::vector<std::vector<VkCommandBuffer>> cmds(submitCount);

    for (uint32_t i = 0; i < submitCount; ++i) {
        const auto& s2 = pSubmits[i];
        waitSem[i].resize(s2.waitSemaphoreInfoCount);
        waitStages[i].resize(s2.waitSemaphoreInfoCount);
        waitVal[i].resize(s2.waitSemaphoreInfoCount);
        for (uint32_t j = 0; j < s2.waitSemaphoreInfoCount; ++j) {
            waitSem[i][j] = s2.pWaitSemaphoreInfos[j].semaphore;
            waitVal[i][j] = s2.pWaitSemaphoreInfos[j].value;
            VkPipelineStageFlags st = ConvertStage2(s2.pWaitSemaphoreInfos[j].stageMask);
            if (st == 0) st = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;   // sync1 forbids an empty wait stage mask
            waitStages[i][j] = st;
        }
        sigSem[i].resize(s2.signalSemaphoreInfoCount);
        sigVal[i].resize(s2.signalSemaphoreInfoCount);
        for (uint32_t j = 0; j < s2.signalSemaphoreInfoCount; ++j) {
            sigSem[i][j] = s2.pSignalSemaphoreInfos[j].semaphore;
            sigVal[i][j] = s2.pSignalSemaphoreInfos[j].value;
        }
        cmds[i].resize(s2.commandBufferInfoCount);
        for (uint32_t j = 0; j < s2.commandBufferInfoCount; ++j)
            cmds[i][j] = s2.pCommandBufferInfos[j].commandBuffer;

        tl[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tl[i].pNext = nullptr;
        tl[i].waitSemaphoreValueCount = s2.waitSemaphoreInfoCount;
        tl[i].pWaitSemaphoreValues = waitVal[i].empty() ? nullptr : waitVal[i].data();
        tl[i].signalSemaphoreValueCount = s2.signalSemaphoreInfoCount;
        tl[i].pSignalSemaphoreValues = sigVal[i].empty() ? nullptr : sigVal[i].data();

        si[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si[i].pNext = &tl[i];   // values of binary semaphores are ignored by the driver
        si[i].waitSemaphoreCount = s2.waitSemaphoreInfoCount;
        si[i].pWaitSemaphores = waitSem[i].empty() ? nullptr : waitSem[i].data();
        si[i].pWaitDstStageMask = waitStages[i].empty() ? nullptr : waitStages[i].data();
        si[i].commandBufferCount = s2.commandBufferInfoCount;
        si[i].pCommandBuffers = cmds[i].empty() ? nullptr : cmds[i].data();
        si[i].signalSemaphoreCount = s2.signalSemaphoreInfoCount;
        si[i].pSignalSemaphores = sigSem[i].empty() ? nullptr : sigSem[i].data();
    }
    if (FirstTime("sync2:queueSubmit2")) LogLine("SYNC2: vkQueueSubmit2 -> vkQueueSubmit (+timeline values)");
    const VkResult sr = real(queue, submitCount, si.data(), fence);
    // [v3.2] a lost device / failed submit used to be invisible in the log
    InterlockedIncrement(&g_submitCount);
    InterlockedExchange(&g_lastSubmitRes, static_cast<LONG>(sr));
    InterlockedExchange64(&g_lastSubmitTick, static_cast<LONG64>(GetTickCount64()));
    if (sr != VK_SUCCESS) {
        static volatile LONG badSubmit = 0;
        const LONG n = InterlockedIncrement(&badSubmit);
        if (n <= 20 || n % 500 == 0)
            LogLine("!!! vkQueueSubmit2 FAILED: result=%d%s (failures so far=%ld)", static_cast<int>(sr),
                sr == VK_ERROR_DEVICE_LOST ? " VK_ERROR_DEVICE_LOST (GPU hang / driver reset)" : "", static_cast<long>(n));
    }
    return sr;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit2KHR(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence) {
    return vkQueueSubmit2(queue, submitCount, pSubmits, fence);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* pInfo) {
    if (!pInfo) return;
    auto real = RealExport<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    if (!real) return;

    VkPipelineStageFlags srcStage = 0, dstStage = 0;
    std::vector<VkMemoryBarrier> mem(pInfo->memoryBarrierCount);
    for (uint32_t i = 0; i < pInfo->memoryBarrierCount; ++i) {
        const auto& s = pInfo->pMemoryBarriers[i];
        srcStage |= ConvertStage2(s.srcStageMask);
        dstStage |= ConvertStage2(s.dstStageMask);
        mem[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mem[i].srcAccessMask = ConvertAccess2(s.srcAccessMask);
        mem[i].dstAccessMask = ConvertAccess2(s.dstAccessMask);
    }
    std::vector<VkBufferMemoryBarrier> buf(pInfo->bufferMemoryBarrierCount);
    for (uint32_t i = 0; i < pInfo->bufferMemoryBarrierCount; ++i) {
        const auto& s = pInfo->pBufferMemoryBarriers[i];
        srcStage |= ConvertStage2(s.srcStageMask);
        dstStage |= ConvertStage2(s.dstStageMask);
        buf[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buf[i].srcAccessMask = ConvertAccess2(s.srcAccessMask);
        buf[i].dstAccessMask = ConvertAccess2(s.dstAccessMask);
        buf[i].srcQueueFamilyIndex = s.srcQueueFamilyIndex;
        buf[i].dstQueueFamilyIndex = s.dstQueueFamilyIndex;
        buf[i].buffer = s.buffer;
        buf[i].offset = s.offset;
        buf[i].size = s.size;
    }
    std::vector<VkImageMemoryBarrier> img(pInfo->imageMemoryBarrierCount);
    for (uint32_t i = 0; i < pInfo->imageMemoryBarrierCount; ++i) {
        const auto& s = pInfo->pImageMemoryBarriers[i];
        srcStage |= ConvertStage2(s.srcStageMask);
        dstStage |= ConvertStage2(s.dstStageMask);
        img[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img[i].srcAccessMask = ConvertAccess2(s.srcAccessMask);
        img[i].dstAccessMask = ConvertAccess2(s.dstAccessMask);
        img[i].oldLayout = MapLayout2(s.oldLayout, s.subresourceRange.aspectMask);
        img[i].newLayout = MapLayout2(s.newLayout, s.subresourceRange.aspectMask);
        img[i].srcQueueFamilyIndex = s.srcQueueFamilyIndex;
        img[i].dstQueueFamilyIndex = s.dstQueueFamilyIndex;
        img[i].image = s.image;
        img[i].subresourceRange = s.subresourceRange;
    }
    // [FIX] sync1 requires non-empty stage masks (sync2 allows NONE).
    if (!srcStage) srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (!dstStage) dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dynrend::TrackImages((uint32_t)img.size(), img.empty() ? nullptr : img.data());
    rtfill::TrackBarriers((uint32_t)img.size(), img.empty() ? nullptr : img.data()); // [RTFILL]
if (FirstTime("sync2:cmdPipelineBarrier2")) LogLine("SYNC2: vkCmdPipelineBarrier2 -> vkCmdPipelineBarrier");
    real(cb, srcStage, dstStage, pInfo->dependencyFlags,
         pInfo->memoryBarrierCount, mem.empty() ? nullptr : mem.data(),
         pInfo->bufferMemoryBarrierCount, buf.empty() ? nullptr : buf.data(),
         pInfo->imageMemoryBarrierCount, img.empty() ? nullptr : img.data());
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* pInfo) {
    vkCmdPipelineBarrier2(cb, pInfo);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t query) {
    auto real = RealExport<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp");
    if (real) { VkPipelineStageFlags st = ConvertStage2(stage); if (!st) st = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; real(cb, static_cast<VkPipelineStageFlagBits>(st), pool, query); }
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdWriteTimestamp2KHR(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t query) {
    vkCmdWriteTimestamp2(cb, stage, pool, query);
}

// ---------------------------------------------------------------------------
// ---- FIX: full sync2 / 1.3 command coverage ----
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* p) {
    auto real = RealExport<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    if (!real || !p) return;
    std::vector<VkBufferCopy> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i)
        r[i] = { p->pRegions[i].srcOffset, p->pRegions[i].dstOffset, p->pRegions[i].size };
    if (FirstTime("sync2:CopyBuffer2")) LogLine("SYNC2: vkCmdCopyBuffer2 -> vkCmdCopyBuffer");
    real(cb, p->srcBuffer, p->dstBuffer, p->regionCount, r.empty() ? nullptr : r.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer2KHR(VkCommandBuffer cb, const VkCopyBufferInfo2* p) { vkCmdCopyBuffer2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* p) {
    auto real = RealExport<PFN_vkCmdCopyImage>("vkCmdCopyImage");
    if (!real || !p) return;
    std::vector<VkImageCopy> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i) {
        const auto& s = p->pRegions[i];
        r[i] = { s.srcSubresource, s.srcOffset, s.dstSubresource, s.dstOffset, s.extent };
    }
    if (FirstTime("sync2:CopyImage2")) LogLine("SYNC2: vkCmdCopyImage2 -> vkCmdCopyImage");
    real(cb, p->srcImage, p->srcImageLayout, p->dstImage, p->dstImageLayout, p->regionCount, r.empty() ? nullptr : r.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage2KHR(VkCommandBuffer cb, const VkCopyImageInfo2* p) { vkCmdCopyImage2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* p) {
    auto real = RealExport<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    if (!real || !p) return;
    std::vector<VkBufferImageCopy> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i) {
        const auto& s = p->pRegions[i];
        r[i] = { s.bufferOffset, s.bufferRowLength, s.bufferImageHeight, s.imageSubresource, s.imageOffset, s.imageExtent };
    }
    if (FirstTime("sync2:CopyBufferToImage2")) LogLine("SYNC2: vkCmdCopyBufferToImage2 -> vkCmdCopyBufferToImage");
    real(cb, p->srcBuffer, p->dstImage, p->dstImageLayout, p->regionCount, r.empty() ? nullptr : r.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage2KHR(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* p) { vkCmdCopyBufferToImage2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* p) {
    auto real = RealExport<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    if (!real || !p) return;
    std::vector<VkBufferImageCopy> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i) {
        const auto& s = p->pRegions[i];
        r[i] = { s.bufferOffset, s.bufferRowLength, s.bufferImageHeight, s.imageSubresource, s.imageOffset, s.imageExtent };
    }
    real(cb, p->srcImage, p->srcImageLayout, p->dstBuffer, p->regionCount, r.empty() ? nullptr : r.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer2KHR(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* p) { vkCmdCopyImageToBuffer2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* p) {
    auto real = RealExport<PFN_vkCmdBlitImage>("vkCmdBlitImage");
    if (!real || !p) return;
    std::vector<VkImageBlit> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i) {
        const auto& s = p->pRegions[i];
        r[i] = { s.srcSubresource, { s.srcOffsets[0], s.srcOffsets[1] }, s.dstSubresource, { s.dstOffsets[0], s.dstOffsets[1] } };
    }
    if (FirstTime("sync2:BlitImage2")) LogLine("SYNC2: vkCmdBlitImage2 -> vkCmdBlitImage");
    real(cb, p->srcImage, p->srcImageLayout, p->dstImage, p->dstImageLayout, p->regionCount, r.empty() ? nullptr : r.data(), p->filter);
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage2KHR(VkCommandBuffer cb, const VkBlitImageInfo2* p) { vkCmdBlitImage2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* p) {
    auto real = RealExport<PFN_vkCmdResolveImage>("vkCmdResolveImage");
    if (!real || !p) return;
    std::vector<VkImageResolve> r(p->regionCount);
    for (uint32_t i = 0; i < p->regionCount; ++i) {
        const auto& s = p->pRegions[i];
        r[i] = { s.srcSubresource, s.srcOffset, s.dstSubresource, s.dstOffset, s.extent };
    }
    real(cb, p->srcImage, p->srcImageLayout, p->dstImage, p->dstImageLayout, p->regionCount, r.empty() ? nullptr : r.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage2KHR(VkCommandBuffer cb, const VkResolveImageInfo2* p) { vkCmdResolveImage2(cb, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdSetEvent2(VkCommandBuffer cb, VkEvent ev, const VkDependencyInfo* p) {
    auto real = RealExport<PFN_vkCmdSetEvent>("vkCmdSetEvent");
    if (!real) return;
    VkPipelineStageFlags s = 0;
    if (p) {
        for (uint32_t i = 0; i < p->memoryBarrierCount; ++i) s |= ConvertStage2(p->pMemoryBarriers[i].srcStageMask);
        for (uint32_t i = 0; i < p->bufferMemoryBarrierCount; ++i) s |= ConvertStage2(p->pBufferMemoryBarriers[i].srcStageMask);
        for (uint32_t i = 0; i < p->imageMemoryBarrierCount; ++i) s |= ConvertStage2(p->pImageMemoryBarriers[i].srcStageMask);
    }
    if (!s) s = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (FirstTime("sync2:SetEvent2")) LogLine("SYNC2: vkCmdSetEvent2 -> vkCmdSetEvent");
    real(cb, ev, s);
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent2KHR(VkCommandBuffer cb, VkEvent ev, const VkDependencyInfo* p) { vkCmdSetEvent2(cb, ev, p); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdResetEvent2(VkCommandBuffer cb, VkEvent ev, VkPipelineStageFlags2 stage) {
    auto real = RealExport<PFN_vkCmdResetEvent>("vkCmdResetEvent");
    if (real) { VkPipelineStageFlags st = ConvertStage2(stage); if (!st) st = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; real(cb, ev, st); }
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent2KHR(VkCommandBuffer cb, VkEvent ev, VkPipelineStageFlags2 s) { vkCmdResetEvent2(cb, ev, s); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdWaitEvents2(VkCommandBuffer cb, uint32_t n, const VkEvent* evs, const VkDependencyInfo* deps) {
    auto real = RealExport<PFN_vkCmdWaitEvents>("vkCmdWaitEvents");
    if (!real || !n || !evs) return;
    VkPipelineStageFlags src = 0, dst = 0;
    std::vector<VkMemoryBarrier> mem; std::vector<VkBufferMemoryBarrier> buf; std::vector<VkImageMemoryBarrier> img;
    for (uint32_t d = 0; d < n && deps; ++d) {
        const auto& p = deps[d];
        for (uint32_t i = 0; i < p.memoryBarrierCount; ++i) {
            const auto& s = p.pMemoryBarriers[i];
            src |= ConvertStage2(s.srcStageMask); dst |= ConvertStage2(s.dstStageMask);
            VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, ConvertAccess2(s.srcAccessMask), ConvertAccess2(s.dstAccessMask) };
            mem.push_back(b);
        }
        for (uint32_t i = 0; i < p.bufferMemoryBarrierCount; ++i) {
            const auto& s = p.pBufferMemoryBarriers[i];
            src |= ConvertStage2(s.srcStageMask); dst |= ConvertStage2(s.dstStageMask);
            VkBufferMemoryBarrier b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, ConvertAccess2(s.srcAccessMask), ConvertAccess2(s.dstAccessMask),
                s.srcQueueFamilyIndex, s.dstQueueFamilyIndex, s.buffer, s.offset, s.size };
            buf.push_back(b);
        }
        for (uint32_t i = 0; i < p.imageMemoryBarrierCount; ++i) {
            const auto& s = p.pImageMemoryBarriers[i];
            src |= ConvertStage2(s.srcStageMask); dst |= ConvertStage2(s.dstStageMask);
            VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, ConvertAccess2(s.srcAccessMask), ConvertAccess2(s.dstAccessMask),
                MapLayout2(s.oldLayout, s.subresourceRange.aspectMask), MapLayout2(s.newLayout, s.subresourceRange.aspectMask), s.srcQueueFamilyIndex, s.dstQueueFamilyIndex, s.image, s.subresourceRange };
            img.push_back(b);
        }
    }
    if (!src) src = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (!dst) dst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (FirstTime("sync2:WaitEvents2")) LogLine("SYNC2: vkCmdWaitEvents2 -> vkCmdWaitEvents");
    dynrend::TrackImages((uint32_t)img.size(), img.empty() ? nullptr : img.data());
    real(cb, n, evs, src, dst,
        (uint32_t)mem.size(), mem.empty() ? nullptr : mem.data(),
        (uint32_t)buf.size(), buf.empty() ? nullptr : buf.data(),
        (uint32_t)img.size(), img.empty() ? nullptr : img.data());
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents2KHR(VkCommandBuffer cb, uint32_t n, const VkEvent* e, const VkDependencyInfo* d) { vkCmdWaitEvents2(cb, n, e, d); }
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t count,
    const VkBuffer* bufs, const VkDeviceSize* offs, const VkDeviceSize* sizes, const VkDeviceSize* strides) {
    // [FIX] The system loader exports vkCmdBindVertexBuffers2 even when the 1.2 driver does not implement it: the
    // exported trampoline then jumps through a NULL dispatch slot (EXEC@0). Use the real VK_EXT_extended_dynamic_state
    // function when it was enabled, otherwise fall back to the legacy call (strides come from the pipeline).
    {
        static PFN_vkCmdBindVertexBuffers2EXT ext = nullptr;
        static volatile LONG resolved = 0;
        if (g_edsEnabled && !resolved && g_lastDevice && g_realGdpa) {
            ext = reinterpret_cast<PFN_vkCmdBindVertexBuffers2EXT>(g_realGdpa(g_lastDevice, "vkCmdBindVertexBuffers2EXT"));
            InterlockedExchange(&resolved, 1);
        }
        if (ext) { ext(cb, first, count, bufs, offs, sizes, strides); return; }
    }
    auto real = RealExport<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    if (real) real(cb, first, count, bufs, offs);
}

// [FIX] Адаптер vkCmdBindIndexBuffer2KHR -> vkCmdBindIndexBuffer (разный порядок аргументов size/indexType).
static VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer2Adapter(
    VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize, VkIndexType indexType) {
    auto real = RealExport<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    if (real) real(cb, buffer, offset, indexType);
}
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier(VkCommandBuffer cb,
    VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
    uint32_t mc, const VkMemoryBarrier* mb, uint32_t bc, const VkBufferMemoryBarrier* bb,
    uint32_t ic, const VkImageMemoryBarrier* ib) {
    dynrend::TrackImages(ic, ib);
    rtfill::TrackBarriers(ic, ib); // [RTFILL]
    auto real = RealExport<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    if (real) real(cb, src, dst, flags, mc, mb, bc, bb, ic, ib);
}
namespace pdata {
    SRWLOCK lock = SRWLOCK_INIT;
    std::map<uint64_t, std::map<uint64_t, uint64_t>>* slots = nullptr;
    inline uint64_t K(VkPrivateDataSlot s) { return (uint64_t)(uintptr_t)s; }
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreatePrivateDataSlot(VkDevice, const VkPrivateDataSlotCreateInfo*, const VkAllocationCallbacks*, VkPrivateDataSlot* p) {
    if (!p) return VK_ERROR_INITIALIZATION_FAILED;
    *p = MakeFake<VkPrivateDataSlot>();
    AcquireSRWLockExclusive(&pdata::lock);
    if (!pdata::slots) pdata::slots = new std::map<uint64_t, std::map<uint64_t, uint64_t>>();
    (*pdata::slots)[pdata::K(*p)];
    ReleaseSRWLockExclusive(&pdata::lock);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyPrivateDataSlot(VkDevice, VkPrivateDataSlot s, const VkAllocationCallbacks*) {
    AcquireSRWLockExclusive(&pdata::lock);
    if (pdata::slots) pdata::slots->erase(pdata::K(s));
    ReleaseSRWLockExclusive(&pdata::lock);
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkSetPrivateData(VkDevice, VkObjectType, uint64_t obj, VkPrivateDataSlot s, uint64_t data) {
    AcquireSRWLockExclusive(&pdata::lock);
    if (pdata::slots) (*pdata::slots)[pdata::K(s)][obj] = data;
    ReleaseSRWLockExclusive(&pdata::lock);
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPrivateData(VkDevice, VkObjectType, uint64_t obj, VkPrivateDataSlot s, uint64_t* pData) {
    uint64_t v = 0;
    AcquireSRWLockShared(&pdata::lock);
    if (pdata::slots) {
        auto it = pdata::slots->find(pdata::K(s));
        if (it != pdata::slots->end()) { auto j = it->second.find(obj); if (j != it->second.end()) v = j->second; }
    }
    ReleaseSRWLockShared(&pdata::lock);
    if (pData) *pData = v;
}
static VKAPI_ATTR void VKAPI_CALL Stub_NoopCmd() {}
static VKAPI_ATTR VkResult VKAPI_CALL Stub_ReturnSuccess() { return VK_SUCCESS; }   // [v3.2]
// v7: SPIR-V analysis, crash tracer, driver protection
// ---------------------------------------------------------------------------
namespace {
    // ---- config (VulkanProxy.ini next to the dll, [spirv] section) ----------
    struct SpvConfig {
        // Defaults for GCN2 (R7 260): more conservative so the game reaches the menu.
        int denyFloat16 = 1, denyInt16 = 1, denyDotProduct = 1, denyRayTracing = 1, denyMesh = 1;
        int patchVersion = 1;   // rewrite SPIR-V 1.6 -> 1.5 (driver supports up to 1.5)
        int logAll = 0;         // log every shader, not just unusual ones
        int swrtForceFeatures = 0; // [SWRT] force shaderInt64/bufferDeviceAddress/scalarBlockLayout into vkCreateDevice
        int swrtPatchBuffers = 1;  // [SWRT] strip RT usage bits from vkCreateBuffer
        int softwareRT = 1;        // [SWRT] emulate ray queries with a software BVH traversal (falls back to stripping)
        int rewriteRayQuery = 1;   // strip ray queries from shaders (no-hit semantics) instead of denying them
        int dumpRewrite = 0;       // dump the first N rewritten shaders (input + output)
        int dumpRewriteFail = 3;   // dump the first N shaders the rewriter could not handle
        int useReplacements = 1;   // load replacement shaders from spv_replace\\spv_<hash>.spv
        float brightness = 1.0f;   // post-process gain on the final image (1.0 = off)
        int tryVariants = 1;       // if the driver crashes/rejects a module, retry with workaround variants
        int keepCode = 1;          // remember each module's SPIR-V so failed pipelines can be dumped
        int dumpSamples = 0;       // dump N working compute shaders of two kinds (for comparison with crashing ones)
        int serializeCompile = -1; // 1 = one driver call at a time, 0 = unlimited, -1 (default) = auto (2..4 parallel)
        int compileThreads = 0;    // >0 overrides the number of parallel driver compile calls
        int compileTimeoutSec = 60; // [HANGFIX] a pipeline compile longer than this is ABANDONED (FAKE pipeline); 0 = off
        int maxAbandoned = 6;      // [HANGFIX] after this many stuck compiles, big shaders are faked without trying
        int heavyShaderWords = 0;  // [HANGFIX] >0: shaders with more SPIR-V words than this are never sent to the driver
        int watchdog = 1;          // [HANGFIX] watchdog thread: heartbeat + in-flight report + periodic cache save
        int heavyComputeWords = 75000; // [v3.2] compute stage with >= this many SPIR-V words is never sent to the driver (0 = off)
        int computeTimeoutSec = 12;    // [v3.2] compute compile longer than this is abandoned (never succeeded after 3 s in the log)
        int pinStuckToOneCore = 1;     // [v3.2] abandoned (stuck) compile threads share ONE core
        int stallDumpSec = 20;         // [v3.2] no vkQueuePresentKHR for this long -> dump all thread stacks (0 = off)
        int pipelineCache = 1;     // 1 = proxy keeps a persistent VkPipelineCache (VulkanProxy_pipeline.cache)
        int promoteFloat16 = 0;   // старый промоутер - ВЫКЛ (ломает layout буферов)
        int lowerFp16 = 0;        // ВЫКЛ: заменяем на комбинированный проход
        int lowerFp16AndInt16 = 1; // новый: fp16 + int16 ALU -> 32 бит
        int catchDriverCrash = 1; // catch access violations inside the driver and fall back to FAKE objects
		int dropF16Cap = 0;       // 1 = ещё и убрать OpCapability Float16 после понижения
        uint32_t denyComputeCaps[16] = {}; int nDenyComputeCaps = 0;  // deny COMPUTE shaders that use any of these capabilities
        uint32_t denyAnyCaps[16] = {};     int nDenyAnyCaps = 0;      // deny ANY shader that uses any of these capabilities
        uint32_t denyComputeAll[16] = {};  int nDenyComputeAll = 0;   // deny COMPUTE shaders that use ALL of these capabilities together
    };
    SpvConfig g_spvCfg;
    volatile LONG g_spvCfgLoaded = 0;

    // ---- replacement shaders: spv_replace\\spv_<fnv1a64 of the ORIGINAL module>.spv ------------------
    inline uint64_t Fnv1a64(const void* data, size_t bytes) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
        return h;
    }
    std::unordered_map<uint64_t, std::wstring>* g_replMap = nullptr;
    void ScanReplacements(const wchar_t* iniPath) {
        wchar_t dir[2048]{};
        wcsncpy_s(dir, iniPath, _TRUNCATE);
        wchar_t* slash = wcsrchr(dir, L'\\');
        if (!slash) return;
        *(slash + 1) = 0;
        wchar_t pattern[2200]{};
        _snwprintf_s(pattern, 2200, _TRUNCATE, L"%lsspv_replace\\spv_*.spv", dir);
        WIN32_FIND_DATAW fd{};
        HANDLE h = FindFirstFileW(pattern, &fd);
        if (h == INVALID_HANDLE_VALUE) return;
        g_replMap = new std::unordered_map<uint64_t, std::wstring>();
        do {
            // name: spv_<16 hex digits>.spv
            const wchar_t* n = fd.cFileName;
            if (wcslen(n) < 4 + 16) continue;
            wchar_t hex[17]{};
            wcsncpy_s(hex, n + 4, 16);
            wchar_t* e = nullptr;
            const unsigned long long v = wcstoull(hex, &e, 16);
            if (e != hex + 16) continue;
            wchar_t full[2400]{};
            _snwprintf_s(full, 2400, _TRUNCATE, L"%lsspv_replace\\%ls", dir, n);
            (*g_replMap)[v] = full;
        } while (FindNextFileW(h, &fd));
        FindClose(h);
        LogLine("SPIRV: %zu replacement shader file(s) found in spv_replace", g_replMap->size());
    }
    bool ReadReplacement(const std::wstring& path, std::vector<uint32_t>& out) {
        HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz{};
        bool ok = false;
        if (GetFileSizeEx(f, &sz) && sz.QuadPart > 20 && sz.QuadPart < (64ll << 20) && (sz.QuadPart % 4) == 0) {
            out.resize(static_cast<size_t>(sz.QuadPart / 4));
            DWORD got = 0;
            ok = ReadFile(f, out.data(), static_cast<DWORD>(sz.QuadPart), &got, nullptr) && got == sz.QuadPart && out[0] == 0x07230203u;
        }
        CloseHandle(f);
        if (!ok) out.clear();
        return ok;
    }

    void LoadSpvConfigImpl() {
        wchar_t path[2048]{};
        GetLogPath(path, sizeof(path) / sizeof(path[0]));
        wchar_t* slash = wcsrchr(path, L'\\');
        if (slash) wcscpy_s(slash + 1, (sizeof(path) / sizeof(path[0])) - static_cast<size_t>(slash + 1 - path), L"VulkanProxy.ini");
        else wcscpy_s(path, sizeof(path) / sizeof(path[0]), L"VulkanProxy.ini");
        auto rd = [&](const wchar_t* k, int d) { return static_cast<int>(GetPrivateProfileIntW(L"spirv", k, d, path)); };
        g_spvCfg.denyFloat16    = rd(L"DenyFloat16", 1);
        g_spvCfg.denyInt16      = rd(L"DenyInt16", 1);
        g_spvCfg.denyDotProduct = rd(L"DenyDotProduct", 1);
        g_spvCfg.denyRayTracing = rd(L"DenyRayTracing", 1);
        g_spvCfg.denyMesh       = rd(L"DenyMesh", 1);
        g_spvCfg.patchVersion   = rd(L"PatchVersion", 1);
        g_spvCfg.logAll         = rd(L"LogAll", 0);
        g_spvCfg.serializeCompile = rd(L"SerializeCompile", -1);
        g_spvCfg.compileThreads = rd(L"CompileThreads", 0);
        g_spvCfg.compileTimeoutSec = rd(L"CompileTimeoutSec", 60);
        g_spvCfg.maxAbandoned = rd(L"MaxAbandoned", 6);
        g_spvCfg.heavyShaderWords = rd(L"HeavyShaderWords", 0);
        g_spvCfg.watchdog = rd(L"Watchdog", 1);
        g_spvCfg.heavyComputeWords = rd(L"HeavyComputeWords", 75000);
        g_spvCfg.computeTimeoutSec = rd(L"ComputeTimeoutSec", 12);
        g_spvCfg.pinStuckToOneCore = rd(L"PinStuckThreads", 1);
        g_spvCfg.stallDumpSec = rd(L"StallDumpSec", 20);
        g_spvCfg.pipelineCache = rd(L"PipelineCache", 1);
        g_spvCfg.keepCode = rd(L"KeepShaderCode", 1);
        g_spvCfg.tryVariants = rd(L"TryVariants", 1);
        {
            wchar_t bb[64]{};
            GetPrivateProfileStringW(L"spirv", L"BrightnessGain", L"1.0", bb, 64, path);
            const double g = wcstod(bb, nullptr);
            g_spvCfg.brightness = (g > 1.0 && g <= 8.0) ? static_cast<float>(g) : 1.0f;
        }
        g_spvCfg.useReplacements = rd(L"UseReplacements", 1);
        g_spvCfg.dumpSamples = rd(L"DumpSamples", 0);
        g_spvCfg.rewriteRayQuery = rd(L"RewriteRayQuery", 1);
        g_spvCfg.softwareRT = rd(L"SoftwareRT", 1);
        g_spvCfg.swrtForceFeatures = rd(L"SwrtForceFeatures", 0);
        g_spvCfg.swrtPatchBuffers = rd(L"SwrtPatchBuffers", 1);
        g_spvCfg.dumpRewrite = rd(L"DumpRewrite", 0);
        g_spvCfg.dumpRewriteFail = rd(L"DumpRewriteFail", 3);
        InterlockedExchange(&g_logVerbose, rd(L"Verbose", 0));
        g_spvCfg.catchDriverCrash = rd(L"CatchDriverCrash", 1);
        g_spvCfg.promoteFloat16 = rd(L"PromoteFloat16", 0);
        g_spvCfg.lowerFp16 = rd(L"LowerFp16", 0);
        g_spvCfg.lowerFp16AndInt16 = rd(L"LowerFp16AndInt16", 1);
		g_spvCfg.dropF16Cap = rd(L"DropF16Cap", 0);
        auto rdList = [&](const wchar_t* key, uint32_t* out, int& n) {
            wchar_t buf[256]{};
            GetPrivateProfileStringW(L"spirv", key, L"", buf, 256, path);
            n = 0;
            for (wchar_t* p = buf; *p && n < 16;) {
                while (*p == L' ' || *p == L',') ++p;
                if (!*p) break;
                wchar_t* e = nullptr;
                const unsigned long v = wcstoul(p, &e, 10);
                if (e == p) break;
                out[n++] = static_cast<uint32_t>(v);
                p = e;
            }
        };
        rdList(L"DenyComputeCaps", g_spvCfg.denyComputeCaps, g_spvCfg.nDenyComputeCaps);
        rdList(L"DenyAnyCaps", g_spvCfg.denyAnyCaps, g_spvCfg.nDenyAnyCaps);
        rdList(L"DenyComputeAllCaps", g_spvCfg.denyComputeAll, g_spvCfg.nDenyComputeAll);
        if (g_spvCfg.useReplacements) ScanReplacements(path);
        LogLine("SPIRV config (%ls): DenyFloat16=%d DenyInt16=%d DenyDotProduct=%d DenyRayTracing=%d DenyMesh=%d PatchVersion=%d LogAll=%d Serialize=%d Catch=%d DenyComputeAll=%d caps Brightness=%.2f",
            path, g_spvCfg.denyFloat16, g_spvCfg.denyInt16, g_spvCfg.denyDotProduct,
            g_spvCfg.denyRayTracing, g_spvCfg.denyMesh, g_spvCfg.patchVersion, g_spvCfg.logAll,
            g_spvCfg.serializeCompile, g_spvCfg.catchDriverCrash, g_spvCfg.nDenyComputeAll, g_spvCfg.brightness);
    }

    // Blocking, thread-safe: other threads must WAIT for the ini to be read, otherwise shaders created at the
    // same moment would be analysed with default values (this made the first few shaders ignore the ini).
    SRWLOCK g_spvCfgLock = SRWLOCK_INIT;
    void LoadSpvConfig() {
        if (g_spvCfgLoaded == 2) return;
        AcquireSRWLockExclusive(&g_spvCfgLock);
        if (g_spvCfgLoaded != 2) { LoadSpvConfigImpl(); g_spvCfgLoaded = 2; }
        ReleaseSRWLockExclusive(&g_spvCfgLock);
    }

    // ---- breadcrumb ring buffer, dumped by the crash handler -----------------
    constexpr int kCrumbCount = 64;
    constexpr int kCrumbLen = 240;
    char g_crumbs[kCrumbCount][kCrumbLen];
    volatile LONG g_crumbIdx = 0;

    void Crumb(const char* fmt, ...) noexcept {
        const LONG i = (InterlockedIncrement(&g_crumbIdx) - 1) & (kCrumbCount - 1);
        char tmp[kCrumbLen];
        va_list a; va_start(a, fmt);
        vsnprintf_s(tmp, sizeof(tmp), _TRUNCATE, fmt, a);
        va_end(a);
        _snprintf_s(g_crumbs[i], kCrumbLen, _TRUNCATE, "[t%lu %llu] %s", GetCurrentThreadId(),
            static_cast<unsigned long long>(GetTickCount64()), tmp);
    }

// ---------------------------------------------------------------------------
// SPIR-V rewriter: removes ray-query usage so a GPU/driver without ray tracing can still run the shader.
//   * OpTypeAccelerationStructureKHR / OpTypeRayQueryKHR and everything built from them are deleted
//   * ray-query instructions that return a value are replaced by "null" (no hit) constants
//   * the RayQueryKHR capability and SPV_KHR_ray_query extension are removed
// Pure C++17, no Windows dependencies (unit-tested offline with spirv-val).
// ---------------------------------------------------------------------------
namespace spvrw {
// generated from spirv.core.grammar.json
struct SpvOpDesc { uint16_t op; const char* kinds; };
static const SpvOpDesc kSpvOps[] = {
    {0,""},
    {1,"TR"},
    {2,"S"},
    {3,"LLIS"},
    {4,"S"},
    {5,"IS"},
    {6,"ILS"},
    {7,"RS"},
    {8,"ILL"},
    {10,"S"},
    {11,"RS"},
    {12,"TRIL(I)"},
    {14,"LL"},
    {15,"LIS(I)"},
    {16,"IL"},
    {17,"L"},
    {19,"R"},
    {20,"R"},
    {21,"RLL"},
    {22,"RLL"},
    {23,"RIL"},
    {24,"RIL"},
    {25,"RILLLLLLL"},
    {26,"R"},
    {27,"RI"},
    {28,"RII"},
    {29,"RI"},
    {30,"R(I)"},
    {31,"RS"},
    {32,"RLI"},
    {33,"RI(I)"},
    {34,"R"},
    {35,"R"},
    {36,"R"},
    {37,"R"},
    {38,"RL"},
    {39,"IL"},
    {41,"TR"},
    {42,"TR"},
    {43,"TRN"},
    {44,"TR(I)"},
    {45,"TRLLL"},
    {46,"TR"},
    {48,"TR"},
    {49,"TR"},
    {50,"TRN"},
    {51,"TR(I)"},
    {52,"TRL"},
    {54,"TRLI"},
    {55,"TR"},
    {56,""},
    {57,"TRI(I)"},
    {59,"TRLI"},
    {60,"TRIII"},
    {61,"TRIA"},
    {62,"IIA"},
    {63,"IIAA"},
    {64,"IIIAA"},
    {65,"TRI(I)"},
    {66,"TRI(I)"},
    {67,"TRII(I)"},
    {68,"TRIL"},
    {69,"TRI"},
    {70,"TRII(I)"},
    {71,"IL"},
    {72,"ILL"},
    {73,"R"},
    {74,"I(I)"},
    {75,"I(IL)"},
    {77,"TRII"},
    {78,"TRIII"},
    {79,"TRII(L)"},
    {80,"TR(I)"},
    {81,"TRI(L)"},
    {82,"TRII(L)"},
    {83,"TRI"},
    {84,"TRI"},
    {86,"TRII"},
    {87,"TRIIM"},
    {88,"TRIIM"},
    {89,"TRIIIM"},
    {90,"TRIIIM"},
    {91,"TRIIM"},
    {92,"TRIIM"},
    {93,"TRIIIM"},
    {94,"TRIIIM"},
    {95,"TRIIM"},
    {96,"TRIIIM"},
    {97,"TRIIIM"},
    {98,"TRIIM"},
    {99,"IIIM"},
    {100,"TRI"},
    {101,"TRI"},
    {102,"TRI"},
    {103,"TRII"},
    {104,"TRI"},
    {105,"TRII"},
    {106,"TRI"},
    {107,"TRI"},
    {109,"TRI"},
    {110,"TRI"},
    {111,"TRI"},
    {112,"TRI"},
    {113,"TRI"},
    {114,"TRI"},
    {115,"TRI"},
    {116,"TRI"},
    {117,"TRI"},
    {118,"TRI"},
    {119,"TRI"},
    {120,"TRI"},
    {121,"TRI"},
    {122,"TRI"},
    {123,"TRIL"},
    {124,"TRI"},
    {126,"TRI"},
    {127,"TRI"},
    {128,"TRII"},
    {129,"TRII"},
    {130,"TRII"},
    {131,"TRII"},
    {132,"TRII"},
    {133,"TRII"},
    {134,"TRII"},
    {135,"TRII"},
    {136,"TRII"},
    {137,"TRII"},
    {138,"TRII"},
    {139,"TRII"},
    {140,"TRII"},
    {141,"TRII"},
    {142,"TRII"},
    {143,"TRII"},
    {144,"TRII"},
    {145,"TRII"},
    {146,"TRII"},
    {147,"TRII"},
    {148,"TRII"},
    {149,"TRII"},
    {150,"TRII"},
    {151,"TRII"},
    {152,"TRII"},
    {154,"TRI"},
    {155,"TRI"},
    {156,"TRI"},
    {157,"TRI"},
    {158,"TRI"},
    {159,"TRI"},
    {160,"TRI"},
    {161,"TRII"},
    {162,"TRII"},
    {163,"TRII"},
    {164,"TRII"},
    {165,"TRII"},
    {166,"TRII"},
    {167,"TRII"},
    {168,"TRI"},
    {169,"TRIII"},
    {170,"TRII"},
    {171,"TRII"},
    {172,"TRII"},
    {173,"TRII"},
    {174,"TRII"},
    {175,"TRII"},
    {176,"TRII"},
    {177,"TRII"},
    {178,"TRII"},
    {179,"TRII"},
    {180,"TRII"},
    {181,"TRII"},
    {182,"TRII"},
    {183,"TRII"},
    {184,"TRII"},
    {185,"TRII"},
    {186,"TRII"},
    {187,"TRII"},
    {188,"TRII"},
    {189,"TRII"},
    {190,"TRII"},
    {191,"TRII"},
    {194,"TRII"},
    {195,"TRII"},
    {196,"TRII"},
    {197,"TRII"},
    {198,"TRII"},
    {199,"TRII"},
    {200,"TRI"},
    {201,"TRIIII"},
    {202,"TRIII"},
    {203,"TRIII"},
    {204,"TRI"},
    {205,"TRI"},
    {207,"TRI"},
    {208,"TRI"},
    {209,"TRI"},
    {210,"TRI"},
    {211,"TRI"},
    {212,"TRI"},
    {213,"TRI"},
    {214,"TRI"},
    {215,"TRI"},
    {218,""},
    {219,""},
    {220,"I"},
    {221,"I"},
    {224,"III"},
    {225,"II"},
    {227,"TRIII"},
    {228,"IIII"},
    {229,"TRIIII"},
    {230,"TRIIIIII"},
    {231,"TRIIIIII"},
    {232,"TRIII"},
    {233,"TRIII"},
    {234,"TRIIII"},
    {235,"TRIIII"},
    {236,"TRIIII"},
    {237,"TRIIII"},
    {238,"TRIIII"},
    {239,"TRIIII"},
    {240,"TRIIII"},
    {241,"TRIIII"},
    {242,"TRIIII"},
    {245,"TR(II)"},
    {246,"IIL"},
    {247,"IL"},
    {248,"R"},
    {249,"I"},
    {250,"III(L)"},
    {251,"II(LI)"},
    {252,""},
    {253,""},
    {254,"I"},
    {255,""},
    {256,"IL"},
    {257,"IL"},
    {259,"TRIIIIII"},
    {260,"III"},
    {261,"TRII"},
    {262,"TRII"},
    {263,"TRIII"},
    {264,"TRILI"},
    {265,"TRILI"},
    {266,"TRILI"},
    {267,"TRILI"},
    {268,"TRILI"},
    {269,"TRILI"},
    {270,"TRILI"},
    {271,"TRILI"},
    {274,"TRIIII"},
    {275,"TRIIII"},
    {276,"TRIIIIII"},
    {277,"TRIIIIII"},
    {278,"TRIIII"},
    {279,"TRIIII"},
    {280,"IIII"},
    {281,"IIII"},
    {282,"TRI"},
    {283,"TRIII"},
    {284,"TRIII"},
    {285,"TRIIIII"},
    {286,"TRIIIII"},
    {287,"IIIII"},
    {288,"IIIII"},
    {291,"TRIIII"},
    {292,"TRIIIIIIIIII(I)"},
    {293,"TRIIIII"},
    {294,"TRIIIII"},
    {295,"TRIIII"},
    {296,"TRIIII"},
    {297,"I"},
    {298,"I"},
    {299,"TR"},
    {300,"TRI"},
    {301,"II"},
    {302,"III"},
    {303,"TR"},
    {304,"TRIII"},
    {305,"TRIIM"},
    {306,"TRIIM"},
    {307,"TRIIIM"},
    {308,"TRIIIM"},
    {309,"TRIIM"},
    {310,"TRIIM"},
    {311,"TRIIIM"},
    {312,"TRIIIM"},
    {313,"TRIIM"},
    {314,"TRIIIM"},
    {315,"TRIIIM"},
    {316,"TRI"},
    {317,""},
    {318,"TRIII"},
    {319,"III"},
    {320,"TRIIM"},
    {321,"TRI"},
    {322,"R"},
    {323,"TRLLL"},
    {324,"TRI"},
    {325,"TRIIIII"},
    {326,"TRIIII"},
    {327,"R"},
    {328,"TRI"},
    {329,"III"},
    {330,"S"},
    {331,"IL"},
    {332,"IL"},
    {333,"TRI"},
    {334,"TRII"},
    {335,"TRII"},
    {336,"TRII"},
    {337,"TRIII"},
    {338,"TRII"},
    {339,"TRII"},
    {340,"TRII"},
    {341,"TRIII"},
    {342,"TRILI"},
    {343,"TRII"},
    {344,"TRII"},
    {345,"TRIII"},
    {346,"TRIII"},
    {347,"TRIII"},
    {348,"TRIII"},
    {349,"TRILII"},
    {350,"TRILII"},
    {351,"TRILII"},
    {352,"TRILII"},
    {353,"TRILII"},
    {354,"TRILII"},
    {355,"TRILII"},
    {356,"TRILII"},
    {357,"TRILII"},
    {358,"TRILII"},
    {359,"TRILII"},
    {360,"TRILII"},
    {361,"TRILII"},
    {362,"TRILII"},
    {363,"TRILII"},
    {364,"TRILII"},
    {365,"TRIII"},
    {366,"TRIII"},
    {400,"TRI"},
    {401,"TRII"},
    {402,"TRII"},
    {403,"TRII"},
    {4160,"TRII"},
    {4161,"TRI"},
    {4162,"TRI"},
    {4416,""},
    {4417,"RL"},
    {4418,"TRLII"},
    {4419,"TRII(I)"},
    {4420,"TRII(I)"},
    {4421,"TRI"},
    {4422,"TRI"},
    {4423,"TRIII(I)"},
    {4424,"TRIII(I)"},
    {4425,"TRIIL"},
    {4426,"IIIII"},
    {4428,"TRI"},
    {4429,"TRI"},
    {4430,"TRI"},
    {4431,"TRIIII"},
    {4432,"TRII"},
    {4433,"TRIL(I)"},
    {4445,"IIIIIIIIIII"},
    {4446,"II"},
    {4447,"TRI"},
    {4448,""},
    {4449,""},
    {4450,"TRIIL"},
    {4451,"TRIIL"},
    {4452,"TRIIL"},
    {4453,"TRIIIL"},
    {4454,"TRIIIL"},
    {4455,"TRIIIL"},
    {4456,"RIIIII"},
    {4457,"TRIIIA"},
    {4458,"IIIIA"},
    {4459,"TRIIIL"},
    {4460,"TRI"},
    {4461,"TRI"},
    {4462,"TRI"},
    {4463,"TRI"},
    {4472,"R"},
    {4473,"IIIIIIII"},
    {4474,"I"},
    {4475,"II"},
    {4476,"I"},
    {4477,"TRI"},
    {4479,"TRII"},
    {4480,"TRI"},      // [FIX] GetRayTMinKHR
    {4481,"TRI"},      // [FIX] GetRayFlagsKHR
    {4482,"TRII"},     // [FIX] GetIntersectionTKHR
    {4483,"TRII"},     // [FIX] GetIntersectionInstanceCustomIndexKHR
    {4484,"TRII"},     // [FIX] GetIntersectionInstanceIdKHR
    {4485,"TRII"},     // [FIX] GetIntersectionInstanceShaderBindingTableRecordOffsetKHR
    {4486,"TRII"},     // [FIX] GetIntersectionGeometryIndexKHR
    {4487,"TRII"},     // [FIX] GetIntersectionPrimitiveIndexKHR
    {4488,"TRII"},     // [FIX] GetIntersectionBarycentricsKHR
    {4489,"TRII"},     // [FIX] GetIntersectionFrontFaceKHR
    {4490,"TRI"},      // [FIX] GetIntersectionCandidateAABBOpaqueKHR
    {4491,"TRII"},     // [FIX] GetIntersectionObjectRayDirectionKHR
    {4492,"TRII"},     // [FIX] GetIntersectionObjectRayOriginKHR
    {4493,"TRI"},      // [FIX] GetWorldRayDirectionKHR
    {4494,"TRI"},      // [FIX] GetWorldRayOriginKHR
    {4495,"TRII"},     // [FIX] GetIntersectionObjectToWorldKHR
    {4496,"TRII"},     // [FIX] GetIntersectionWorldToObjectKHR
    {4500,"TRIIIII"},
    {4501,"TRIIIII"},
    {4502,"TRIIIII"},
    {4503,"TRIIIII"},
    {5000,"TRILI"},
    {5001,"TRILI"},
    {5002,"TRILI"},
    {5003,"TRILI"},
    {5004,"TRILI"},
    {5005,"TRILI"},
    {5006,"TRILI"},
    {5007,"TRILI"},
    {5011,"TRII"},
    {5012,"TRIII"},
    {5056,"TRI"},
    {5074,"TRIII"},
    {5075,"I"},
    {5076,"RI"},
    {5078,"TRI"},
    {5090,"TRI"},
    {5101,"TRII"},
    {5103,"RS"},
    {5104,"RS"},
    {5110,"TRI"},
    {5111,"TRI"},
    {5249,"IIIIIIIIIIIIII"},
    {5250,"IIIIIIIIIIIII"},
    {5251,"IIIIIII"},
    {5252,"TRI"},
    {5253,"TRI"},
    {5254,"TRI"},
    {5255,"TRI"},
    {5256,"IIIIIIIIIIIII"},
    {5257,"TRI"},
    {5258,"TRI"},
    {5259,"I"},
    {5260,"IIIIIIIIIIII"},
    {5261,"IIIIIIIIIIIII"},
    {5262,"IIIIIIIIIIII"},
    {5263,"IIIIII"},
    {5264,"II"},
    {5265,"TRI"},
    {5266,"II"},
    {5267,"TRI"},
    {5268,"TRI"},
    {5269,"TRI"},
    {5270,"TRI"},
    {5271,"TRI"},
    {5272,"TRI"},
    {5273,"TRI"},
    {5274,"TRI"},
    {5275,"TRI"},
    {5276,"TRI"},
    {5277,"TRI"},
    {5278,"TRI"},
    {5279,"III"},
    {5280,"II"},
    {5281,"R"},
    {5283,"TRIIIIM"},
    {5288,"RII"},
    {5289,"TRIIIIIIIIIIL"},
    {5290,"IIIIIII"},
    {5291,"III"},
    {5292,"TRIIIIIIIIIIIIIL"},
    {5293,"TRI"},
    {5294,"IIII"},
    {5295,"II"},
    {5296,"TRI"},
    {5299,"II"},
    {5300,"TRIIIII"},
    {5301,"TRIIIII"},
    {5302,"TRIIA"},
    {5303,"IIIA"},
    {5334,"TRII"},
    {5335,""},
    {5336,""},
    {5337,"IIIIIIIIIII"},
    {5338,"IIIIIIIIIIII"},
    {5339,"IIIIIIIIIIII"},
    {5340,"TRII"},
    {5341,"R"},
    {5344,"II"},
    {5345,"TRII"},
    {5346,"TRI"},
    {5358,"RIIII"},
    {5359,"TRIIIA"},
    {5360,"IIIIA"},
    {5361,"TRIII"},
    {5362,"TRI"},
    {5364,""},
    {5365,""},
    {5366,"TRILI"},
    {5367,"TRIIIAL"},
    {5368,"IIIAL"},
    {5369,"TRII(I)"},
    {5370,"RII"},
    {5371,"RII(I)"},
    {5372,"TR"},
    {5373,"TRI(I)"},
    {5374,"TRI(I)"},
    {5375,"TRI(I)"},
    {5376,"TRII"},
    {5377,"TR"},
    {5378,"TRI(I)"},
    {5379,"TRI(I)"},
    {5380,""},
    {5381,"TR"},
    {5382,"TRIIIII"},
    {5384,"TRI(I)"},
    {5390,"TRI"},
    {5391,"TRI"},
    {5392,"TRI"},
    {5393,"TRI"},
    {5394,"TRI"},
    {5395,"TRI"},
    {5396,"TRI"},
    {5397,"L"},
    {5398,"TRIIIIL"},
    {5427,"TRII"},
    {5428,"TRII"},
    {5429,"TRII"},
    {5430,"TRII"},
    {5431,"TRII"},
    {5432,"TRI"},
    {5433,"TRI"},
    {5434,"TRI"},
    {5435,"TRI"},
    {5436,"TRI"},
    {5437,"TRI"},
    {5438,"TRII"},
    {5439,"TRII"},
    {5571,"TRII"},
    {5572,"TRIII"},
    {5573,"TRIII"},
    {5574,"TRII"},
    {5575,"TRI"},
    {5576,"II"},
    {5577,"TRII"},
    {5578,"III"},
    {5580,"TRIIII"},
    {5581,"IIIII"},
    {5585,"TRI"},
    {5586,"TRI"},
    {5587,"TRII"},
    {5588,"TRII"},
    {5589,"TRII"},
    {5590,"TRII"},
    {5591,"TRII"},
    {5592,"TRII"},
    {5593,"TRII"},
    {5594,"TRII"},
    {5595,"TRII"},
    {5596,"TRII"},
    {5597,"TRII"},
    {5598,"TRII"},
    {5600,"TRI"},
    {5601,"TR(I)"},
    {5609,"RS"},
    {5610,"TRIISS"},
    {5611,"TRI(I)"},
    {5614,"TRIIII"},
    {5615,"TRIIII"},
    {5630,"I"},
    {5631,"TRII"},
    {5632,"IL"},
    {5633,"ILL"},
    {5699,"TRII"},
    {5700,"RI"},
    {5701,"R"},
    {5702,"R"},
    {5703,"R"},
    {5704,"R"},
    {5705,"R"},
    {5706,"R"},
    {5707,"R"},
    {5708,"R"},
    {5709,"R"},
    {5710,"R"},
    {5711,"R"},
    {5712,"R"},
    {5713,"TRII"},
    {5714,"TRII"},
    {5715,"TRII"},
    {5716,"TRII"},
    {5717,"TRII"},
    {5718,"TRII"},
    {5719,"TRII"},
    {5720,"TRII"},
    {5721,"TR"},
    {5722,"TR"},
    {5723,"TR"},
    {5724,"TRIIII"},
    {5725,"TRII"},
    {5726,"TR"},
    {5727,"TR"},
    {5728,"TRI"},
    {5729,"TRII"},
    {5730,"TRII"},
    {5731,"TRIII"},
    {5732,"TRI"},
    {5733,"TRI"},
    {5734,"TRI"},
    {5735,"TRI"},
    {5736,"TRI"},
    {5737,"TRI"},
    {5738,"TRI"},
    {5739,"TRI"},
    {5740,"TRI"},
    {5741,"TRI"},
    {5742,"TRI"},
    {5743,"TRI"},
    {5744,"TRI"},
    {5745,"TRI"},
    {5746,"TRIII"},
    {5747,"TRIII"},
    {5748,"TRIII"},
    {5749,"TRIIII"},
    {5750,"TRII"},
    {5751,"TRIIII"},
    {5752,"TRI"},
    {5753,"TRII"},
    {5754,"TRI"},
    {5755,"TRII"},
    {5756,"TRII"},
    {5757,"TRIII"},
    {5758,"TRIIII"},
    {5759,"TRIIII"},
    {5760,"TRIIIII"},
    {5761,"TRIII"},
    {5762,"TRIIII"},
    {5763,"TRIIII"},
    {5764,"TRIIIII"},
    {5765,"TRI"},
    {5766,"TRI"},
    {5767,"TRI"},
    {5768,"TRI"},
    {5769,"TRI"},
    {5770,"TRII"},
    {5771,"TRII"},
    {5772,"TRII"},
    {5773,"TRIII"},
    {5774,"TRIII"},
    {5775,"TRIII"},
    {5776,"TRII"},
    {5777,"TRI"},
    {5778,"TRI"},
    {5779,"TRI"},
    {5780,"TRI"},
    {5781,"TRIIIIIII"},
    {5782,"TRIIIIIIII"},
    {5783,"TRI"},
    {5784,"TRI"},
    {5785,"TRI"},
    {5786,"TRIII"},
    {5787,"TRIIII"},
    {5788,"TRIII"},
    {5789,"TRIIII"},
    {5790,"TRI"},
    {5791,"TRI"},
    {5792,"TRIIIIII"},
    {5793,"TRIIIIIIII"},
    {5794,"TRIIIIIIIIIII"},
    {5795,"TRII"},
    {5796,"TRI"},
    {5797,"TRII"},
    {5798,"TRIIII"},
    {5799,"TRII"},
    {5800,"TRI"},
    {5801,"TRII"},
    {5802,"TRII"},
    {5803,"TRII"},
    {5804,"TRIII"},
    {5805,"TRIIII"},
    {5806,"TRIII"},
    {5807,"TRIIII"},
    {5808,"TRI"},
    {5809,"TRI"},
    {5810,"TRI"},
    {5811,"TRI"},
    {5812,"TRI"},
    {5813,"TRI"},
    {5814,"TRI"},
    {5815,"TRI"},
    {5816,"TRI"},
    {5818,"TRI"},
    {5819,"TR"},
    {5820,"I"},
    {5840,"TRILLLLLL"},
    {5841,"TRILLLLL"},
    {5842,"TRILLLLL"},
    {5843,"TRILLLL"},
    {5846,"TRILILLLLL"},
    {5847,"TRILILLLLL"},
    {5848,"TRILILLLLL"},
    {5849,"TRILILLLLL"},
    {5850,"TRILIL"},
    {5851,"TRILIL"},
    {5852,"TRILIL"},
    {5853,"TRILIL"},
    {5854,"TRILIL"},
    {5855,"TRILLLLL"},
    {5856,"TRILLLLL"},
    {5857,"TRILLLLL"},
    {5858,"TRILILLLLL"},
    {5859,"TRILLLLL"},
    {5860,"TRILLLLL"},
    {5861,"TRILLLLL"},
    {5862,"TRILLLLL"},
    {5863,"TRILLLLL"},
    {5864,"TRILLLLL"},
    {5865,"TRILLLLL"},
    {5866,"TRILLLLL"},
    {5867,"TRILLLLL"},
    {5868,"TRILLLLL"},
    {5869,"TRILLLLL"},
    {5870,"TRILLLLL"},
    {5871,"TRILLLLL"},
    {5872,"TRILLLLL"},
    {5873,"TRILLLLL"},
    {5874,"TRILLLLL"},
    {5875,"TRILLLLL"},
    {5876,"TRILLLLL"},
    {5877,"TRILLLLL"},
    {5878,"TRILLLLL"},
    {5879,"TRILILLLLL"},
    {5880,"TRILILLLLL"},
    {5881,"TRILILLLLL"},
    {5882,"TRILILLLL"},
    {5887,"(L)"},
    {5911,"RI"},
    {5912,"RII"},
    {5913,"R(I)"},
    {5923,"TRIILLLLL"},
    {5924,"TRIILLLLL"},
    {5925,"TRIILLLLL"},
    {5926,"TRIILLLLL"},
    {5927,"TRIILLLLL"},
    {5928,"TRIILLLLL"},
    {5929,"TRIILLLLL"},
    {5930,"TRIILLLLL"},
    {5931,"TRIILLLLL"},
    {5932,"TRIILLLLL"},
    {5933,"TRIILLLLL"},
    {5934,"TRI"},
    {5938,"TRI"},
    {5946,"TRII"},
    {5947,"TRII"},
    {5949,"TRII"},
    {6016,"TRI"},
    {6017,"TRI"},
    {6018,"TRII"},
    {6019,"TRII"},
    {6020,"TRII"},
    {6021,"TRII"},
    {6022,"TRII"},
    {6023,"TRII"},
    {6024,"TRII"},
    {6025,"TRII"},
    {6026,"TRI"},
    {6027,"TRII"},
    {6028,"TRII"},
    {6029,"TRI"},
    {6030,"TRI"},
    {6031,"TRII"},
    {6032,"TRII"},
    {6035,"TRIIII"},
    {6086,"RL"},
    {6090,"(I)"},
    {6091,"(I)"},
    {6092,"(I)"},
    {6096,"TR(I)"},
    {6116,"TRI"},
    {6117,"TRI"},
    {6142,"III"},
    {6143,"III"},
    {6145,"TRI"},
    {6221,"IIA"},
    {6231,"IIIIIIIIII"},
    {6232,"IIIIIIIIII"},
    {6233,"IIIIIIIIII"},
    {6234,"IIIIIIIII"},
    {6235,"IIIIIIIIII"},
    {6237,"TRIIIIL"},
    {6401,"TRILI"},
    {6402,"TRILI"},
    {6403,"TRILI"},
    {6404,"TRILI"},
    {6405,"TRILI"},
    {6406,"TRILI"},
    {6407,"TRILI"},
    {6408,"TRILI"},
    {6428,"TRILII"},
    {6429,"IILI"},
};
    inline const char* KindsFor(uint32_t op) {
        size_t lo = 0, hi = sizeof(kSpvOps) / sizeof(kSpvOps[0]);
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (kSpvOps[mid].op < op) lo = mid + 1; else hi = mid;
        }
        if (lo < sizeof(kSpvOps) / sizeof(kSpvOps[0]) && kSpvOps[lo].op == op) return kSpvOps[lo].kinds;
        return nullptr;
    }

    // Calls f(wordIndex, role) for every id operand of an instruction. role: 'T' result type, 'R' result id, 'I' id use.
    template<class F> void ForEachId(const uint32_t* w, uint32_t op, uint32_t wc, F&& f) {
        const char* k = KindsFor(op);
        if (!k) return;
        size_t pos = 1;
        auto token = [&](char t) {
            if (pos >= wc) return;
            switch (t) {
            case 'T': f(pos, 'T'); ++pos; break;
            case 'R': f(pos, 'R'); ++pos; break;
            case 'I': f(pos, 'I'); ++pos; break;
            case 'L': ++pos; break;
            case 'N': pos = wc; break;
            case 'S': {
                while (pos < wc) {
                    const uint32_t x = w[pos++];
                    if ((x & 0xFF) == 0 || (x & 0xFF00) == 0 || (x & 0xFF0000) == 0 || (x & 0xFF000000) == 0) break;
                }
                break;
            }
            case 'M': {
                ++pos;                                 // image operand mask
                while (pos < wc) { f(pos, 'I'); ++pos; }
                break;
            }
            case 'A': {
                const uint32_t mask = w[pos++];
                if (mask & 0x2) ++pos;                 // Aligned literal
                if ((mask & 0x8) && pos < wc) { f(pos, 'I'); ++pos; }
                if ((mask & 0x10) && pos < wc) { f(pos, 'I'); ++pos; }
                break;
            }
            default: break;
            }
        };
        for (size_t i = 0; k[i];) {
            if (k[i] == '(') {
                size_t j = i + 1;
                while (k[j] && k[j] != ')') ++j;
                while (pos < wc) {
                    const size_t before = pos;
                    for (size_t c = i + 1; c < j; ++c) token(k[c]);
                    if (pos == before) break;
                }
                i = j + 1;
            } else { token(k[i]); ++i; }
        }
    }

#include "swrt_rq.h"   // [SWRT] EmulateRayQuery: OpRayQuery* -> software BVH traversal

    struct Inst { uint32_t op, wc; size_t off; };

    inline bool IsRqValueOp(uint32_t op) {
        // [FIX] Покрываем все геттеры ray query 4480..4496 (иначе StripRayQuery падает
        // с dangling reference -> шейдеры освещения DENIED -> квадраты в мире).
        return op == 4477 || op == 4479 || op == 5340 ||
               (op >= 4480 && op <= 4496) ||
               (op >= 6016 && op <= 6032);
    }
    inline bool IsRqVoidOp(uint32_t op) { return op >= 4473 && op <= 4476; }

    // Experimental variants for modules that crash the old driver's SPIR-V front end.
    //   1: OpConvertUToPtr -> OpBitcast (equivalent for PhysicalStorageBuffer pointers)
    //   2: variant 1 + drop AliasedPointer / RestrictPointer decorations
    inline bool ApplyVariant(int variant, const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
        if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
        out.clear();
        out.reserve(n);
        out.insert(out.end(), in, in + 5);
        bool changed = false;
        for (size_t i = 5; i < n;) {
            const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
            if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
            if (variant >= 1 && op == 120 && wc == 4) {                      // OpConvertUToPtr
                out.push_back((4u << 16) | 124u);                            // OpBitcast
                out.insert(out.end(), in + i + 1, in + i + wc);
                changed = true;
            } else if (variant >= 2 && op == 71 && wc >= 3 && (in[i + 2] == 5355 || in[i + 2] == 5356)) {
                changed = true;                                              // drop decoration
            } else {
                out.insert(out.end(), in + i, in + i + wc);
            }
            i += wc;
        }
        if (!changed) { err = "variant not applicable"; return false; }
        return true;
    }
	
	
	
	
	// v8: убираем инструкции, которых старый драйвер не знает.
//  * OpReadClockKHR (5056, cap ShaderClockKHR=5055) -> нулевая константа
//  * старые subgroup-опы SPV_KHR_shader_ballot -> core GroupNonUniform
inline bool LowerUnsupportedOps(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
    if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
    std::vector<Inst> insts;
    for (size_t i = 5; i < n;) {
        const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
        if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
        insts.push_back({ op, wc, i });
        i += wc;
    }
    auto oldBallot = [](uint32_t op) {
        return op == 4421 || op == 4422 || op == 4428 || op == 4429 || op == 4430 || op == 4432;
    };
    bool any = false, needScope = false, needShuffle = false, hasShuffle = false;
    uint32_t uintTy = 0; bool uintTyExists = false;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 21 && I.wc >= 4 && w[2] == 32 && w[3] == 0) { uintTy = w[1]; uintTyExists = true; }
        if (I.op == 17 && I.wc >= 2 && w[1] == 65) hasShuffle = true;
        if (I.op == 5056) any = true;
        if (oldBallot(I.op)) { any = true; needScope = true; }
        if (I.op == 4432) needShuffle = true;
    }
    if (!any) { err = "nothing to lower"; return false; }

    uint32_t bound = in[3];
    std::map<uint32_t, uint32_t> nullFor;
    std::vector<uint32_t> nullOrder;
    for (const Inst& I : insts) {
        if (I.op != 5056) continue;
        if (I.wc < 4) { err = "short OpReadClockKHR"; return false; }
        const uint32_t t = in[I.off + 1];
        if (!nullFor.count(t)) { nullFor[t] = bound++; nullOrder.push_back(t); }
    }
    uint32_t scopeId = 0;
    if (needScope) { if (!uintTyExists) uintTy = bound++; scopeId = bound++; }

    out.clear();
    out.reserve(n + 32);
    out.insert(out.end(), in, in + 5);
    bool capDone = false, extras = false;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        const uint32_t op = I.op;
        if (op == 17 && I.wc >= 2) {
            const uint32_t c = w[1];
            if (c == 5055 || c == 4423) continue;
            out.insert(out.end(), w, w + I.wc);
            if (!capDone) {
                capDone = true;
                if (needShuffle && !hasShuffle) { out.push_back((2u << 16) | 17u); out.push_back(65); }
            }
            continue;
        }
        if (op == 10 && I.wc >= 2) {
            const char* s = reinterpret_cast<const char*>(w + 1);
            const size_t lim = (I.wc - 1) * 4;
            if (strncmp(s, "SPV_KHR_shader_clock", lim) == 0 || strncmp(s, "SPV_KHR_shader_ballot", lim) == 0) continue;
        }
        if (op == 54 && !extras) {
            extras = true;
            if (needScope && !uintTyExists) {
                out.push_back((4u << 16) | 21u); out.push_back(uintTy); out.push_back(32); out.push_back(0);
            }
            if (needScope) { out.push_back((4u << 16) | 43u); out.push_back(uintTy); out.push_back(scopeId); out.push_back(3); }
            for (uint32_t t : nullOrder) { out.push_back((3u << 16) | 46u); out.push_back(t); out.push_back(nullFor[t]); }
        }
        if (op == 5056) {
            out.push_back((4u << 16) | 83u);
            out.push_back(w[1]); out.push_back(w[2]); out.push_back(nullFor[w[1]]);
            continue;
        }
        if (oldBallot(op)) {
            uint32_t nop;
            switch (op) {
            case 4421: nop = 339; break;   // Ballot
            case 4422: nop = 338; break;   // BroadcastFirst
            case 4428: nop = 334; break;   // All
            case 4429: nop = 335; break;   // Any
            case 4430: nop = 336; break;   // AllEqual
            default:   nop = 345; break;   // Shuffle
            }
            out.push_back(((I.wc + 1) << 16) | nop);
            out.push_back(w[1]); out.push_back(w[2]); out.push_back(scopeId);
            for (uint32_t k = 3; k < I.wc; ++k) out.push_back(w[k]);
            continue;
        }
        out.insert(out.end(), w, w + I.wc);
    }
    out[3] = bound;
    return true;
}
	
	

    // Replace every OpImageFetch (texel fetch) by a null constant of the same result type. Used as a last resort
    // for pipelines the old driver refuses; geometry still draws, only the fetched data (e.g. vertex offsets) is lost.
    inline bool StripImageFetch(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
        if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
        std::vector<Inst> insts;
        for (size_t i = 5; i < n;) {
            const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
            if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
            insts.push_back({ op, wc, i });
            i += wc;
        }
        uint32_t bound = in[3];
        std::unordered_map<uint32_t, uint32_t> nullFor;
        std::vector<uint32_t> order;
        for (const Inst& I : insts)
            if (I.op == 95 && I.wc >= 5) {                      // OpImageFetch
                const uint32_t t = in[I.off + 1];
                if (!nullFor.count(t)) { nullFor[t] = bound++; order.push_back(t); }
            }
        if (order.empty()) { err = "no OpImageFetch"; return false; }
        out.clear();
        out.reserve(n + 16);
        out.insert(out.end(), in, in + 5);
        bool consts = false;
        for (const Inst& I : insts) {
            const uint32_t* w = in + I.off;
            if (I.op == 54 && !consts) {
                consts = true;
                for (uint32_t t : order) { out.push_back((3u << 16) | 46u); out.push_back(t); out.push_back(nullFor[t]); }
            }
            if (I.op == 95 && I.wc >= 5) {
                out.push_back((4u << 16) | 83u);                // OpCopyObject %type %result %null
                out.push_back(w[1]); out.push_back(w[2]); out.push_back(nullFor[w[1]]);
            } else out.insert(out.end(), w, w + I.wc);
        }
        out[3] = bound;
        return true;
    }

    // Returns true on success (out filled). On failure, err explains and the caller keeps the original module.
    
    // Convert IEEE-754 binary16 bits to binary32 bits.
    inline uint32_t Float16BitsToFloat32Bits(uint16_t h) {
        const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        const uint32_t exp  = (h >> 10) & 0x1Fu;
        const uint32_t mant = h & 0x3FFu;
        if (exp == 0) {
            if (mant == 0) return sign; // +/- 0
            // subnormal -> normalize
            uint32_t m = mant;
            uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            return sign | (e << 23) | (m << 13);
        }
        if (exp == 31) {
            // Inf / NaN
            return sign | 0x7F800000u | (mant << 13);
        }
        return sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    
	
	
	// Структурная самопроверка. Никогда не отдаём драйверу модуль, который мы сами сломали.
// Проверяем: (1) каждое используемое id объявлено раньше, (2) инструкции с результатом
// стоят внутри базового блока. При провале вызывающий оставляет ОРИГИНАЛ.
inline bool CheckStructure(const uint32_t* w, size_t n, std::string& err) {
    if (!w || n < 5 || w[0] != 0x07230203u) { err = "not spir-v"; return false; }
    std::unordered_set<uint32_t> defined;
    bool inBlock = false;
    size_t i = 5;
    while (i < n) {
        const uint32_t first = w[i];
        const uint32_t wc = first >> 16, op = first & 0xFFFFu;
        if (wc == 0 || i + wc > n) { err = "bad word count"; return false; }
        const char* k = KindsFor(op);
        const bool hasRes = (k && k[0] == 'T' && k[1] == 'R' && wc >= 3);
        if (op == 54) inBlock = false;
        else if (op == 248) inBlock = true;
        else if (op == 56) inBlock = false;
        // Ссылки вперёд, разрешённые спекой SPIR-V:
        //  15 EntryPoint / 16 ExecutionMode / 331 ExecutionModeId -> id функции (объявлена ниже)
        //  5/6 имена, 71..75/332/5632/5633 декорации, 245 Phi, 246/247 merge,
        //  249..251 переходы, 39 TypeForwardPointer
        const bool fwdOk = op == 15 || op == 16 || op == 331 || op == 5 || op == 6 ||
                           op == 71 || op == 72 || op == 73 || op == 74 || op == 75 ||
                           op == 332 || op == 5632 || op == 5633 || op == 245 ||
                           op == 246 || op == 247 || op == 249 || op == 250 ||
                           op == 251 || op == 39;
        if (k && !fwdOk) {
            bool bad = false; uint32_t badId = 0;
            // OpFunctionCall: слово 3 - id вызываемой функции, она может быть объявлена ниже.
            // Аргументы (слова 4+) проверяем как обычно.
            ForEachId(w + i, op, wc, [&](size_t idx, char role) {
                if (role == 'R') return;
                if (op == 57 && idx == 3) return;
                if (!defined.count(w[i + idx])) { bad = true; badId = w[i + idx]; }
            });
            if (bad) { err = "op " + std::to_string(op) + " uses undefined id " + std::to_string(badId); return false; }
        }
        if (hasRes) {
            const bool globalOk = op == 1 || op == 54 || op == 55 || op == 59 ||
                                  (op >= 41 && op <= 46) || (op >= 48 && op <= 52);
            if (!globalOk && !inBlock) { err = "op " + std::to_string(op) + " result outside a block"; return false; }
        }
        if (k && k[0] == 'R' && wc >= 2) defined.insert(w[i + 1]);
        else if (hasRes) defined.insert(w[i + 2]);
        if (op == 39 && wc >= 2) defined.insert(w[i + 1]);
        if (op == 249 || op == 250 || op == 251 || op == 252 || op == 253 ||
            op == 254 || op == 255 || op == 4416) inBlock = false;
        i += wc;
    }
    return true;
}
	
	
// v10: fp16 + int16 ALU -> fp32/int32, память остаётся 16-битной.
// Драйвер: storageBuffer16BitAccess=1, bufferDeviceAddress=1, shaderFloat16=0, shaderInt16=0.
// v10b: fp16 + int16 ALU -> fp32/int32, память остаётся 16-битной.
// Исправлено: (1) конверсии только внутри блоков, (2) типы создаются до использования,
// (3) поддержка Int16 арифметики.
// [FIX v11 для GCN2 / R7 260X] Полное понижение 16-bit ALU + Bitcast:
// 1) Поддерживает 16-битные OpBitcast (v2<f16><->v2<i16><->i32, f16<->i16, v4<f16><->v2<i32>)
//    через GLSL.std.450 PackHalf2x16 / UnpackHalf2x16 — это снимает 152 падения (-13)
//    графических пайплайнов (силуэт Слеера в меню и чёрное оружие/геометрия в мире).
// 2) Не трогает OpFunction (54) и не флашит конверсии вне базовых блоков (снимает все
//    15 ошибок "BAD MODULE: op 115 result outside a block").
// 3) Отслеживает ВСЕ 16-битные int-типы (i16 и u16 вместе), OpPhi-петли, сравнения (bool)
//    и OpCompositeExtract из структур.
inline bool LowerFp16AndInt16Alu(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err, bool dropCaps) {
    if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
    std::vector<Inst> insts;
    for (size_t i = 5; i < n;) {
        const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
        if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
        insts.push_back({ op, wc, i });
        i += wc;
    }
    auto constLike = [](uint32_t op) {
        return op == 1 || op == 41 || op == 42 || op == 43 || op == 44 || op == 45 ||
               op == 46 || op == 48 || op == 49 || op == 50 || op == 51;
    };
    auto keepNarrow = [](uint32_t op) {
        return op == 61 || op == 55 || op == 57 || op == 59 || op == 60 ||
               (op >= 113 && op <= 115) || (op >= 86 && op <= 98) || (op >= 100 && op <= 107);
    };
    auto usesNarrowOperand = [](uint32_t op) {
        return op == 62 || op == 254 || op == 57 || op == 113 || op == 114 || op == 115 || op == 116;
    };

    std::unordered_map<uint32_t, uint32_t> tKind, tWidth, tSign, vComp, vLen, mCol, mCnt;
    std::vector<uint32_t> narrowScalars;
    uint32_t floatTy = 0, u32Ty = 0, i32Ty = 0, v2f32Ty = 0, v2u32Ty = 0, glslId = 0;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 11 && I.wc >= 3) {
            const char* s = reinterpret_cast<const char*>(w + 2);
            if (strncmp(s, "GLSL.std.450", (I.wc - 2) * 4) == 0) glslId = w[1];
        }
        if (I.op == 22 && I.wc >= 3) {
            tKind[w[1]] = 22; tWidth[w[1]] = w[2];
            if (w[2] == 16) narrowScalars.push_back(w[1]);
            if (w[2] == 32 && !floatTy) floatTy = w[1];
        }
        if (I.op == 21 && I.wc >= 4) {
            tKind[w[1]] = 21; tWidth[w[1]] = w[2]; tSign[w[1]] = w[3];
            if (w[2] == 16) narrowScalars.push_back(w[1]);
            if (w[2] == 32 && w[3] == 0 && !u32Ty) u32Ty = w[1];
            if (w[2] == 32 && w[3] == 1 && !i32Ty) i32Ty = w[1];
        }
        if (I.op == 23 && I.wc >= 4) {
            vComp[w[1]] = w[2]; vLen[w[1]] = w[3];
            if (w[2] == floatTy && w[3] == 2 && !v2f32Ty) v2f32Ty = w[1];
            if (w[2] == u32Ty && w[3] == 2 && !v2u32Ty) v2u32Ty = w[1];
        }
        if (I.op == 24 && I.wc >= 4) { mCol[w[1]] = w[2]; mCnt[w[1]] = w[3]; }
        if (I.op == 54) break;
    }
    if (narrowScalars.empty()) { err = "no 16-bit scalar type"; return false; }

    auto scalarOf = [&](uint32_t t) { if (mCol.count(t)) t = mCol[t]; if (vComp.count(t)) t = vComp[t]; return t; };
    std::unordered_set<uint32_t> narrow(narrowScalars.begin(), narrowScalars.end());
    for (const auto& kv : vComp) if (narrow.count(kv.second)) narrow.insert(kv.first);
    for (const auto& kv : mCol) if (narrow.count(kv.second)) narrow.insert(kv.first);

    std::unordered_map<uint32_t, uint32_t> vTy;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 54) continue;
        const char* k = KindsFor(I.op);
        if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3) vTy[w[2]] = w[1];
        if (I.op == 59 && I.wc >= 4) vTy[w[2]] = w[1];
    }

    bool hasBitcast16 = false;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 52 && I.wc >= 3 && narrow.count(w[1])) { err = "16-bit spec-const-op"; return false; }
        if (I.op == 124 && I.wc >= 4) {
            const uint32_t dt = w[1];
            auto it = vTy.find(w[3]);
            const uint32_t st = (it != vTy.end()) ? it->second : 0;
            if (narrow.count(dt) || narrow.count(st)) {
                auto elemBits = [&](uint32_t t) -> uint32_t { uint32_t sc = scalarOf(t); return tWidth.count(sc) ? tWidth[sc] : 0; };
                auto lanes = [&](uint32_t t) -> uint32_t { return vLen.count(t) ? vLen[t] : 1; };
                const uint32_t db = elemBits(dt) * lanes(dt), sb = elemBits(st) * lanes(st);
                if (!db || db != sb || mCnt.count(dt) || mCnt.count(st) || (db != 16 && db != 32 && db != 64)) {
                    err = "16-bit bitcast";
                    return false;
                }
                hasBitcast16 = true;
            }
        }
    }

    uint32_t bound = in[3];
    std::vector<uint32_t> extras, preMemExtras;
    std::unordered_map<uint32_t, uint32_t> wide;
    auto existing32 = [&](uint32_t t16) -> uint32_t {
        for (const auto& kv : tKind) {
            const uint32_t o = kv.first;
            if (kv.second == tKind[t16] && tWidth[o] == 32 && tSign[o] == tSign[t16]) return o;
        }
        return 0;
    };
    auto ensureF32 = [&]() {
        if (!floatTy) { floatTy = bound++; extras.push_back((3u<<16)|22u); extras.push_back(floatTy); extras.push_back(32); tKind[floatTy]=22; tWidth[floatTy]=32; }
        return floatTy;
    };
    auto ensureU32 = [&]() {
        if (!u32Ty) { u32Ty = bound++; extras.push_back((4u<<16)|21u); extras.push_back(u32Ty); extras.push_back(32); extras.push_back(0); tKind[u32Ty]=21; tWidth[u32Ty]=32; tSign[u32Ty]=0; }
        return u32Ty;
    };
    auto ensureV2F32 = [&]() {
        ensureF32();
        for (const auto& kv : vComp) if (kv.second == floatTy && vLen[kv.first] == 2) { v2f32Ty = kv.first; break; }
        if (!v2f32Ty) { v2f32Ty = bound++; extras.push_back((4u<<16)|23u); extras.push_back(v2f32Ty); extras.push_back(floatTy); extras.push_back(2); vComp[v2f32Ty]=floatTy; vLen[v2f32Ty]=2; }
        return v2f32Ty;
    };
    auto ensureV2U32 = [&]() {
        ensureU32();
        for (const auto& kv : vComp) if (kv.second == u32Ty && vLen[kv.first] == 2) { v2u32Ty = kv.first; break; }
        if (!v2u32Ty) { v2u32Ty = bound++; extras.push_back((4u<<16)|23u); extras.push_back(v2u32Ty); extras.push_back(u32Ty); extras.push_back(2); vComp[v2u32Ty]=u32Ty; vLen[v2u32Ty]=2; }
        return v2u32Ty;
    };

    for (uint32_t s16 : narrowScalars) {
        if (tKind[s16] == 22) { wide[s16] = ensureF32(); }
        else {
            uint32_t w32 = existing32(s16);
            if (!w32) {
                w32 = bound++;
                extras.push_back((4u<<16)|21u); extras.push_back(w32); extras.push_back(32); extras.push_back(tSign[s16]);
                tKind[w32]=21; tWidth[w32]=32; tSign[w32]=tSign[s16];
                if (tSign[s16] == 0 && !u32Ty) u32Ty = w32;
                if (tSign[s16] == 1 && !i32Ty) i32Ty = w32;
            }
            wide[s16] = w32;
        }
    }

    uint32_t c0_f32 = 0, c16_u32 = 0, cFFFF_u32 = 0;
    if (hasBitcast16) {
        if (!glslId) {
            glslId = bound++;
            preMemExtras.push_back((6u<<16)|11u); preMemExtras.push_back(glslId);
            preMemExtras.push_back(0x4C534C47u); preMemExtras.push_back(0x6474732Eu);
            preMemExtras.push_back(0x3035342Eu); preMemExtras.push_back(0x00000000u);
        }
        ensureF32(); ensureU32(); ensureV2F32(); ensureV2U32();
        c0_f32 = bound++; extras.push_back((4u<<16)|43u); extras.push_back(floatTy); extras.push_back(c0_f32); extras.push_back(0u);
        c16_u32 = bound++; extras.push_back((4u<<16)|43u); extras.push_back(u32Ty); extras.push_back(c16_u32); extras.push_back(16u);
        cFFFF_u32 = bound++; extras.push_back((4u<<16)|43u); extras.push_back(u32Ty); extras.push_back(cFFFF_u32); extras.push_back(0xFFFFu);
    }

    for (uint32_t t : narrow) {
        if (!vLen.count(t)) continue;
        const uint32_t len = vLen[t], base = wide[vComp[t]];
        uint32_t f = 0;
        for (const auto& kv : vComp) if (kv.second == base && vLen[kv.first] == len) { f = kv.first; break; }
        if (!f) { f = bound++; extras.push_back((4u<<16)|23u); extras.push_back(f); extras.push_back(base); extras.push_back(len); vComp[f]=base; vLen[f]=len; }
        wide[t] = f;
    }
    for (uint32_t t : narrow) {
        if (!mCnt.count(t)) continue;
        const uint32_t cnt = mCnt[t], col = wide[mCol[t]];
        uint32_t f = 0;
        for (const auto& kv : mCol) if (kv.second == col && mCnt[kv.first] == cnt) { f = kv.first; break; }
        if (!f) { f = bound++; extras.push_back((4u<<16)|24u); extras.push_back(f); extras.push_back(col); extras.push_back(cnt); mCol[f]=col; mCnt[f]=cnt; }
        wide[t] = f;
    }
    auto convOp = [&](uint32_t ty) -> uint32_t {
        const uint32_t s = scalarOf(ty);
        if (tKind[s] == 22) return 115;
        return tSign[s] ? 114u : 113u;
    };
    std::unordered_map<uint32_t, uint32_t> wId, nId;
    auto W = [&](uint32_t id){ auto it=wId.find(id); return it==wId.end()?id:it->second; };
    auto N = [&](uint32_t id){ auto it=nId.find(id); return it==nId.end()?id:it->second; };
    for (const Inst& I : insts) {
        if (!constLike(I.op) || I.wc < 3) continue;
        const uint32_t* w = in + I.off;
        const uint32_t t = w[1], id = w[2];
        if (!narrow.count(t)) continue;
        const uint32_t nid = bound++;
        if (I.op == 43 || I.op == 50) {
            if (I.wc < 4) { err = "short 16-bit constant"; return false; }
            const uint16_t bits = (uint16_t)(w[3] & 0xFFFFu);
            uint32_t val;
            if (tKind[scalarOf(t)] == 22) val = Float16BitsToFloat32Bits(bits);
            else if (tSign[scalarOf(t)]) val = (uint32_t)(int32_t)(int16_t)bits;
            else val = bits;
            extras.push_back((4u<<16)|I.op); extras.push_back(wide[t]); extras.push_back(nid); extras.push_back(val);
        } else if (I.op == 44 || I.op == 51) {
            extras.push_back((I.wc<<16)|I.op); extras.push_back(wide[t]); extras.push_back(nid);
            for (uint32_t k = 3; k < I.wc; ++k) extras.push_back(W(w[k]));
        } else {
            extras.push_back((3u<<16)|I.op); extras.push_back(wide[t]); extras.push_back(nid);
        }
        wId[id] = nid; nId[id] = id;
    }

    auto computeResWide = [&](const Inst& I, bool& hasResOut, bool& resWideOut) {
        const uint32_t* w = in + I.off;
        const uint32_t op = I.op;
        if (op == 54) { hasResOut = false; resWideOut = false; return; }
        const char* k = KindsFor(op);
        hasResOut = (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3);
        if (!hasResOut || !narrow.count(w[1])) { resWideOut = false; return; }
        bool keep = keepNarrow(op);
        if ((op == 81 || op == 77) && I.wc >= 4) {
            auto it = vTy.find(w[3]);
            if (it == vTy.end() || !narrow.count(it->second)) keep = true;
        }
        resWideOut = !keep && !constLike(op);
    };

    std::unordered_map<size_t, uint32_t> convNid;
    for (size_t ii = 0; ii < insts.size(); ++ii) {
        const Inst& I = insts[ii];
        const uint32_t* w = in + I.off;
        bool hasRes = false, resWide = false;
        computeResWide(I, hasRes, resWide);
        if (!hasRes || constLike(I.op)) continue;
        const uint32_t resTy = w[1], resId = w[2];
        if (!narrow.count(resTy) || !wide.count(resTy)) continue;
        const uint32_t nid = bound++;
        convNid[ii] = nid;
        if (resWide) { wId[resId] = resId; nId[resId] = nid; }
        else         { nId[resId] = resId; wId[resId] = nid; }
    }

    auto emitConv = [&](std::vector<uint32_t>& dst, uint32_t toTy, uint32_t fromTy, uint32_t newId, uint32_t srcId) {
        const bool toMat = mCnt.count(toTy)!=0, fromMat = mCnt.count(fromTy)!=0;
        if (!toMat || !fromMat) {
            dst.push_back((4u<<16)|convOp(toTy)); dst.push_back(toTy); dst.push_back(newId); dst.push_back(srcId);
            return;
        }
        const uint32_t C = mCnt[toTy], toCol = mCol[toTy], fromCol = mCol[fromTy];
        std::vector<uint32_t> cols;
        for (uint32_t i2 = 0; i2 < C; ++i2) {
            const uint32_t e = bound++;
            dst.push_back((5u<<16)|81u); dst.push_back(fromCol); dst.push_back(e); dst.push_back(srcId); dst.push_back(i2);
            const uint32_t c = bound++;
            dst.push_back((4u<<16)|convOp(toCol)); dst.push_back(toCol); dst.push_back(c); dst.push_back(e);
            cols.push_back(c);
        }
        dst.push_back(((3u+C)<<16)|80u); dst.push_back(toTy); dst.push_back(newId);
        for (uint32_t c : cols) dst.push_back(c);
    };

    auto emitBitcast16 = [&](std::vector<uint32_t>& dst, uint32_t dstOrigTy, uint32_t dstResId, uint32_t srcOrigTy, uint32_t wideSrcId) {
        const uint32_t wideDstTy = narrow.count(dstOrigTy) ? wide[dstOrigTy] : dstOrigTy;
        const uint32_t wideSrcTy = narrow.count(srcOrigTy) ? wide[srcOrigTy] : srcOrigTy;
        auto isFloat = [&](uint32_t t) { return tKind[scalarOf(t)] == 22; };
        auto isSigned = [&](uint32_t t) { return tKind[scalarOf(t)] == 21 && tSign[scalarOf(t)] != 0; };
        auto origBits = [&](uint32_t t) -> uint32_t { return tWidth[scalarOf(t)] * (vLen.count(t) ? vLen[t] : 1); };
        auto lanes = [&](uint32_t t) -> uint32_t { return vLen.count(t) ? vLen[t] : 1; };
        auto emitPack2x16 = [&](uint32_t f2Id) -> uint32_t {
            uint32_t p = bound++;
            dst.push_back((6u<<16)|12u); dst.push_back(u32Ty); dst.push_back(p); dst.push_back(glslId); dst.push_back(58u); dst.push_back(f2Id);
            return p;
        };
        auto emitUnpack2x16 = [&](uint32_t uId) -> uint32_t {
            uint32_t v = bound++;
            dst.push_back((6u<<16)|12u); dst.push_back(v2f32Ty); dst.push_back(v); dst.push_back(glslId); dst.push_back(62u); dst.push_back(uId);
            return v;
        };
        auto asU32 = [&](uint32_t id, uint32_t ty) -> uint32_t {
            if (ty == u32Ty) return id;
            uint32_t c = bound++;
            dst.push_back((4u<<16)|124u); dst.push_back(u32Ty); dst.push_back(c); dst.push_back(id);
            return c;
        };
        auto packInt16x2ToU32 = [&](uint32_t loId, uint32_t hiId, uint32_t elemWideTy) -> uint32_t {
            uint32_t lu = asU32(loId, elemWideTy), hu = asU32(hiId, elemWideTy);
            uint32_t lm = bound++, sh = bound++, r = bound++;
            dst.push_back((5u<<16)|199u); dst.push_back(u32Ty); dst.push_back(lm); dst.push_back(lu); dst.push_back(cFFFF_u32);
            dst.push_back((5u<<16)|196u); dst.push_back(u32Ty); dst.push_back(sh); dst.push_back(hu); dst.push_back(c16_u32);
            dst.push_back((5u<<16)|197u); dst.push_back(u32Ty); dst.push_back(r); dst.push_back(lm); dst.push_back(sh);
            return r;
        };
        auto cast32 = [&](uint32_t outId, uint32_t outTy, uint32_t inId, uint32_t inTy) {
            if (outTy == inTy) { dst.push_back((4u<<16)|83u); dst.push_back(outTy); dst.push_back(outId); dst.push_back(inId); }
            else { dst.push_back((4u<<16)|124u); dst.push_back(outTy); dst.push_back(outId); dst.push_back(inId); }
        };
        auto signOrMask16 = [&](uint32_t uId, bool sgn, uint32_t targetTy, uint32_t outId) {
            if (!sgn) {
                if (targetTy == u32Ty) { dst.push_back((5u<<16)|199u); dst.push_back(u32Ty); dst.push_back(outId); dst.push_back(uId); dst.push_back(cFFFF_u32); }
                else {
                    uint32_t m = bound++;
                    dst.push_back((5u<<16)|199u); dst.push_back(u32Ty); dst.push_back(m); dst.push_back(uId); dst.push_back(cFFFF_u32);
                    dst.push_back((4u<<16)|124u); dst.push_back(targetTy); dst.push_back(outId); dst.push_back(m);
                }
            } else {
                uint32_t shl = bound++;
                dst.push_back((5u<<16)|196u); dst.push_back(u32Ty); dst.push_back(shl); dst.push_back(uId); dst.push_back(c16_u32);
                if (targetTy == u32Ty) { dst.push_back((5u<<16)|195u); dst.push_back(u32Ty); dst.push_back(outId); dst.push_back(shl); dst.push_back(c16_u32); }
                else {
                    uint32_t sar = bound++;
                    dst.push_back((5u<<16)|195u); dst.push_back(u32Ty); dst.push_back(sar); dst.push_back(shl); dst.push_back(c16_u32);
                    dst.push_back((4u<<16)|124u); dst.push_back(targetTy); dst.push_back(outId); dst.push_back(sar);
                }
            }
        };

        const uint32_t sb = origBits(srcOrigTy), sl = lanes(srcOrigTy);
        const uint32_t nWords32 = (sb == 64) ? 2u : 1u;
        uint32_t words[2]{};
        if (isFloat(srcOrigTy)) {
            if (sl == 1) {
                uint32_t v2 = bound++;
                dst.push_back((5u<<16)|80u); dst.push_back(v2f32Ty); dst.push_back(v2); dst.push_back(wideSrcId); dst.push_back(c0_f32);
                words[0] = emitPack2x16(v2);
            } else if (sl == 2) {
                if (wideSrcTy == v2f32Ty) words[0] = emitPack2x16(wideSrcId);
                else {
                    uint32_t e0 = bound++, e1 = bound++, v2 = bound++;
                    dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e0); dst.push_back(wideSrcId); dst.push_back(0);
                    dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e1); dst.push_back(wideSrcId); dst.push_back(1);
                    dst.push_back((5u<<16)|80u); dst.push_back(v2f32Ty); dst.push_back(v2); dst.push_back(e0); dst.push_back(e1);
                    words[0] = emitPack2x16(v2);
                }
            } else {
                uint32_t e0=bound++,e1=bound++,e2=bound++,e3=bound++,v0=bound++,v1=bound++;
                dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e0); dst.push_back(wideSrcId); dst.push_back(0);
                dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e1); dst.push_back(wideSrcId); dst.push_back(1);
                dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e2); dst.push_back(wideSrcId); dst.push_back(2);
                dst.push_back((5u<<16)|81u); dst.push_back(floatTy); dst.push_back(e3); dst.push_back(wideSrcId); dst.push_back(3);
                dst.push_back((5u<<16)|80u); dst.push_back(v2f32Ty); dst.push_back(v0); dst.push_back(e0); dst.push_back(e1);
                dst.push_back((5u<<16)|80u); dst.push_back(v2f32Ty); dst.push_back(v1); dst.push_back(e2); dst.push_back(e3);
                words[0] = emitPack2x16(v0); words[1] = emitPack2x16(v1);
            }
        } else if (tWidth[scalarOf(srcOrigTy)] == 16) {
            const uint32_t elemTy = vComp.count(wideSrcTy) ? vComp[wideSrcTy] : wideSrcTy;
            if (sl == 1) { words[0] = asU32(wideSrcId, elemTy); }
            else if (sl == 2) {
                uint32_t e0 = bound++, e1 = bound++;
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e0); dst.push_back(wideSrcId); dst.push_back(0);
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e1); dst.push_back(wideSrcId); dst.push_back(1);
                words[0] = packInt16x2ToU32(e0, e1, elemTy);
            } else {
                uint32_t e0=bound++,e1=bound++,e2=bound++,e3=bound++;
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e0); dst.push_back(wideSrcId); dst.push_back(0);
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e1); dst.push_back(wideSrcId); dst.push_back(1);
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e2); dst.push_back(wideSrcId); dst.push_back(2);
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e3); dst.push_back(wideSrcId); dst.push_back(3);
                words[0] = packInt16x2ToU32(e0, e1, elemTy); words[1] = packInt16x2ToU32(e2, e3, elemTy);
            }
        } else {
            if (sl == 1) { words[0] = asU32(wideSrcId, wideSrcTy); }
            else {
                const uint32_t elemTy = vComp[wideSrcTy];
                uint32_t e0 = bound++, e1 = bound++;
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e0); dst.push_back(wideSrcId); dst.push_back(0);
                dst.push_back((5u<<16)|81u); dst.push_back(elemTy); dst.push_back(e1); dst.push_back(wideSrcId); dst.push_back(1);
                words[0] = asU32(e0, elemTy); words[1] = asU32(e1, elemTy);
            }
        }

        const uint32_t dl = lanes(dstOrigTy);
        if (isFloat(dstOrigTy)) {
            if (dl == 1) {
                uint32_t v2 = emitUnpack2x16(words[0]);
                dst.push_back((5u<<16)|81u); dst.push_back(wideDstTy); dst.push_back(dstResId); dst.push_back(v2); dst.push_back(0);
            } else if (dl == 2) {
                if (wideDstTy == v2f32Ty) { dst.push_back((6u<<16)|12u); dst.push_back(v2f32Ty); dst.push_back(dstResId); dst.push_back(glslId); dst.push_back(62u); dst.push_back(words[0]); }
                else { uint32_t v2 = emitUnpack2x16(words[0]); cast32(dstResId, wideDstTy, v2, v2f32Ty); }
            } else {
                uint32_t v0 = emitUnpack2x16(words[0]), v1 = emitUnpack2x16(words[1]);
                dst.push_back((9u<<16)|79u); dst.push_back(wideDstTy); dst.push_back(dstResId);
                dst.push_back(v0); dst.push_back(v1); dst.push_back(0); dst.push_back(1); dst.push_back(2); dst.push_back(3);
            }
        } else if (tWidth[scalarOf(dstOrigTy)] == 16) {
            const bool sgn = isSigned(dstOrigTy);
            const uint32_t elemTy = vComp.count(wideDstTy) ? vComp[wideDstTy] : wideDstTy;
            if (dl == 1) { signOrMask16(words[0], sgn, elemTy, dstResId); }
            else if (dl == 2) {
                uint32_t hiSh = bound++, l = bound++, h = bound++;
                dst.push_back((5u<<16)|194u); dst.push_back(u32Ty); dst.push_back(hiSh); dst.push_back(words[0]); dst.push_back(c16_u32);
                signOrMask16(words[0], sgn, elemTy, l); signOrMask16(hiSh, sgn, elemTy, h);
                dst.push_back((5u<<16)|80u); dst.push_back(wideDstTy); dst.push_back(dstResId); dst.push_back(l); dst.push_back(h);
            } else {
                uint32_t hi0 = bound++, hi1 = bound++, c0 = bound++, c1 = bound++, c2 = bound++, c3 = bound++;
                dst.push_back((5u<<16)|194u); dst.push_back(u32Ty); dst.push_back(hi0); dst.push_back(words[0]); dst.push_back(c16_u32);
                dst.push_back((5u<<16)|194u); dst.push_back(u32Ty); dst.push_back(hi1); dst.push_back(words[1]); dst.push_back(c16_u32);
                signOrMask16(words[0], sgn, elemTy, c0); signOrMask16(hi0, sgn, elemTy, c1);
                signOrMask16(words[1], sgn, elemTy, c2); signOrMask16(hi1, sgn, elemTy, c3);
                dst.push_back((7u<<16)|80u); dst.push_back(wideDstTy); dst.push_back(dstResId); dst.push_back(c0); dst.push_back(c1); dst.push_back(c2); dst.push_back(c3);
            }
        } else {
            if (nWords32 == 1) { cast32(dstResId, wideDstTy, words[0], u32Ty); }
            else {
                const uint32_t elemTy = vComp[wideDstTy];
                uint32_t e0 = bound++, e1 = bound++;
                cast32(e0, elemTy, words[0], u32Ty); cast32(e1, elemTy, words[1], u32Ty);
                dst.push_back((5u<<16)|80u); dst.push_back(wideDstTy); dst.push_back(dstResId); dst.push_back(e0); dst.push_back(e1);
            }
        }
    };

    out.clear(); out.reserve(n + 4096);
    out.insert(out.end(), in, in + 5);
    std::vector<uint32_t> pend;
    bool extrasDone = false, preMemDone = false, inBlock = false;
    auto flush = [&]() { if (inBlock && !pend.empty()) out.insert(out.end(), pend.begin(), pend.end()); pend.clear(); };
    auto isTerm = [](uint32_t op) { return (op >= 249 && op <= 255) || op == 4416; };

    for (size_t ii = 0; ii < insts.size(); ++ii) {
        const Inst& I = insts[ii];
        const uint32_t* w = in + I.off;
        const uint32_t op = I.op;
        if (op == 8 || op == 9) continue;
        if (dropCaps && op == 17 && I.wc >= 2 && (w[1]==9||w[1]==8||w[1]==22)) continue;
        if (op == 14 && !preMemDone) { preMemDone = true; out.insert(out.end(), preMemExtras.begin(), preMemExtras.end()); }
        if (op == 54 && !extrasDone) { extrasDone = true; out.insert(out.end(), extras.begin(), extras.end()); }
        if (op == 56 || op == 54) { pend.clear(); inBlock = false; out.insert(out.end(), w, w + I.wc); continue; }
        if (op != 245 && op != 248 && op != 55) flush();
        if (constLike(op)) { out.insert(out.end(), w, w + I.wc); continue; }

        if (op == 124 && I.wc >= 4) {
            const uint32_t dstTy = w[1], dstId = w[2], srcId = w[3];
            auto it = vTy.find(srcId);
            const uint32_t srcTy = (it != vTy.end()) ? it->second : 0;
            if (narrow.count(dstTy) || narrow.count(srcTy)) {
                emitBitcast16(out, dstTy, dstId, srcTy, W(srcId));
                if (convNid.count(ii)) emitConv(out, dstTy, wide[dstTy], convNid[ii], dstId);
                continue;
            }
        }

        std::vector<uint32_t> v(w, w + I.wc);
        bool hasRes = false, resWide = false;
        computeResWide(I, hasRes, resWide);
        const uint32_t resTy = hasRes ? w[1] : 0;
        const uint32_t resId = hasRes ? w[2] : 0;
        const bool resNarrow = hasRes && narrow.count(resTy) != 0;
        if (resWide) v[1] = wide[resTy];

        const bool keepArgsNarrow = !resWide && usesNarrowOperand(op);
        ForEachId(v.data(), op, I.wc, [&](size_t idx, char role){
            if (role != 'I') return;
            auto it = vTy.find(v[idx]);
            if (it == vTy.end() || !narrow.count(it->second)) return;
            v[idx] = keepArgsNarrow ? N(v[idx]) : W(v[idx]);
        });
        out.insert(out.end(), v.begin(), v.end());
        if (op == 248) { inBlock = true; flush(); continue; }
        if (isTerm(op)) { inBlock = false; continue; }
        if (!resNarrow || !convNid.count(ii)) continue;
        const uint32_t nid = convNid[ii];
        const uint32_t toTy = resWide ? resTy : wide[resTy];
        const uint32_t fromTy = resWide ? wide[resTy] : resTy;
        if (op == 55 || op == 245) emitConv(pend, toTy, fromTy, nid, resId);
        else if (inBlock) emitConv(out, toTy, fromTy, nid, resId);
    }
    flush();
    if (!extrasDone) out.insert(out.end(), extras.begin(), extras.end());
    out[3] = bound;
    return true;
}

	
	
	
// v9b: fp16 ALU -> fp32, 16-битная память не тронута.
// Исправлено: OpUndef=1, spec-const опкоды 48..51, все новые типы/константы
// идут ОДНИМ блоком перед первым OpFunction, матрицы поддержаны.
inline bool LowerFp16Alu(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err, bool dropF16Cap) {
    if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
    std::vector<Inst> insts;
    for (size_t i = 5; i < n;) {
        const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
        if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
        insts.push_back({ op, wc, i });
        i += wc;
    }
    auto constLike = [](uint32_t op) {
        return op == 1 || op == 41 || op == 42 || op == 43 || op == 44 || op == 45 ||
               op == 46 || op == 48 || op == 49 || op == 50 || op == 51;
    };
    auto keepNarrow = [](uint32_t op) {
        return op == 61 || op == 55 || op == 57 || (op >= 110 && op <= 115) ||
               (op >= 86 && op <= 98) || (op >= 100 && op <= 107);
    };
    uint32_t halfTy = 0, floatTy = 0;
    std::unordered_map<uint32_t, uint32_t> vComp, vLen, mCol, mCnt;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 22 && I.wc >= 3) {
            if (w[2] == 16) halfTy = w[1];
            if (w[2] == 32 && !floatTy) floatTy = w[1];
        }
        if (I.op == 23 && I.wc >= 4) { vComp[w[1]] = w[2]; vLen[w[1]] = w[3]; }
        if (I.op == 24 && I.wc >= 4) { mCol[w[1]] = w[2]; mCnt[w[1]] = w[3]; }
        if (I.op == 54) break;
    }
    if (!halfTy) { err = "no fp16 type"; return false; }
    std::unordered_set<uint32_t> narrow;
    narrow.insert(halfTy);
    for (const auto& kv : vComp) if (kv.second == halfTy) narrow.insert(kv.first);
    for (const auto& kv : mCol) if (narrow.count(kv.second)) narrow.insert(kv.first);
    std::unordered_map<uint32_t, uint32_t> vTy;
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        const char* k = KindsFor(I.op);
        if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3) vTy[w[2]] = w[1];
        if (I.op == 59 && I.wc >= 4) vTy[w[2]] = w[1];
    }
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        if (I.op == 52 && I.wc >= 3 && narrow.count(w[1])) { err = "fp16 spec-const-op"; return false; }
        if (I.op == 124 && I.wc >= 4) {
            if (narrow.count(w[1])) { err = "fp16 bitcast"; return false; }
            auto it = vTy.find(w[3]);
            if (it != vTy.end() && narrow.count(it->second)) { err = "fp16 bitcast"; return false; }
        }
    }
    uint32_t bound = in[3];
    std::vector<uint32_t> extras;
    if (!floatTy) {
        floatTy = bound++;
        extras.push_back((3u << 16) | 22u); extras.push_back(floatTy); extras.push_back(32);
    }
    std::unordered_map<uint32_t, uint32_t> wide;
    wide[halfTy] = floatTy;
    for (uint32_t t : narrow) {
        if (t == halfTy || !vLen.count(t)) continue;
        const uint32_t len = vLen[t];
        uint32_t f = 0;
        for (const auto& kv : vComp) if (kv.second == floatTy && vLen[kv.first] == len) { f = kv.first; break; }
        if (!f) {
            f = bound++;
            extras.push_back((4u << 16) | 23u); extras.push_back(f); extras.push_back(floatTy); extras.push_back(len);
            vComp[f] = floatTy; vLen[f] = len;
        }
        wide[t] = f;
    }
    for (uint32_t t : narrow) {
        if (!mCnt.count(t)) continue;
        const uint32_t cnt = mCnt[t], col = wide[mCol[t]];
        uint32_t f = 0;
        for (const auto& kv : mCol) if (kv.second == col && mCnt[kv.first] == cnt) { f = kv.first; break; }
        if (!f) {
            f = bound++;
            extras.push_back((4u << 16) | 24u); extras.push_back(f); extras.push_back(col); extras.push_back(cnt);
            mCol[f] = col; mCnt[f] = cnt;
        }
        wide[t] = f;
    }
    std::unordered_map<uint32_t, uint32_t> wId, nId;
    auto W = [&](uint32_t id) { auto it = wId.find(id); return it == wId.end() ? id : it->second; };
    auto N = [&](uint32_t id) { auto it = nId.find(id); return it == nId.end() ? id : it->second; };
    for (const Inst& I : insts) {
        if (!constLike(I.op) || I.wc < 3) continue;
        const uint32_t* w = in + I.off;
        const uint32_t t = w[1], id = w[2];
        if (!narrow.count(t)) continue;
        const uint32_t nid = bound++;
        if (I.op == 43 || I.op == 50) {
            if (I.wc < 4) { err = "short fp16 constant"; return false; }
            extras.push_back((4u << 16) | I.op); extras.push_back(wide[t]); extras.push_back(nid);
            extras.push_back(Float16BitsToFloat32Bits(static_cast<uint16_t>(w[3] & 0xFFFFu)));
        } else if (I.op == 44 || I.op == 51) {
            extras.push_back((I.wc << 16) | I.op); extras.push_back(wide[t]); extras.push_back(nid);
            for (uint32_t k = 3; k < I.wc; ++k) extras.push_back(W(w[k]));
        } else {
            extras.push_back((3u << 16) | I.op); extras.push_back(wide[t]); extras.push_back(nid);
        }
        wId[id] = nid; nId[id] = id;
    }
    auto emitConv = [&](std::vector<uint32_t>& dst, uint32_t toTy, uint32_t fromTy, uint32_t newId, uint32_t srcId) {
        const bool toMat = mCnt.count(toTy) != 0, fromMat = mCnt.count(fromTy) != 0;
        if (!toMat && !fromMat) {
            dst.push_back((4u << 16) | 115u); dst.push_back(toTy); dst.push_back(newId); dst.push_back(srcId);
            return;
        }
        if (!toMat || !fromMat || mCnt[toTy] != mCnt[fromTy]) {
            dst.push_back((4u << 16) | 115u); dst.push_back(toTy); dst.push_back(newId); dst.push_back(srcId);
            return;
        }
        const uint32_t C = mCnt[toTy], toCol = mCol[toTy], fromCol = mCol[fromTy];
        std::vector<uint32_t> cols;
        for (uint32_t i = 0; i < C; ++i) {
            const uint32_t e = bound++;
            dst.push_back((5u << 16) | 81u); dst.push_back(fromCol); dst.push_back(e); dst.push_back(srcId); dst.push_back(i);
            const uint32_t c = bound++;
            dst.push_back((4u << 16) | 115u); dst.push_back(toCol); dst.push_back(c); dst.push_back(e);
            cols.push_back(c);
        }
        dst.push_back(((3u + C) << 16) | 80u); dst.push_back(toTy); dst.push_back(newId);
        for (uint32_t c : cols) dst.push_back(c);
    };
    out.clear(); out.reserve(n + 2048);
    out.insert(out.end(), in, in + 5);
    std::vector<uint32_t> pend;
    bool extrasDone = false;
    auto flush = [&]() { out.insert(out.end(), pend.begin(), pend.end()); pend.clear(); };
    for (const Inst& I : insts) {
        const uint32_t* w = in + I.off;
        const uint32_t op = I.op;
		        // OpLine(8)/OpNoLine(9) - чисто отладочная информация. Вырезаем: они стоят
        // между OpFunctionParameter и OpLabel (и между OpPhi) и ломают порядок блока,
        // из-за чего отложенный OpFConvert оказывался вне блока.
        if (op == 8 || op == 9) continue;
        if (dropF16Cap && op == 17 && I.wc >= 2 && (w[1] == 9 || w[1] == 8 || w[1] == 22)) continue;
        if (op == 54 && !extrasDone) { extrasDone = true; out.insert(out.end(), extras.begin(), extras.end()); }
        if (op != 245 && op != 248 && op != 55) flush();
        std::vector<uint32_t> v(w, w + I.wc);
        const char* k = KindsFor(op);
        const bool hasRes = (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3);
        const uint32_t resTy = hasRes ? w[1] : 0;
        const uint32_t resId = hasRes ? w[2] : 0;
        const bool resNarrow = hasRes && narrow.count(resTy) != 0;
        bool keep = keepNarrow(op);
        if ((op == 81 || op == 77) && I.wc >= 4) {
            auto it = vTy.find(w[3]);
            if (it != vTy.end() && narrow.count(it->second)) keep = false;
        }
        const bool resWide = resNarrow && !keep && !constLike(op);
        if (resWide) v[1] = wide[resTy];
        ForEachId(v.data(), op, I.wc, [&](size_t idx, char role) {
            if (role != 'I') return;
            auto it = vTy.find(v[idx]);
            if (it == vTy.end() || !narrow.count(it->second)) return;
            v[idx] = resWide ? W(v[idx]) : N(v[idx]);
        });
        out.insert(out.end(), v.begin(), v.end());
        if (op == 248) { flush(); continue; }
        if (!resNarrow || constLike(op)) continue;
        const uint32_t nid = bound++;
        uint32_t toTy, fromTy;
        if (resWide) { toTy = resTy; fromTy = wide[resTy]; wId[resId] = resId; nId[resId] = nid; }
        else { toTy = wide[resTy]; fromTy = resTy; nId[resId] = resId; wId[resId] = nid; }
        if (op == 55 || op == 245) emitConv(pend, toTy, fromTy, nid, resId);
        else emitConv(out, toTy, fromTy, nid, resId);
    }
    flush();
    if (!extrasDone) out.insert(out.end(), extras.begin(), extras.end());
    out[3] = bound;
    return true;
}
	
	
	
	
	
	
	
	
	// Promote 16-bit floats to 32-bit (for GPUs without fp16 ALU / drivers that fail on fp16 subgroup ops).
    // Produces VALID SPIR-V: the 16-bit float type and the 32-bit float type are merged into one id,
    // every type that becomes identical afterwards (vectors, pointers, images, function types...) is merged too,
    // constants are converted, and OpFConvert between the (now equal) types becomes OpCopyObject.
    inline bool PromoteFloat16(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
        if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
        std::vector<Inst> insts;
        for (size_t i = 5; i < n;) {
            const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
            if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
            insts.push_back({ op, wc, i });
            i += wc;
        }
        std::unordered_set<uint32_t> f16;
        for (const Inst& I : insts) if (I.op == 22 && I.wc >= 3 && in[I.off + 2] == 16) f16.insert(in[I.off + 1]);
        if (f16.empty()) { err = "no 16-bit float"; return false; }
        // refuse modules where a 16-bit float is bit-cast to/from 16-bit integers (cannot be promoted safely)
            // Int16 сам по себе не мешает. Опасен только OpBitcast с участием 16-битного float.
    {
        std::unordered_set<uint32_t> f16Any = f16;
        for (const Inst& I : insts)
            if (I.op == 23 && I.wc >= 4 && f16.count(in[I.off + 2])) f16Any.insert(in[I.off + 1]);
        std::unordered_map<uint32_t, uint32_t> vType;
        for (const Inst& I : insts) {
            const char* k = KindsFor(I.op);
            if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3) vType[in[I.off + 2]] = in[I.off + 1];
            if (I.op == 59 && I.wc >= 4) vType[in[I.off + 2]] = in[I.off + 1];
        }
        for (const Inst& I : insts) {
            if (I.op != 124 || I.wc < 4) continue;
            if (f16Any.count(in[I.off + 1])) { err = "bitcast of 16-bit float"; return false; }
            auto it = vType.find(in[I.off + 3]);
            if (it != vType.end() && f16Any.count(it->second)) { err = "bitcast of 16-bit float"; return false; }
        }
    }

        std::unordered_map<uint32_t, uint32_t> remap;               // id -> surviving id
        auto R = [&](uint32_t id) { auto it = remap.find(id); return it == remap.end() ? id : it->second; };

        // survivor of the float types: the first declared one
        uint32_t surv = 0;
        for (const Inst& I : insts) {
            if (I.op == 22 && I.wc >= 3 && (in[I.off + 2] == 32 || in[I.off + 2] == 16)) {
                const uint32_t id = in[I.off + 1];
                if (!surv) surv = id; else remap[id] = surv;
            }
        }

        // type declarations that must be unique (non-aggregates)
        auto uniqueType = [](uint32_t op) {
            return op == 19 || op == 20 || op == 21 || op == 22 || op == 23 || op == 24 || op == 25 || op == 26 ||
                   op == 27 || op == 32 || op == 33;
        };
        std::map<std::vector<uint32_t>, uint32_t> seen;
        std::vector<char> drop(insts.size(), 0);
        // first pass: build keys in declaration order with progressively refined remap
        std::vector<std::vector<uint32_t>> rewritten(insts.size());
        for (size_t ii = 0; ii < insts.size(); ++ii) {
            const Inst& I = insts[ii];
            const uint32_t* w = in + I.off;
            rewritten[ii].assign(w, w + I.wc);
            if (I.op == 54) break;                                    // functions start: all types/constants are before
            // apply the remap to id operands so far
            ForEachId(rewritten[ii].data(), I.op, I.wc, [&](size_t idx, char role) {
                if (role != 'R') rewritten[ii][idx] = R(rewritten[ii][idx]);
            });
            if (!uniqueType(I.op)) continue;
            const uint32_t id = rewritten[ii][1];
            if (I.op == 22) {                                         // float: only the survivor is kept, as 32-bit
                if (id != surv) { drop[ii] = 1; continue; }
                rewritten[ii][2] = 32;
            }
            std::vector<uint32_t> key(rewritten[ii].begin() + 2, rewritten[ii].end());
            key.insert(key.begin(), I.op);
            if (I.op == 32 || I.op == 33 || I.op == 23 || I.op == 24 || I.op == 25 || I.op == 27) {}
            auto it = seen.find(key);
            if (it == seen.end()) { seen[key] = id; }
            else if (it->second != id) { remap[id] = it->second; drop[ii] = 1; }
        }
        // second pass over the whole module: apply the final remap, convert constants, fix conversions
        std::unordered_map<uint32_t, uint32_t> typeOf;               // value id -> result type id (remapped)
        out.clear();
        out.reserve(n);
        out.insert(out.end(), in, in + 5);
        std::set<std::vector<uint32_t>> decorations;
        bool pastTypes = false;
        for (size_t ii = 0; ii < insts.size(); ++ii) {
            const Inst& I = insts[ii];
            if (drop[ii]) continue;
            std::vector<uint32_t> w(in + I.off, in + I.off + I.wc);
            if (I.op == 54) pastTypes = true;
            const bool wasF16Const = (I.op == 43 || I.op == 48) && I.wc >= 4 && f16.count(in[I.off + 1]);
            ForEachId(w.data(), I.op, I.wc, [&](size_t idx, char role) {
                if (role != 'R') w[idx] = R(w[idx]);
            });
            if (I.op == 17 && I.wc >= 2 && (w[1] == 9 || w[1] == 8)) continue;                    // Float16 / Float16Buffer
            if (I.op == 22 && I.wc >= 3 && w[2] == 16) w[2] = 32;
            if (wasF16Const) w[3] = Float16BitsToFloat32Bits(static_cast<uint16_t>(in[I.off + 3] & 0xFFFFu));
            if (I.op == 71 || I.op == 72) {                                                        // drop exact duplicate decorations
                std::vector<uint32_t> key(w.begin() + 1, w.end());
                key.insert(key.begin(), I.op);
                if (!decorations.insert(key).second) continue;
            }
            const char* k = KindsFor(I.op);
            if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3) typeOf[w[2]] = w[1];
            if (I.op == 115 /*OpFConvert*/ && I.wc == 4) {
                auto it = typeOf.find(w[3]);
                if (it != typeOf.end() && it->second == w[1]) w[0] = (4u << 16) | 83u;            // OpCopyObject
            }
            out.insert(out.end(), w.begin(), w.end());
        }
        (void)pastTypes;
        return true;
    }

inline bool StripRayQuery(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
        if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
        std::vector<Inst> insts;
        for (size_t i = 5; i < n;) {
            const uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
            if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
            insts.push_back({ op, wc, i });
            i += wc;
        }

        std::unordered_set<uint32_t> removedTypes, removedValues;
        bool any = false;
        // ---- pass 1: find acceleration-structure / ray-query types and everything derived from them
        for (const Inst& I : insts) {
            const uint32_t* w = in + I.off;
            switch (I.op) {
            case 5341: case 4472: removedTypes.insert(w[1]); any = true; break;
            case 32: if (I.wc >= 4 && removedTypes.count(w[3])) removedTypes.insert(w[1]); break;
            case 28: if (I.wc >= 4 && removedTypes.count(w[2])) removedTypes.insert(w[1]); break;
            case 29: if (I.wc >= 3 && removedTypes.count(w[2])) removedTypes.insert(w[1]); break;
            case 30: for (uint32_t k = 2; k < I.wc; ++k) if (removedTypes.count(w[k])) { err = "struct contains AS/RayQuery"; return false; } break;
            case 33: for (uint32_t k = 2; k < I.wc; ++k) if (removedTypes.count(w[k])) { err = "function type uses AS/RayQuery"; return false; } break;
            case 59: if (I.wc >= 4 && removedTypes.count(w[1])) removedValues.insert(w[2]); break;
            default: {
                const char* k = KindsFor(I.op);
                if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3 && removedTypes.count(w[1])) {
                    if (I.op == 55 || I.op == 245 || I.op == 169 || I.op == 57) { err = "AS/RayQuery passed through function/phi/select"; return false; }
                    removedValues.insert(w[2]);
                }
            }
            }
        }
        if (!any) { err = "no ray query"; return false; }

        // ---- pass 2: which result types need a null constant for replaced instructions
        uint32_t bound = in[3];
        std::unordered_map<uint32_t, uint32_t> nullFor;     // type id -> new constant id
        std::vector<uint32_t> nullOrder;
        for (const Inst& I : insts) {
            if (!IsRqValueOp(I.op)) continue;
            const uint32_t* w = in + I.off;
            if (I.wc < 4) { err = "short rq op"; return false; }
            if (removedTypes.count(w[1])) { err = "rq op returns removed type"; return false; }
            if (!nullFor.count(w[1])) { nullFor[w[1]] = bound++; nullOrder.push_back(w[1]); }
        }

        // ---- pass 3: emit
        out.clear();
        out.reserve(n);
        out.insert(out.end(), in, in + 5);
        bool constsEmitted = false;
        auto emitConsts = [&]() {
            if (constsEmitted) return;
            constsEmitted = true;
            for (uint32_t t : nullOrder) { out.push_back((3u << 16) | 46u); out.push_back(t); out.push_back(nullFor[t]); }
        };
        auto isRemoved = [&](uint32_t id) { return removedTypes.count(id) || removedValues.count(id); };

        for (const Inst& I : insts) {
            const uint32_t* w = in + I.off;
            const uint32_t op = I.op;
            if (op == 54 /*OpFunction*/) emitConsts();
            if (op == 17 && I.wc >= 2 && (w[1] == 4472 || w[1] == 4478 || w[1] == 4471)) continue;   // capabilities
            if (op == 10 && I.wc >= 2) {                                                              // extension
                const char* s = reinterpret_cast<const char*>(w + 1);
                if (strncmp(s, "SPV_KHR_ray_query", (I.wc - 1) * 4) == 0 && strlen("SPV_KHR_ray_query") <= (I.wc - 1) * 4) continue;
            }
            if ((op == 5 /*Name*/ || op == 71 /*Decorate*/ || op == 72 /*MemberDecorate*/ || op == 6 /*MemberName*/ ||
                 op == 332 /*DecorateId*/ || op == 5632 /*DecorateString*/ || op == 5633 /*MemberDecorateString*/) &&
                I.wc >= 2 && isRemoved(w[1])) continue;
            if (op == 5341 || op == 4472) continue;
            if ((op == 32 || op == 28 || op == 29) && I.wc >= 3 && removedTypes.count(w[1])) continue;
            if (op == 59 && I.wc >= 3 && removedValues.count(w[2])) continue;
            if (IsRqVoidOp(op)) continue;
            if (IsRqValueOp(op)) {
                out.push_back((4u << 16) | 83u);           // OpCopyObject
                out.push_back(w[1]); out.push_back(w[2]); out.push_back(nullFor[w[1]]);
                continue;
            }
            {
                const char* k = KindsFor(op);
                if (k && k[0] == 'T' && k[1] == 'R' && I.wc >= 3 && removedValues.count(w[2])) continue;   // loads/access chains of AS
            }
            if (op == 15 /*OpEntryPoint*/) {
                // keep everything up to the name string, filter the interface list
                size_t pos = 3;
                while (pos < I.wc) {
                    const uint32_t x = w[pos++];
                    if ((x & 0xFF) == 0 || (x & 0xFF00) == 0 || (x & 0xFF0000) == 0 || (x & 0xFF000000) == 0) break;
                }
                std::vector<uint32_t> tmp(w, w + pos);
                for (size_t k2 = pos; k2 < I.wc; ++k2) if (!removedValues.count(w[k2])) tmp.push_back(w[k2]);
                tmp[0] = (static_cast<uint32_t>(tmp.size()) << 16) | 15u;
                out.insert(out.end(), tmp.begin(), tmp.end());
                continue;
            }
            out.insert(out.end(), w, w + I.wc);
        }
        out[3] = bound;

        // ---- verify nothing in the output still refers to removed ids
        for (size_t i = 5; i < out.size();) {
            const uint32_t wc = out[i] >> 16, op = out[i] & 0xFFFF;
            bool bad = false;
            ForEachId(out.data() + i, op, wc, [&](size_t idx, char role) {
                if (role != 'R' && isRemoved(out[i + idx])) bad = true;
            });
            if (bad) { err = "dangling reference to removed id in op " + std::to_string(op); return false; }
            i += wc;
        }
        return true;
    }
}

    // ---- SPIR-V module info ---------------------------------------------------
    struct SpvInfo {
        uint32_t version = 0;
        uint32_t model = 0xFFFFFFFFu;
        std::string caps;
        bool deny = false;
        std::string reason;
        bool rqOnly = false;      // uses RayQuery but no other ray-tracing feature (can be rewritten)
        bool denyOther = false;   // denied for a reason other than ray tracing
        bool rewritten = false;
        size_t origWords = 0;     // [SWRT] size of the ORIGINAL module (software RT makes the sent code larger)
        std::shared_ptr<std::vector<uint32_t>> code;   // final SPIR-V sent to the driver (for post-mortem dumps)
    };
    SRWLOCK g_spvLock = SRWLOCK_INIT;
    std::map<uint64_t, SpvInfo>* g_spvMap = nullptr;

    const char* CapName(uint32_t c) {
        switch (c) {
        case 1: return "Shader"; case 9: return "Float16"; case 10: return "Float64"; case 11: return "Int64";
        case 22: return "Int16"; case 39: return "Int8"; case 61: return "GroupNonUniform";
        case 62: return "GroupNonUniformVote"; case 63: return "GroupNonUniformArithmetic";
        case 64: return "GroupNonUniformBallot"; case 65: return "GroupNonUniformShuffle";
        case 66: return "GroupNonUniformShuffleRelative"; case 67: return "GroupNonUniformClustered";
        case 68: return "GroupNonUniformQuad"; case 4422: return "FragmentShadingRate";
        case 4433: return "StorageBuffer16BitAccess"; case 4434: return "UniformAndStorageBuffer16BitAccess";
        case 4435: return "StoragePushConstant16"; case 4436: return "StorageInputOutput16";
        case 4472: return "RayQueryKHR"; case 4479: return "RayTracingKHR";
        case 5283: return "MeshShadingEXT"; case 5301: return "ShaderNonUniform";
        case 5302: return "RuntimeDescriptorArray"; case 5345: return "VulkanMemoryModel";
        case 5347: return "PhysicalStorageBufferAddresses"; case 5340: return "RayTracingNV";
        case 6016: return "DotProductInputAll"; case 6017: return "DotProductInput4x8Bit";
        case 6018: return "DotProductInput4x8BitPacked"; case 6019: return "DotProduct";
        default: return nullptr;
        }
    }

    SpvInfo AnalyzeSpirv(const uint32_t* w, size_t nWords) {
        SpvInfo info;
        if (!w || nWords < 5 || w[0] != 0x07230203u) { info.reason = "not SPIR-V"; return info; }
        info.version = w[1];
        bool f16 = false, i16 = false, dot = false, rt = false, mesh = false, rqCap = false, rtHard = false;
        std::vector<uint32_t> capIds;
        size_t i = 5;
        char buf[48];
        while (i < nWords) {
            const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
            if (wc == 0 || i + wc > nWords) break;
            if (op == 54) break;                                   // OpFunction: end of declarations
            if (op == 17 && wc >= 2) {                             // OpCapability
                const uint32_t c = w[i + 1];
                capIds.push_back(c);
                const char* n = CapName(c);
                if (n) info.caps += n; else { sprintf_s(buf, sizeof(buf), "cap%u", c); info.caps += buf; }
                info.caps += ' ';
                if (c == 9) f16 = true;
                if (c == 22) i16 = true;
                if (c >= 6016 && c <= 6019) dot = true;
                if (c == 4472 || c == 4479 || c == 5340 || c == 5341 || c == 5342) rt = true;
                if (c == 4472) rqCap = true;
                if (c == 4479 || c == 5340 || c == 5341 || c == 5342) rtHard = true;
                if (c == 5283 || c == 5266) mesh = true;
            } else if (op == 15 && wc >= 2 && info.model == 0xFFFFFFFFu) {   // OpEntryPoint
                info.model = w[i + 1];
                if (info.model >= 5267 && info.model <= 5268) mesh = true;  // Task/Mesh NV
                if (info.model >= 5313 && info.model <= 5321) { rt = true; rtHard = true; }     // RT stages
                if (info.model == 5364 || info.model == 5365) mesh = true;   // Task/Mesh EXT
            }
            i += wc;
        }
        if (g_spvCfg.denyFloat16 && f16)    { info.deny = true; info.denyOther = true; info.reason += "Float16 "; }
        if (g_spvCfg.denyInt16 && i16)      { info.deny = true; info.denyOther = true; info.reason += "Int16 "; }
        if (g_spvCfg.denyDotProduct && dot) { info.deny = true; info.denyOther = true; info.reason += "DotProduct "; }
        if (g_spvCfg.denyRayTracing && rt)  { info.deny = true; info.reason += "RayTracing "; }
        if (g_spvCfg.denyMesh && mesh)      { info.deny = true; info.denyOther = true; info.reason += "Mesh "; }
        info.rqOnly = rqCap && !rtHard;
        for (uint32_t c : capIds) {
            for (int k = 0; k < g_spvCfg.nDenyAnyCaps; ++k)
                if (g_spvCfg.denyAnyCaps[k] == c) {
                    // Cap46 SampledBuffer: prefer rewrite path when RayQuery is present; otherwise soft-deny.
                    if (c == 46 && rqCap && !rtHard) {
                        info.reason += "AnyCap46(soft) ";
                    } else {
                        info.deny = true; info.denyOther = true;
                        info.reason += "AnyCap" + std::to_string(c) + " ";
                    }
                }
            if (info.model == 5)
                for (int k = 0; k < g_spvCfg.nDenyComputeCaps; ++k)
                    if (g_spvCfg.denyComputeCaps[k] == c) { info.deny = true; info.denyOther = true; info.reason += "ComputeCap" + std::to_string(c) + " "; }
        }
        if (info.model == 5 && g_spvCfg.nDenyComputeAll > 0) {
            bool all = true;
            for (int k = 0; k < g_spvCfg.nDenyComputeAll && all; ++k) {
                bool found = false;
                for (uint32_t c : capIds) if (c == g_spvCfg.denyComputeAll[k]) { found = true; break; }
                if (!found) all = false;
            }
            if (all) { info.deny = true; info.denyOther = true; info.reason += "ComputeCombo "; }
        }
        return info;
    }

    bool GetSpvInfo(VkShaderModule m, SpvInfo& out) {
        bool ok = false;
        AcquireSRWLockShared(&g_spvLock);
        if (g_spvMap) {
            auto it = g_spvMap->find(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m)));
            if (it != g_spvMap->end()) { out = it->second; ok = true; }
        }
        ReleaseSRWLockShared(&g_spvLock);
        return ok;
    }
    // ---- learned "toxic" pipeline signatures ----------------------------------------------------------
    // The old driver needs ~5 s and then fails (-13) on some shader families (GroupNonUniform together with fp16/int16/
    // cube-array caps). Instead of faking such pipelines blindly, the proxy LEARNS: a signature is only treated as toxic
    // after 3 failures with no success; before that the driver gets a chance (so working shaders are not lost).
    struct SigStat { int failed = 0, ok = 0; };
    SRWLOCK g_sigLock = SRWLOCK_INIT;
    std::map<std::string, SigStat>& SigMap() { static auto* m = new std::map<std::string, SigStat>(); return *m; }
    // Only shader families known to be risky get a signature (others always go to the driver).
    // Only shader families known to be risky get a signature (others always go to the driver).
 std::string SigOfModule(VkShaderModule m, uint32_t stage) {
     SpvInfo si;
     if (!GetSpvInfo(m, si)) return std::string();
if (si.rewritten) return std::string();
     const bool gnu = si.caps.find("GroupNonUniform") != std::string::npos;
     const bool heavy = si.caps.find("Float16") != std::string::npos ||
                        si.caps.find("Int16") != std::string::npos ||
                        si.caps.find("cap45") != std::string::npos ||
                        si.caps.find("cap46") != std::string::npos;
     if (!gnu || !heavy) return std::string();
     return std::to_string(stage) + ":" + si.caps;
 }
    bool SigToxic(const std::string& sig) {
        if (sig.empty()) return false;
        bool t = false;
        AcquireSRWLockShared(&g_sigLock);
        auto it = SigMap().find(sig);
        if (it != SigMap().end()) t = it->second.failed >= 3 && it->second.ok == 0;
        ReleaseSRWLockShared(&g_sigLock);
        return t;
    }
    void SigRecord(const std::string& sig, bool ok) {
        if (sig.empty()) return;
        AcquireSRWLockExclusive(&g_sigLock);
        SigStat& st = SigMap()[sig];
        if (ok) ++st.ok; else ++st.failed;
        const bool nowToxic = !ok && st.failed == 3 && st.ok == 0;
        ReleaseSRWLockExclusive(&g_sigLock);
        if (nowToxic) LogLine("SIGNATURE learned as TOXIC (3 driver failures, 0 successes): %s", sig.c_str());
    }

    void PutSpvInfo(VkShaderModule m, const SpvInfo& info) {
        AcquireSRWLockExclusive(&g_spvLock);
        if (!g_spvMap) g_spvMap = new std::map<uint64_t, SpvInfo>();
        (*g_spvMap)[static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m))] = info;
        ReleaseSRWLockExclusive(&g_spvLock);
    }
    bool GetSpvInfoPublic(VkShaderModule m, std::string& caps, bool& risky) {
        SpvInfo si;
        if (!GetSpvInfo(m, si)) return false;
        caps = si.caps;
        const bool gnu = si.caps.find("GroupNonUniform") != std::string::npos;
        const bool heavy = si.caps.find("Float16") != std::string::npos || si.caps.find("Int16") != std::string::npos ||
                           si.caps.find("cap45") != std::string::npos || si.caps.find("cap46") != std::string::npos;
        risky = gnu && heavy;
        return true;
    }
    void EraseSpvInfo(VkShaderModule m) {
        AcquireSRWLockExclusive(&g_spvLock);
        if (g_spvMap) g_spvMap->erase(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m)));
        ReleaseSRWLockExclusive(&g_spvLock);
    }

    // Short text describing a stage module, for breadcrumbs.
    void DescribeModule(VkShaderModule m, char* out, size_t cap) {
        SpvInfo si;
        if (IsFake(m)) { _snprintf_s(out, cap, _TRUNCATE, "FAKE"); return; }
        if (GetSpvInfo(m, si)) _snprintf_s(out, cap, _TRUNCATE, "%p v%x model=%u [%s]", (void*)m, si.version, si.model, si.caps.c_str());
        else _snprintf_s(out, cap, _TRUNCATE, "%p (unknown)", (void*)m);
    }

    // ---- run a driver call with crash recovery + optional serialization -----------
    // Why not only __try/__except: the game installs its own vectored exception handler that kills the
    // process BEFORE frame-based SEH is consulted. So recovery is done from OUR vectored handler:
    // the thread's register context is saved before the driver call (like setjmp) and restored by the
    // handler when the driver faults inside the call.
    SRWLOCK g_compileLock = SRWLOCK_INIT;
    struct ThreadGuard { CONTEXT ctx; volatile LONG active; volatile LONG tripped; };
    thread_local ThreadGuard t_guard;
    thread_local char t_job[256];
    thread_local int t_timeoutOverrideSec = 0;   // [v3.2] per-call compile timeout (compute), 0 = use CompileTimeoutSec
    inline void SetJob(const char* fmt, ...) noexcept {
        va_list a; va_start(a, fmt);
        vsnprintf_s(t_job, sizeof(t_job), _TRUNCATE, fmt, a);
        va_end(a);
    }
#ifdef _MSC_VER
#define VKP_NOINLINE __declspec(noinline)
#else
#define VKP_NOINLINE __attribute__((noinline))
#endif
    template<class F> VKP_NOINLINE bool GuardedRun(F& f) {
        t_guard.tripped = 0;
        t_guard.active = 0;
        RtlCaptureContext(&t_guard.ctx);            // execution resumes here after a recovered fault
        if (t_guard.tripped) { t_guard.active = 0; t_guard.tripped = 0; return false; }
        t_guard.active = 1;
        f();
        t_guard.active = 0;
        return true;
    }
#ifdef _MSC_VER
    template<class F> bool SehRun(F& f) {
        __try { return GuardedRun(f); }
        __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    }
#else
    template<class F> bool SehRun(F& f) { return GuardedRun(f); }
#endif
    // Global count of recovered driver crashes. After a threshold we become more aggressive
    // and force remaining modern shaders/pipelines to FAKE so the game can reach the menu.
    volatile LONG g_driverCrashCount = 0;

    // Returns false if the driver crashed (access violation inside the call).
    // [FIX] Every driver call used to take ONE global exclusive lock. The log shows ~6 s per heavy pipeline, so the
    // game's worker threads compiled its thousands of PSOs strictly one after another: the "black screen after the
    // intro" is that PSO warm-up and it took ~N x 6 s. Now a counting semaphore allows a few parallel compiles
    // (SerializeCompile=1 in VulkanProxy.ini restores the old one-at-a-time behaviour, CompileThreads=N overrides).
    // ======================================================================================================
    // [HANGFIX] In-flight registry, watchdog, timed driver calls and hang quarantine.
    //
    // Why: the old DriverCall() only wrote to the log AFTER the driver returned. A driver call that never returns
    // (LLPC/LLVM compiling a huge ray-query fragment shader on GCN2) therefore left NO trace at all: the log simply
    // stopped, the 4 compile slots stayed occupied for ever, every other pipeline waited on the gate, the game's
    // main thread waited for its PSO jobs -> black screen. The pipeline cache stayed at 8 KB because it was only
    // written from the present call / after a graphics pipeline, and never earlier than 120 s after the last save.
    // ======================================================================================================
    struct InFlight { volatile LONG state; volatile LONG abandoned; DWORD tid; ULONGLONG t0; ULONGLONG lastWarn; char job[240]; };
    constexpr int kInFlightMax = 256;
    InFlight g_inflight[kInFlightMax];
    volatile LONG g_gateBusy = 0;
    volatile LONG g_abandoned = 0;      // stuck driver calls that were abandoned and are still running
    volatile LONG g_timedOutTotal = 0;
    int g_gateSize = 0;
    namespace pcache { void MaybeSave(bool force); }

    int InFlightBegin(const char* job) {
        for (int i = 0; i < kInFlightMax; ++i) {
            if (InterlockedCompareExchange(&g_inflight[i].state, 1, 0) == 0) {
                g_inflight[i].abandoned = 0;
                g_inflight[i].tid = GetCurrentThreadId();
                g_inflight[i].t0 = GetTickCount64();
                g_inflight[i].lastWarn = g_inflight[i].t0;
                strncpy_s(g_inflight[i].job, sizeof(g_inflight[i].job), job ? job : "", _TRUNCATE);
                InterlockedExchange(&g_inflight[i].state, 2);
                return i;
            }
        }
        return -1;
    }
    void InFlightEnd(int i) { if (i >= 0 && i < kInFlightMax) InterlockedExchange(&g_inflight[i].state, 0); }

    void DumpAllThreads(const char* why);   // [v3.2] defined after ModuleOffset()
    DWORD WINAPI WatchdogProc(LPVOID) {
        ULONGLONG lastBeat = 0;
        ULONGLONG lastStall = 0;
        int stallDumps = 0;
        LONG prevPresents = 0, prevSubmits = 0;
        for (;;) {
            Sleep(5000);
            const ULONGLONG now = GetTickCount64();
            int busy = 0;
            for (int i = 0; i < kInFlightMax; ++i) {
                InFlight& e = g_inflight[i];
                if (e.state != 2) continue;
                ++busy;
                const ULONGLONG age = now - e.t0;
                if (age >= 20000 && now - e.lastWarn >= 30000) {
                    e.lastWarn = now;
                    char job[240];
                    strncpy_s(job, sizeof(job), e.job, _TRUNCATE);
                    LogLine("WATCHDOG: driver call still running after %llu s (thread %lu)%s: %s",
                        static_cast<unsigned long long>(age / 1000), static_cast<unsigned long>(e.tid),
                        e.abandoned ? " [ABANDONED - result is ignored]" : "", job);
                }
            }
            if (now - lastBeat >= 60000) {
                lastBeat = now;
                LogLine("WATCHDOG heartbeat: in-flight=%d gate=%ld/%d abandoned-running=%ld timed-out-total=%ld recovered-crashes=%ld",
                    busy, static_cast<long>(g_gateBusy), g_gateSize, static_cast<long>(g_abandoned),
                    static_cast<long>(g_timedOutTotal), static_cast<long>(g_driverCrashCount));
                // [v3.2] tells "frozen" apart from "running but black": are frames still being presented?
                const LONG p = g_presentCount, sb = g_submitCount;
                const ULONGLONG lp = static_cast<ULONGLONG>(g_lastPresentTick);
                LogLine("FRAMES: presents=%ld (+%ld in last minute) submits=%ld (+%ld) lastPresentResult=%ld lastSubmitResult=%ld secondsSinceLastPresent=%llu",
                    static_cast<long>(p), static_cast<long>(p - prevPresents), static_cast<long>(sb), static_cast<long>(sb - prevSubmits),
                    static_cast<long>(g_lastPresentRes), static_cast<long>(g_lastSubmitRes),
                    static_cast<unsigned long long>(lp ? (now - lp) / 1000 : 0));
                prevPresents = p; prevSubmits = sb;
            }
            // [v3.2] stall detector: the game stopped presenting -> show where every thread is waiting.
            if (g_spvCfg.stallDumpSec > 0 && g_presentCount > 0 && stallDumps < 6) {
                const ULONGLONG lp = static_cast<ULONGLONG>(g_lastPresentTick);
                if (lp && now - lp >= static_cast<ULONGLONG>(g_spvCfg.stallDumpSec) * 1000u && now - lastStall >= 60000) {
                    lastStall = now; ++stallDumps;
                    char why[160];
                    sprintf_s(why, sizeof(why), "no vkQueuePresentKHR for %llu s (presents=%ld submits=%ld lastPresentResult=%ld lastSubmitResult=%ld)",
                        static_cast<unsigned long long>((now - lp) / 1000), static_cast<long>(g_presentCount),
                        static_cast<long>(g_submitCount), static_cast<long>(g_lastPresentRes), static_cast<long>(g_lastSubmitRes));
                    DumpAllThreads(why);
                }
            }
        }
        return 0;
    }
    // The pipeline cache is saved from its own thread: it must never be blocked by (or block) the game's threads.
    DWORD WINAPI CacheSaverProc(LPVOID) {
        for (;;) { Sleep(15000); pcache::MaybeSave(false); }
        return 0;
    }

    HANDLE CompileGate() {
        static HANDLE h = []() -> HANDLE {
            LoadSpvConfig();
            int n = g_spvCfg.compileThreads;
            if (n <= 0) {
                if (g_spvCfg.serializeCompile == 1) n = 1;
                else if (g_spvCfg.serializeCompile == 0) n = 64;
                else {
                    const DWORD cpu = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
                    n = static_cast<int>(cpu / 2);
                    if (n < 2) n = 2;
                    if (n > 4) n = 4;
                }
            }
            g_gateSize = n;
            LogLine("Driver compile gate: up to %d concurrent driver calls; compile timeout=%d s watchdog=%d maxAbandoned=%d heavyWords=%d",
                n, g_spvCfg.compileTimeoutSec, g_spvCfg.watchdog, g_spvCfg.maxAbandoned, g_spvCfg.heavyShaderWords);
            if (g_spvCfg.watchdog) {
                HANDLE w = CreateThread(nullptr, 0, WatchdogProc, nullptr, 0, nullptr);
                if (w) CloseHandle(w);
                HANDLE sv = CreateThread(nullptr, 0, CacheSaverProc, nullptr, 0, nullptr);
                if (sv) CloseHandle(sv);
            }
            return CreateSemaphoreW(nullptr, n, n, nullptr);
        }();
        return h;
    }

    inline void GateEnter(HANDLE gate) { if (gate) { WaitForSingleObject(gate, INFINITE); InterlockedIncrement(&g_gateBusy); } }
    inline void GateLeave(HANDLE gate) { if (gate) { InterlockedDecrement(&g_gateBusy); ReleaseSemaphore(gate, 1, nullptr); } }

    template<class F> bool DriverCall(F&& f, bool gated = true) {
        HANDLE gate = CompileGate();
        if (!gated) gate = nullptr;
        GateEnter(gate);
        const ULONGLONG t0 = GetTickCount64();
        const int slot = InFlightBegin(t_job);
        bool ok = true;
        if (g_spvCfg.catchDriverCrash) ok = SehRun(f); else f();
        InFlightEnd(slot);
        const ULONGLONG dt = GetTickCount64() - t0;
        GateLeave(gate);
        if (!ok) {
            const LONG c = InterlockedIncrement(&g_driverCrashCount);
            LogLine("CAUGHT driver crash after %llu ms (total recovered=%ld). job: %s",
                static_cast<unsigned long long>(dt), (long)c, t_job);
        } else if (dt > 3000) {
            LogLine("SLOW driver call: %llu ms. job: %s", static_cast<unsigned long long>(dt), t_job);
        }
        return ok;
    }

    // ---- timed driver call: runs on a private worker thread so a never-returning compile can be abandoned ----
    // The functor MUST capture only heap/shared state by value (the caller may return while the worker still runs).
    struct TimedTask {
        std::function<void()> fn;
        char job[256];
        HANDLE done;
        volatile LONG refs;     // caller + worker
        volatile LONG state;    // 0 running, 1 finished in time, 2 abandoned by the caller
        volatile LONG ok;
        int slot;
        int catchCrash;
    };
    void TaskRelease(TimedTask* t) {
        if (InterlockedDecrement(&t->refs) == 0) { CloseHandle(t->done); delete t; }
    }
    DWORD WINAPI TimedProc(LPVOID p) {
        TimedTask* t = static_cast<TimedTask*>(p);
        SetJob("%s", t->job);
        bool ok = true;
        if (t->catchCrash) ok = SehRun(t->fn); else t->fn();
        t->ok = ok ? 1 : 0;
        InFlightEnd(t->slot);
        if (InterlockedCompareExchange(&t->state, 1, 0) != 0) {
            InterlockedDecrement(&g_abandoned);
            LogLine("WATCHDOG: an abandoned driver call finally returned (%s). job: %s", ok ? "ok" : "crashed", t->job);
        }
        SetEvent(t->done);
        TaskRelease(t);
        return 0;
    }
    // [v3.2] the log showed 4 abandoned LLPC compiles still burning 4 cores 40 minutes later.
    void PinStuckThread(HANDLE th) {
        if (!g_spvCfg.pinStuckToOneCore || !th) return;
        DWORD_PTR procMask = 0, sysMask = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask) || !procMask) return;
        DWORD_PTR last = 0;
        for (unsigned i = 0; i < sizeof(DWORD_PTR) * 8; ++i) {
            const DWORD_PTR bit = static_cast<DWORD_PTR>(1) << i;
            if (procMask & bit) last = bit;
        }
        if (last && last != procMask) SetThreadAffinityMask(th, last);
    }
    // Returns true when the driver call completed without a crash. timedOut is set when it was abandoned.
    template<class F> bool DriverCallTimed(F&& f, bool& timedOut) {
        timedOut = false;
        int timeoutSec = g_spvCfg.compileTimeoutSec;
        if (timeoutSec > 0 && t_timeoutOverrideSec > 0 && t_timeoutOverrideSec < timeoutSec) timeoutSec = t_timeoutOverrideSec;   // [v3.2]
        const DWORD timeoutMs = timeoutSec > 0 ? static_cast<DWORD>(timeoutSec) * 1000u : 0u;
        if (timeoutMs == 0) return DriverCall(std::forward<F>(f));
        HANDLE gate = CompileGate();
        GateEnter(gate);
        const ULONGLONG t0 = GetTickCount64();
        TimedTask* t = new TimedTask();
        t->fn = std::forward<F>(f);
        strncpy_s(t->job, sizeof(t->job), t_job, _TRUNCATE);
        t->done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        t->refs = 2; t->state = 0; t->ok = 0;
        t->catchCrash = g_spvCfg.catchDriverCrash;
        t->slot = InFlightBegin(t_job);
        HANDLE th = CreateThread(nullptr, static_cast<SIZE_T>(64) << 20, TimedProc, t, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        bool ok;
        if (!th) {
            // could not create a thread: run inline (old behaviour)
            ok = true;
            if (t->catchCrash) ok = SehRun(t->fn); else t->fn();
            InFlightEnd(t->slot);
            CloseHandle(t->done); delete t;
            GateLeave(gate);
            if (!ok) LogLine("CAUGHT driver crash (inline). job: %s", t_job);
            return ok;
        }
        const DWORD w = WaitForSingleObject(t->done, timeoutMs);
        if (w == WAIT_OBJECT_0) {
            ok = t->ok != 0;
        } else {
            InterlockedIncrement(&g_abandoned);
            if (InterlockedCompareExchange(&t->state, 2, 0) == 0) {
                timedOut = true; ok = false;
                InterlockedIncrement(&g_timedOutTotal);
                if (t->slot >= 0) g_inflight[t->slot].abandoned = 1;
                SetThreadPriority(th, THREAD_PRIORITY_IDLE);   // the stuck compile must not starve the game
                PinStuckThread(th);                            // [v3.2] ...and all stuck compiles share ONE core
                LogLine("WATCHDOG: driver call TIMED OUT after %llu ms -> abandoned, pipeline becomes FAKE (stuck thread keeps running at idle priority, abandoned=%ld). job: %s",
                    static_cast<unsigned long long>(GetTickCount64() - t0), static_cast<long>(g_abandoned), t_job);
            } else {   // finished at the very last moment
                InterlockedDecrement(&g_abandoned);
                WaitForSingleObject(t->done, INFINITE);
                ok = t->ok != 0;
            }
        }
        CloseHandle(th);
        const ULONGLONG dt = GetTickCount64() - t0;
        GateLeave(gate);
        TaskRelease(t);
        if (!ok && !timedOut) {
            const LONG c = InterlockedIncrement(&g_driverCrashCount);
            LogLine("CAUGHT driver crash after %llu ms (total recovered=%ld). job: %s", static_cast<unsigned long long>(dt), (long)c, t_job);
        } else if (ok && dt > 3000) {
            LogLine("SLOW driver call: %llu ms. job: %s", static_cast<unsigned long long>(dt), t_job);
        }
        return ok;
    }

    // ---- hang quarantine: pipelines / big shaders that hung once are never sent to the driver again ----------
    SRWLOCK g_hangLock = SRWLOCK_INIT;
    std::set<uint64_t>* g_hangSet = nullptr;
    std::set<uint64_t>* g_pinned = nullptr;     // shader modules still used by an abandoned compile: never destroy them
    bool g_hangLoaded = false;
    void HangFilePath(wchar_t* out, size_t cap) {
        GetLogPath(out, cap);
        wchar_t* sl = wcsrchr(out, L'\\');
        constexpr wchar_t nm[] = L"VulkanProxy_hang.list";
        if (sl && static_cast<size_t>(sl + 1 - out) + (sizeof(nm) / sizeof(nm[0])) <= cap) wcscpy_s(sl + 1, cap - static_cast<size_t>(sl + 1 - out), nm);
        else wcscpy_s(out, cap, nm);
    }
    void HangLoadLocked() {   // g_hangLock held exclusively
        if (g_hangLoaded) return;
        g_hangLoaded = true;
        if (!g_hangSet) g_hangSet = new std::set<uint64_t>();
        wchar_t path[2048]{};
        HangFilePath(path, 2048);
        FILE* f = nullptr;
        if (_wfopen_s(&f, path, L"r") == 0 && f) {
            char line[128];
            while (fgets(line, sizeof(line), f)) {
                const unsigned long long v = _strtoui64(line, nullptr, 16);
                if (v) g_hangSet->insert(static_cast<uint64_t>(v));
            }
            fclose(f);
            LogLine("HANGLIST: loaded %zu quarantined hashes from %ls", g_hangSet->size(), path);
        }
    }
    uint64_t ModuleHash(VkShaderModule m, size_t* words) {
        SpvInfo si;
        if (words) *words = 0;
        if (!GetSpvInfo(m, si) || !si.code || si.code->empty()) return 0;
        if (words) *words = si.code->size();
        uint64_t h = 1469598103934665603ull;
        const uint32_t* p = si.code->data();
        for (size_t i = 0, n = si.code->size(); i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h ? h : 1;
    }
    uint64_t StagesKey(const VkPipelineShaderStageCreateInfo* st, uint32_t n, size_t* maxWords, std::vector<uint64_t>* each) {
        uint64_t k = 1469598103934665603ull;
        size_t mx = 0;
        bool unhashable = false;
        for (uint32_t i = 0; i < n; ++i) {
            size_t w = 0;
            const uint64_t h = ModuleHash(st[i].module, &w);
            if (h == 0) unhashable = true;
            if (each) each->push_back(w >= 30000 ? h : 0);
            mx = (std::max)(mx, w);
            k ^= h; k *= 1099511628211ull;
        }
        if (maxWords) *maxWords = mx;
        if (unhashable) return 0;   // module code not kept (KeepShaderCode=0) or fake module
        return k ? k : 1;
    }
    // true -> do not send this pipeline to the driver (it hung before, or it is huge and we already have stuck compiles)
    bool PipelineQuarantined(const VkPipelineShaderStageCreateInfo* st, uint32_t n, const char** why) {
        size_t mx = 0;
        std::vector<uint64_t> each;
        const uint64_t key = StagesKey(st, n, &mx, &each);
        if (why) *why = nullptr;
        if (g_spvCfg.heavyShaderWords > 0 && mx > static_cast<size_t>(g_spvCfg.heavyShaderWords)) { if (why) *why = "HeavyShaderWords limit"; return true; }
        if (g_spvCfg.maxAbandoned > 0 && g_abandoned >= g_spvCfg.maxAbandoned && mx > 40000) { if (why) *why = "too many stuck compiles already (MaxAbandoned)"; return true; }
        if (key == 0) return false;
        bool hit = false;
        AcquireSRWLockExclusive(&g_hangLock);
        HangLoadLocked();
        if (g_hangSet->count(key)) hit = true;
        for (uint64_t h : each) if (h && g_hangSet->count(h)) { hit = true; break; }
        ReleaseSRWLockExclusive(&g_hangLock);
        if (hit && why) *why = "hung in an earlier launch/pipeline (VulkanProxy_hang.list)";
        return hit;
    }
    void PipelineMarkHung(const VkPipelineShaderStageCreateInfo* st, uint32_t n) {
        size_t mx = 0;
        std::vector<uint64_t> each;
        const uint64_t key = StagesKey(st, n, &mx, &each);
        if (key == 0) return;
        std::vector<uint64_t> add; add.push_back(key);
        for (uint64_t h : each) if (h) add.push_back(h);
        AcquireSRWLockExclusive(&g_hangLock);
        HangLoadLocked();
        if (!g_pinned) g_pinned = new std::set<uint64_t>();
        for (uint32_t i = 0; i < n; ++i) g_pinned->insert(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st[i].module)));
        wchar_t path[2048]{};
        HangFilePath(path, 2048);
        FILE* f = nullptr;
        if (_wfopen_s(&f, path, L"a") == 0 && f) {
            for (uint64_t h : add) if (g_hangSet->insert(h).second) fprintf(f, "%016llx\n", static_cast<unsigned long long>(h));
            fclose(f);
        } else for (uint64_t h : add) g_hangSet->insert(h);
        ReleaseSRWLockExclusive(&g_hangLock);
        LogLine("HANGLIST: pipeline key %016llx (+%zu big stage hashes, largest stage %zu words) quarantined", static_cast<unsigned long long>(key), add.size() - 1, mx);
    }
    bool ModulePinned(VkShaderModule m) {
        bool p = false;
        AcquireSRWLockShared(&g_hangLock);
        if (g_pinned) p = g_pinned->count(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m))) != 0;
        ReleaseSRWLockShared(&g_hangLock);
        return p;
    }

    // ---- persistent pipeline cache ---------------------------------------------------------
    // If the game passes VK_NULL_HANDLE as pipelineCache (the log shows the same heavy modules being compiled over and
    // over, ~6 s each), the proxy supplies its own VkPipelineCache and stores it in VulkanProxy_pipeline.cache next to the
    // dll. The second launch then skips most of the compilation (the long black screen after the intro videos).
    namespace pcache {
        struct Ent { VkDevice dev; VkPipelineCache cache; };
        SRWLOCK g_lk = SRWLOCK_INIT;
        std::vector<Ent> g_ents;
        volatile LONG g_dirty = 0;
        volatile LONG g_saving = 0;
        volatile LONG64 g_lastSave = 0;

        void FilePath(wchar_t* out, size_t cap) {
            GetLogPath(out, cap);
            wchar_t* sl = wcsrchr(out, L'\\');
            constexpr wchar_t nm[] = L"VulkanProxy_pipeline.cache";
            if (sl && static_cast<size_t>(sl + 1 - out) + (sizeof(nm) / sizeof(nm[0])) <= cap)
                wcscpy_s(sl + 1, cap - static_cast<size_t>(sl + 1 - out), nm);
            else
                wcscpy_s(out, cap, nm);
        }

        bool ReadAll(const wchar_t* path, std::vector<uint8_t>& out) {
            HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (f == INVALID_HANDLE_VALUE) return false;
            LARGE_INTEGER sz{};
            bool ok = false;
            if (GetFileSizeEx(f, &sz) && sz.QuadPart > 64 && sz.QuadPart < (1ll << 30)) {
                out.resize(static_cast<size_t>(sz.QuadPart));
                DWORD got = 0;
                ok = ReadFile(f, out.data(), static_cast<DWORD>(sz.QuadPart), &got, nullptr) && got == sz.QuadPart;
            }
            CloseHandle(f);
            if (!ok) out.clear();
            return ok;
        }

        VkPipelineCache Get(VkDevice device, VkPipelineCache app) {
            if (app != VK_NULL_HANDLE || !device) return app;
            LoadSpvConfig();
            if (!g_spvCfg.pipelineCache) return app;
            AcquireSRWLockShared(&g_lk);
            for (const Ent& e : g_ents) if (e.dev == device) { const VkPipelineCache c = e.cache; ReleaseSRWLockShared(&g_lk); return c; }
            ReleaseSRWLockShared(&g_lk);

            AcquireSRWLockExclusive(&g_lk);
            for (const Ent& e : g_ents) if (e.dev == device) { const VkPipelineCache c = e.cache; ReleaseSRWLockExclusive(&g_lk); return c; }
            VkPipelineCache created = VK_NULL_HANDLE;
            auto create = RealExport<PFN_vkCreatePipelineCache>("vkCreatePipelineCache");
            if (create) {
                std::vector<uint8_t> data;
                wchar_t path[2048]{};
                FilePath(path, 2048);
                const bool have = ReadAll(path, data);
                for (int attempt = have ? 0 : 1; attempt < 2 && !created; ++attempt) {
                    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
                    if (attempt == 0) { ci.initialDataSize = data.size(); ci.pInitialData = data.data(); }
                    VkPipelineCache c = VK_NULL_HANDLE;
                    VkResult r = VK_ERROR_UNKNOWN;
                    SetJob("vkCreatePipelineCache (initial data %zu bytes)", ci.initialDataSize);
                    if (DriverCall([&]() { r = create(device, &ci, nullptr, &c); }) && r == VK_SUCCESS && c) created = c;
                    else LogLine("PIPECACHE: vkCreatePipelineCache failed (attempt %d, result=%d)", attempt, (int)r);
                }
                LogLine("PIPECACHE: proxy pipeline cache %s (loaded %zu bytes from disk)", created ? "created" : "UNAVAILABLE", created ? data.size() : (size_t)0);
            }
            g_ents.push_back({ device, created });
            ReleaseSRWLockExclusive(&g_lk);
            return created;
        }

        void Dirty() { InterlockedExchange(&g_dirty, 1); }

        void MaybeSave(bool force) {
            if (!g_dirty) return;
            const LONG64 now = static_cast<LONG64>(GetTickCount64());
            if (!force && now - g_lastSave < 30000) return;
            if (InterlockedCompareExchange(&g_saving, 1, 0) != 0) return;
            g_lastSave = now;
            InterlockedExchange(&g_dirty, 0);
            Ent e{};
            AcquireSRWLockShared(&g_lk);
            for (const Ent& x : g_ents) if (x.cache) { e = x; break; }
            ReleaseSRWLockShared(&g_lk);
            auto getData = e.cache ? RealExport<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData") : nullptr;
            if (getData) {
                size_t sz = 0;
                VkResult r = VK_ERROR_UNKNOWN;
                SetJob("vkGetPipelineCacheData(size)");
                if (DriverCall([&]() { r = getData(e.dev, e.cache, &sz, nullptr); }, false) && r == VK_SUCCESS && sz > 64 && sz < (1ull << 31)) {
                    std::vector<uint8_t> buf(sz);
                    size_t sz2 = sz;
                    r = VK_ERROR_UNKNOWN;
                    SetJob("vkGetPipelineCacheData(%zu bytes)", sz);
                    if (DriverCall([&]() { r = getData(e.dev, e.cache, &sz2, buf.data()); }, false) && r == VK_SUCCESS && sz2 > 64) {
                        wchar_t path[2048]{}, tmp[2100]{};
                        FilePath(path, 2048);
                        _snwprintf_s(tmp, 2100, _TRUNCATE, L"%ls.tmp", path);
                        HANDLE f = CreateFileW(tmp, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                        if (f != INVALID_HANDLE_VALUE) {
                            DWORD w = 0;
                            const BOOL ok = WriteFile(f, buf.data(), static_cast<DWORD>(sz2), &w, nullptr) && w == sz2;
                            CloseHandle(f);
                            if (ok && MoveFileExW(tmp, path, MOVEFILE_REPLACE_EXISTING))
                                LogLine("PIPECACHE: saved %zu KB", sz2 / 1024);
                            else DeleteFileW(tmp);
                        }
                    }
                }
            }
            InterlockedExchange(&g_saving, 0);
        }
    }

    void DumpSpirv(const char* tag, long idx, const uint32_t* code, size_t bytes) {
        wchar_t path[2048]{};
        GetLogPath(path, sizeof(path) / sizeof(path[0]));
        wchar_t* slash = wcsrchr(path, L'\\');
        if (!slash) return;
        wchar_t name[96];
        _snwprintf_s(name, 96, _TRUNCATE, L"spv_%S_%ld.spv", tag, idx);
        wcscpy_s(slash + 1, (sizeof(path) / sizeof(path[0])) - static_cast<size_t>(slash + 1 - path), name);
        HANDLE f = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            DWORD w = 0;
            WriteFile(f, code, static_cast<DWORD>(bytes), &w, nullptr);
            CloseHandle(f);
            LogLine("SPIRV: dumped %zu bytes to %ls", bytes, path);
        }
    }

    // Post-mortem of a pipeline the driver rejected: log the interesting states and dump its shaders.
    void DescribeFailedGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci, long seq) {
        LogLine("FAILED GRAPHICS PIPELINE #%ld: flags=0x%X stages=%u renderPass=%p subpass=%u", seq, (unsigned)ci.flags,
            ci.stageCount, (void*)ci.renderPass, ci.subpass);
        for (uint32_t j = 0; j < ci.stageCount && ci.pStages; ++j) {
            const auto& st = ci.pStages[j];
            SpvInfo si;
            const bool have = GetSpvInfo(st.module, si);
            LogLine("  stage 0x%X entry=%s spec=%u module: spv=%x model=%u caps=[%s]%s", (unsigned)st.stage,
                st.pName ? st.pName : "?", st.pSpecializationInfo ? st.pSpecializationInfo->mapEntryCount : 0u,
                have ? si.version : 0u, have ? si.model : 0u, have ? si.caps.c_str() : "unknown",
                (have && si.rewritten) ? " (REWRITTEN ray-query shader)" : "");
            if (have && si.code && !si.code->empty()) {
                char tag[48];
                _snprintf_s(tag, sizeof(tag), _TRUNCATE, "gpfail%ld_stage%X", seq, (unsigned)st.stage);
                DumpSpirv(tag, 0, si.code->data(), si.code->size() * 4);
            }
        }
        if (ci.pVertexInputState) {
            const auto* vi = ci.pVertexInputState;
            LogLine("  vertex input: bindings=%u attributes=%u", vi->vertexBindingDescriptionCount, vi->vertexAttributeDescriptionCount);
            for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount && a < 24; ++a)
                LogLine("    attr loc=%u binding=%u format=%d offset=%u", vi->pVertexAttributeDescriptions[a].location,
                    vi->pVertexAttributeDescriptions[a].binding, (int)vi->pVertexAttributeDescriptions[a].format,
                    vi->pVertexAttributeDescriptions[a].offset);
        }
        if (ci.pInputAssemblyState) LogLine("  topology=%d primitiveRestart=%u", (int)ci.pInputAssemblyState->topology, ci.pInputAssemblyState->primitiveRestartEnable);
        if (ci.pRasterizationState) LogLine("  raster: polygonMode=%d cull=%u depthClamp=%u discard=%u", (int)ci.pRasterizationState->polygonMode,
            (unsigned)ci.pRasterizationState->cullMode, ci.pRasterizationState->depthClampEnable, ci.pRasterizationState->rasterizerDiscardEnable);
        if (ci.pMultisampleState) LogLine("  samples=%d sampleShading=%u alphaToCoverage=%u", (int)ci.pMultisampleState->rasterizationSamples,
            ci.pMultisampleState->sampleShadingEnable, ci.pMultisampleState->alphaToCoverageEnable);
        if (ci.pColorBlendState) LogLine("  color attachments=%u logicOp=%u", ci.pColorBlendState->attachmentCount, ci.pColorBlendState->logicOpEnable);
        if (ci.pDynamicState) {
            char dyn[300]{}; size_t l = 0;
            for (uint32_t d = 0; d < ci.pDynamicState->dynamicStateCount && l + 12 < sizeof(dyn); ++d)
                l += _snprintf_s(dyn + l, sizeof(dyn) - l, _TRUNCATE, "%d ", (int)ci.pDynamicState->pDynamicStates[d]);
            LogLine("  dynamic states: %s", dyn);
        }
        for (const VkBaseInStructure* n = reinterpret_cast<const VkBaseInStructure*>(ci.pNext); n; n = n->pNext)
            LogLine("  pNext sType=%u", (unsigned)n->sType);
    }

    // ---- crash handler ---------------------------------------------------------
    // Function name for an address inside THIS proxy dll, from vulkan-1.pdb (build with /Zi + /DEBUG,
    // keep the .pdb next to the dll). Uses a private dbghelp session so the game's own is not disturbed.
    bool ProxySymbolName(const void* addr, char* out, size_t cap) {
#ifdef _MSC_VER
        static HANDLE hs = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(0x56504B31));
        static bool init = false, ok = false;
        if (!init) {
            init = true;
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            if (SymInitialize(hs, nullptr, FALSE) && g_thisModule) {
                wchar_t path[MAX_PATH]{};
                GetModuleFileNameW(g_thisModule, path, MAX_PATH);
                auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(g_thisModule);
                auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(reinterpret_cast<const BYTE*>(g_thisModule) + dos->e_lfanew);
                ok = SymLoadModuleExW(hs, nullptr, path, nullptr, reinterpret_cast<DWORD64>(g_thisModule),
                                      nt->OptionalHeader.SizeOfImage, nullptr, 0) != 0;
            }
        }
        if (!ok) return false;
        alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256]{};
        auto* si = reinterpret_cast<SYMBOL_INFO*>(buf);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = 255;
        DWORD64 disp = 0;
        if (SymFromAddr(hs, reinterpret_cast<DWORD64>(addr), &disp, si)) {
            _snprintf_s(out, cap, _TRUNCATE, "%s+0x%llX", si->Name, static_cast<unsigned long long>(disp));
            return true;
        }
#else
        (void)addr; (void)out; (void)cap;
#endif
        return false;
    }

    void ModuleOffset(const void* addr, char* out, size_t cap) {
        HMODULE mod = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(addr), &mod) && mod) {
            char name[MAX_PATH]{};
            GetModuleFileNameA(mod, name, MAX_PATH);
            const char* base = strrchr(name, '\\');
            _snprintf_s(out, cap, _TRUNCATE, "%s+0x%llX", base ? base + 1 : name,
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(mod)));
            char sym[200];
            if (mod == g_thisModule && ProxySymbolName(addr, sym, sizeof(sym))) {
                const size_t l = strlen(out);
                if (l + 4 < cap) _snprintf_s(out + l, cap - l, _TRUNCATE, "  <%s>", sym);
            }
        } else _snprintf_s(out, cap, _TRUNCATE, "%p", addr);
    }

    // [v3.2] Thread dump for the stall detector. While a thread is suspended we ONLY copy its registers and a slice of its
    // stack into preallocated memory (no heap, no loader lock, no log lock -> the suspended thread may own any of them).
    // Symbolisation happens after every thread was resumed.
    struct ThreadSnap { DWORD tid; DWORD64 rip, rsp; int words; DWORD64 stack[384]; };
    void DumpAllThreads(const char* why) {
        LogLine("=== STALL DUMP === %s", why ? why : "");
        const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
        std::vector<ThreadSnap> snaps;
        snaps.reserve(160);
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snap == INVALID_HANDLE_VALUE) { LogLine("STALL DUMP: CreateToolhelp32Snapshot failed (%lu)", GetLastError()); return; }
        std::vector<DWORD> tids;
        THREADENTRY32 te{};
        te.dwSize = sizeof(te);
        if (Thread32First(snap, &te)) {
            do { if (te.th32OwnerProcessID == pid && te.th32ThreadID != self) tids.push_back(te.th32ThreadID); } while (Thread32Next(snap, &te));
        }
        CloseHandle(snap);
        for (DWORD tid : tids) {
            if (snaps.size() >= 150) break;
            HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid);
            if (!th) continue;
            if (SuspendThread(th) != static_cast<DWORD>(-1)) {
                CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
                if (GetThreadContext(th, &ctx)) {
                    ThreadSnap ts;   // pushed below; plain copy, no allocation (capacity reserved)
                    ts.tid = tid; ts.rip = ctx.Rip; ts.rsp = ctx.Rsp; ts.words = 0;
                    SIZE_T got = 0;
                    if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(ctx.Rsp), ts.stack, sizeof(ts.stack), &got) && got >= 8)
                        ts.words = static_cast<int>(got / 8);
                    snaps.push_back(ts);
                }
                ResumeThread(th);
            }
            CloseHandle(th);
        }
        for (const ThreadSnap& ts : snaps) {
            char where[300];
            ModuleOffset(reinterpret_cast<const void*>(static_cast<uintptr_t>(ts.rip)), where, sizeof(where));
            std::string line = "  thread " + std::to_string(ts.tid) + " rip=" + where + " | return addresses on stack:";
            int hits = 0;
            for (int i = 0; i < ts.words && hits < 12; ++i) {
                const DWORD64 w = ts.stack[i];
                if (w < 0x10000ull || w > 0x00007FFFFFFFFFFFull) continue;
                MEMORY_BASIC_INFORMATION mbi{};
                if (!VirtualQuery(reinterpret_cast<LPCVOID>(static_cast<uintptr_t>(w)), &mbi, sizeof(mbi))) continue;
                if (mbi.State != MEM_COMMIT || mbi.Type != MEM_IMAGE) continue;
                if (!(mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY))) continue;
                char mo[300];
                ModuleOffset(reinterpret_cast<const void*>(static_cast<uintptr_t>(w)), mo, sizeof(mo));
                line += " ["; line += mo; line += "]";
                ++hits;
            }
            LogLine("%s", line.c_str());
        }
        LogLine("=== END STALL DUMP (%zu threads) ===", snaps.size());
        const LONG end = g_crumbIdx;
        LogLine("STALL DUMP: last API breadcrumbs (oldest first):");
        for (int k = kCrumbCount - 16; k < kCrumbCount; ++k) {
            const char* c = g_crumbs[(end + k) & (kCrumbCount - 1)];
            if (c[0]) LogLine("    %s", c);
        }
    }

    void ReportCrash(PEXCEPTION_POINTERS ep);
    LONG CALLBACK CrashHandler(PEXCEPTION_POINTERS ep) {
        if (!ep || !ep->ExceptionRecord || !ep->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
        const DWORD c0 = ep->ExceptionRecord->ExceptionCode;
        const bool fatal0 = c0 == EXCEPTION_ACCESS_VIOLATION || 
                            c0 == EXCEPTION_INT_DIVIDE_BY_ZERO || c0 == EXCEPTION_PRIV_INSTRUCTION ||
                            c0 == EXCEPTION_IN_PAGE_ERROR;
        const bool guarded = fatal0 && t_guard.active;
        ReportCrash(ep);
        if (guarded) {
            // Fault inside a protected driver call: resume at the saved context with the "tripped" flag set.
            t_guard.tripped = 1;
            t_guard.active = 0;
            *ep->ContextRecord = t_guard.ctx;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Extra safety: EXEC@0 (null function pointer).
        // Case A: inside a DriverCall-guarded section -> resume at saved context.
        if (fatal0 && c0 == EXCEPTION_ACCESS_VIOLATION &&
            ep->ExceptionRecord->NumberParameters >= 2 &&
            ep->ExceptionRecord->ExceptionInformation[0] == 8 /* EXEC */ &&
            ep->ExceptionRecord->ExceptionInformation[1] == 0 &&
            t_guard.ctx.Rip != 0 && t_guard.active) {
            LogLine("!!! EXEC@0 recovered via saved DriverCall context");
            t_guard.tripped = 1;
            t_guard.active = 0;
            *ep->ContextRecord = t_guard.ctx;
            InterlockedIncrement(&g_driverCrashCount);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Case B: null call from the game itself (stored a nullptr from GIPA/GDPA).
        // On x64 the return address is still on the stack at [RSP]. Skip the failed
        // call by returning 0 (VK_SUCCESS / null handle) to the caller.
        if (fatal0 && c0 == EXCEPTION_ACCESS_VIOLATION &&
            ep->ExceptionRecord->NumberParameters >= 2 &&
            ep->ExceptionRecord->ExceptionInformation[0] == 8 /* EXEC */ &&
            ep->ExceptionRecord->ExceptionInformation[1] == 0 &&
            ep->ContextRecord->Rsp) {
            __try {
                const ULONG64 ret = *reinterpret_cast<ULONG64*>(ep->ContextRecord->Rsp);
                if (ret > 0x10000ull) {
                    char caller[300]; ModuleOffset((void*)ret, caller, sizeof(caller));
                LogLine("!!! EXEC@0 skipped null call, caller=%s", caller);
                    ep->ContextRecord->Rip = ret;
                    ep->ContextRecord->Rsp += 8;  // pop return address
                    ep->ContextRecord->Rax = 0;   // return VK_SUCCESS / nullptr
                    InterlockedIncrement(&g_driverCrashCount);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                // stack unreadable - fall through
            }
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    void ReportCrash(PEXCEPTION_POINTERS ep) {
        if (!ep || !ep->ExceptionRecord) return;
        const DWORD code = ep->ExceptionRecord->ExceptionCode;
        const bool fatal = code == EXCEPTION_ACCESS_VIOLATION || 
                           code == EXCEPTION_STACK_OVERFLOW || code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
                           code == EXCEPTION_PRIV_INSTRUCTION || code == EXCEPTION_IN_PAGE_ERROR;
        if (!fatal) return;
        static volatile LONG reported = 0;
        char where[300];
        ModuleOffset(ep->ExceptionRecord->ExceptionAddress, where, sizeof(where));
        {
            // Report every DISTINCT crash location once (the final fatal crash must not be hidden
            // by earlier, already handled driver crashes).
            char sig[400];
            _snprintf_s(sig, sizeof(sig), _TRUNCATE, "crash:%08lX:%s", code, where);
            if (!FirstTime(sig)) return;
            if (InterlockedIncrement(&reported) > 60) return;
        }
        LogLine("!!! EXCEPTION 0x%08lX at %s (thread %lu)%s", code, where, GetCurrentThreadId(),
            t_guard.active ? " [inside guarded driver call: will recover]" : "");
        if (t_job[0]) LogLine("!!! this thread's current job: %s", t_job);
        if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
            LogLine("!!! access violation: %s address 0x%llX",
                ep->ExceptionRecord->ExceptionInformation[0] == 0 ? "READ" : (ep->ExceptionRecord->ExceptionInformation[0] == 1 ? "WRITE" : "EXEC"),
                static_cast<unsigned long long>(ep->ExceptionRecord->ExceptionInformation[1]));
        if (ep->ContextRecord) {
            const CONTEXT* c = ep->ContextRecord;
            LogLine("!!! RAX=%016llX RBX=%016llX RCX=%016llX RDX=%016llX", c->Rax, c->Rbx, c->Rcx, c->Rdx);
            LogLine("!!! RSI=%016llX RDI=%016llX R8 =%016llX R9 =%016llX", c->Rsi, c->Rdi, c->R8, c->R9);
            LogLine("!!! R14=%016llX R15=%016llX RSP=%016llX RIP=%016llX", c->R14, c->R15, c->Rsp, c->Rip);
        }
        void* frames[32]{};
        const USHORT n = CaptureStackBackTrace(0, 32, frames, nullptr);
        for (USHORT k = 0; k < n; ++k) {
            ModuleOffset(frames[k], where, sizeof(where));
            LogLine("!!!   frame %u: %s", static_cast<unsigned>(k), where);
        }
        // Name the amdvlk64 functions: compare frame addresses with the driver's real entry points.
        if (g_lastDevice && g_realGdpa) {
            static const char* const kExtra[] = {
                "vkCreateShaderModule", "vkCreateGraphicsPipelines", "vkCreateComputePipelines", "vkCmdBindPipeline",
                "vkCmdDraw", "vkCmdDrawIndexed", "vkCmdDispatch", "vkCmdDrawIndirect", "vkCmdDrawIndexedIndirect",
                "vkCmdBindDescriptorSets", "vkCmdPushConstants", "vkCmdBindVertexBuffers", "vkCmdBindIndexBuffer",
                "vkCreatePipelineLayout", "vkCreateDescriptorSetLayout", "vkAllocateDescriptorSets",
                "vkUpdateDescriptorSets", "vkCreateSampler", "vkCreateImage", "vkCreateImageView", "vkCreateBuffer",
                "vkAllocateMemory", "vkBindImageMemory", "vkBindBufferMemory", "vkGetImageMemoryRequirements",
                "vkGetBufferMemoryRequirements", "vkCreateSwapchainKHR", "vkQueuePresentKHR", "vkCreateCommandPool",
                "vkAllocateCommandBuffers", "vkCreateFence", "vkCreateRenderPass", "vkCreateFramebuffer",
                "vkBeginCommandBuffer", "vkEndCommandBuffer", "vkQueueSubmit", "vkDestroyPipeline",
                "vkDestroyShaderModule", "vkCreateDescriptorPool", "vkCreateQueryPool", "vkCreateSemaphore" };
            struct NA { const char* n; uintptr_t a; };
            std::vector<NA> names;
            for (const char* n : kExtra) { auto f = g_realGdpa(g_lastDevice, n); if (f) names.push_back({ n, reinterpret_cast<uintptr_t>(f) }); }
            for (int i = 0; i < VK_PROXY_EXPORT_COUNT; ++i) {
                auto f = g_realGdpa(g_lastDevice, kVkProxyExportNames[i]);
                if (f) names.push_back({ kVkProxyExportNames[i], reinterpret_cast<uintptr_t>(f) });
            }
            for (USHORT k = 0; k < n; ++k) {
                const uintptr_t a = reinterpret_cast<uintptr_t>(frames[k]);
                const NA* best = nullptr;
                for (const NA& e : names) if (e.a <= a && a - e.a < 0x400 && (!best || e.a > best->a)) best = &e;
                if (best) LogLine("!!!   frame %u is inside driver function %s (+0x%llX)", static_cast<unsigned>(k), best->n,
                    static_cast<unsigned long long>(a - best->a));
            }
        }
        LogLine("!!! last API breadcrumbs (oldest first):");
        const LONG end = g_crumbIdx;
        for (int k = t_guard.active ? kCrumbCount - 12 : 0; k < kCrumbCount; ++k) {
            const char* s = g_crumbs[(end + k) & (kCrumbCount - 1)];
            if (s[0]) LogLine("!!!   %s", s);
        }
    }

    PVOID g_vehHandle = nullptr;
    // The game may install its own vectored handler after us; make sure ours runs FIRST.
    void InstallCrashHandlerFirst() {
        if (g_vehHandle) RemoveVectoredExceptionHandler(g_vehHandle);
        g_vehHandle = AddVectoredExceptionHandler(1, CrashHandler);
    }
}

// ---------------------------------------------------------------------------
// v4: diagnostic wrappers for swapchain / shader / present
// ---------------------------------------------------------------------------
// Optional post-process: brightness gain applied to the swapchain image right before it is presented.
// out = dst * (1 + g) is done with fixed-function blending (src = vec4(g), srcFactor = DST_COLOR, dstFactor = ONE),
// repeated N times for gains above 2. Enabled with VulkanProxy.ini  BrightnessGain=1.5  (1.0 = off).
// ---------------------------------------------------------------------------
namespace {
namespace post {
static const uint32_t kGainVert[] = {0x07230203,0x00010000,0x0008000b,0x0000002b,0x00000000,0x00020011,0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x0000001d,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,0x00000070,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00060005,0x0000001b,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x0000001b,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001b,0x00000001,0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x0000001b,0x00000002,0x435f6c67,0x4470696c,0x61747369,0x0065636e,0x00070006,0x0000001b,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,0x00030005,0x0000001d,0x00000000,0x00040047,0x0000000c,0x0000000b,0x0000002a,0x00030047,0x0000001b,0x00000002,0x00050048,0x0000001b,0x00000000,0x0000000b,0x00000000,0x00050048,0x0000001b,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001b,0x00000002,0x0000000b,0x00000003,0x00050048,0x0000001b,0x00000003,0x0000000b,0x00000004,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000002,0x00040020,0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,0x0000000c,0x00000001,0x0004002b,0x0000000a,0x0000000e,0x00000001,0x0004002b,0x0000000a,0x00000010,0x00000002,0x00040017,0x00000017,0x00000006,0x00000004,0x00040015,0x00000018,0x00000020,0x00000000,0x0004002b,0x00000018,0x00000019,0x00000001,0x0004001c,0x0000001a,0x00000006,0x00000019,0x0006001e,0x0000001b,0x00000017,0x00000006,0x0000001a,0x0000001a,0x00040020,0x0000001c,0x00000003,0x0000001b,0x0004003b,0x0000001c,0x0000001d,0x00000003,0x0004002b,0x0000000a,0x0000001e,0x00000000,0x0004002b,0x00000006,0x00000020,0x40000000,0x0004002b,0x00000006,0x00000022,0x3f800000,0x0004002b,0x00000006,0x00000025,0x00000000,0x00040020,0x00000029,0x00000003,0x00000017,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,0x0000000a,0x0000000d,0x0000000c,0x000500c4,0x0000000a,0x0000000f,0x0000000d,0x0000000e,0x000500c7,0x0000000a,0x00000011,0x0000000f,0x00000010,0x0004006f,0x00000006,0x00000012,0x00000011,0x0004003d,0x0000000a,0x00000013,0x0000000c,0x000500c7,0x0000000a,0x00000014,0x00000013,0x00000010,0x0004006f,0x00000006,0x00000015,0x00000014,0x00050050,0x00000007,0x00000016,0x00000012,0x00000015,0x0003003e,0x00000009,0x00000016,0x0004003d,0x00000007,0x0000001f,0x00000009,0x0005008e,0x00000007,0x00000021,0x0000001f,0x00000020,0x00050050,0x00000007,0x00000023,0x00000022,0x00000022,0x00050083,0x00000007,0x00000024,0x00000021,0x00000023,0x00050051,0x00000006,0x00000026,0x00000024,0x00000000,0x00050051,0x00000006,0x00000027,0x00000024,0x00000001,0x00070050,0x00000017,0x00000028,0x00000026,0x00000027,0x00000025,0x00000022,0x00050041,0x00000029,0x0000002a,0x0000001d,0x0000001e,0x0003003e,0x0000002a,0x00000028,0x000100fd,0x00010038};
static const uint32_t kGainFrag[] = {0x07230203,0x00010000,0x0008000b,0x00000013,0x00000000,0x00020011,0x00000001,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00050005,0x00000009,0x4374756f,0x726f6c6f,0x00000000,0x00030005,0x0000000a,0x00004350,0x00040006,0x0000000a,0x00000000,0x00000067,0x00030005,0x0000000c,0x00006370,0x00040047,0x00000009,0x0000001e,0x00000000,0x00030047,0x0000000a,0x00000002,0x00050048,0x0000000a,0x00000000,0x00000023,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,0x0003001e,0x0000000a,0x00000006,0x00040020,0x0000000b,0x00000009,0x0000000a,0x0004003b,0x0000000b,0x0000000c,0x00000009,0x00040015,0x0000000d,0x00000020,0x00000001,0x0004002b,0x0000000d,0x0000000e,0x00000000,0x00040020,0x0000000f,0x00000009,0x00000006,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,0x0000000f,0x00000010,0x0000000c,0x0000000e,0x0004003d,0x00000006,0x00000011,0x00000010,0x00070050,0x00000007,0x00000012,0x00000011,0x00000011,0x00000011,0x00000011,0x0003003e,0x00000009,0x00000012,0x000100fd,0x00010038};

    struct SwapInfo { VkFormat fmt = VK_FORMAT_UNDEFINED; VkExtent2D ext{}; uint32_t generation = 0; };
    struct SwapState {
        uint32_t generation = 0;
        bool ready = false, failed = false;
        VkRenderPass rp = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        std::vector<VkImageView> views;
        std::vector<VkFramebuffer> fbs;
        std::vector<VkCommandBuffer> cbs;
        std::vector<VkSemaphore> sems;
    };
    SRWLOCK g_lock = SRWLOCK_INIT;
    std::map<uint64_t, SwapInfo>& Infos() { static auto* m = new std::map<uint64_t, SwapInfo>(); return *m; }
    std::map<uint64_t, SwapState>& States() { static auto* m = new std::map<uint64_t, SwapState>(); return *m; }
    uint32_t g_gen = 0;

    inline uint64_t H(VkSwapchainKHR s) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(s)); }
    void Record(VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext) {
        AcquireSRWLockExclusive(&g_lock);
        Infos()[H(sc)] = { fmt, ext, ++g_gen };
        ReleaseSRWLockExclusive(&g_lock);
    }

    template<class T> T Fn(VkDevice d, const char* name) { return reinterpret_cast<T>(g_realGdpa(d, name)); }

    bool Build(VkDevice dev, VkQueue queue, VkSwapchainKHR sc, const SwapInfo& info, SwapState& st) {
        auto createRP = Fn<PFN_vkCreateRenderPass>(dev, "vkCreateRenderPass");
        auto createFB = Fn<PFN_vkCreateFramebuffer>(dev, "vkCreateFramebuffer");
        auto createView = Fn<PFN_vkCreateImageView>(dev, "vkCreateImageView");
        auto createSM = Fn<PFN_vkCreateShaderModule>(dev, "vkCreateShaderModule");
        auto destroySM = Fn<PFN_vkDestroyShaderModule>(dev, "vkDestroyShaderModule");
        auto createPL = Fn<PFN_vkCreatePipelineLayout>(dev, "vkCreatePipelineLayout");
        auto createGP = Fn<PFN_vkCreateGraphicsPipelines>(dev, "vkCreateGraphicsPipelines");
        auto createPool = Fn<PFN_vkCreateCommandPool>(dev, "vkCreateCommandPool");
        auto allocCB = Fn<PFN_vkAllocateCommandBuffers>(dev, "vkAllocateCommandBuffers");
        auto beginCB = Fn<PFN_vkBeginCommandBuffer>(dev, "vkBeginCommandBuffer");
        auto endCB = Fn<PFN_vkEndCommandBuffer>(dev, "vkEndCommandBuffer");
        auto cmdBeginRP = Fn<PFN_vkCmdBeginRenderPass>(dev, "vkCmdBeginRenderPass");
        auto cmdEndRP = Fn<PFN_vkCmdEndRenderPass>(dev, "vkCmdEndRenderPass");
        auto cmdBind = Fn<PFN_vkCmdBindPipeline>(dev, "vkCmdBindPipeline");
        auto cmdPush = Fn<PFN_vkCmdPushConstants>(dev, "vkCmdPushConstants");
        auto cmdDraw = Fn<PFN_vkCmdDraw>(dev, "vkCmdDraw");
        auto createSem = Fn<PFN_vkCreateSemaphore>(dev, "vkCreateSemaphore");
        auto getImages = Fn<PFN_vkGetSwapchainImagesKHR>(dev, "vkGetSwapchainImagesKHR");
        auto getQueue = Fn<PFN_vkGetDeviceQueue>(dev, "vkGetDeviceQueue");
        if (!createRP || !createFB || !createView || !createSM || !destroySM || !createPL || !createGP || !createPool ||
            !allocCB || !beginCB || !endCB || !cmdBeginRP || !cmdEndRP || !cmdBind || !cmdPush || !cmdDraw ||
            !createSem || !getImages || !getQueue) { LogLine("BRIGHTNESS: missing driver functions"); return false; }

        // queue family of the presenting queue
        uint32_t family = 0xFFFFFFFFu;
        for (const auto& f : g_createdFamilies) {
            for (uint32_t qi = 0; qi < f.second && family == 0xFFFFFFFFu; ++qi) {
                VkQueue q = VK_NULL_HANDLE;
                getQueue(dev, f.first, qi, &q);
                if (q == queue) family = f.first;
            }
        }
        if (family == 0xFFFFFFFFu) { LogLine("BRIGHTNESS: could not identify the queue family of the present queue"); return false; }

        uint32_t count = 0;
        if (getImages(dev, sc, &count, nullptr) != VK_SUCCESS || count == 0 || count > 16) return false;
        std::vector<VkImage> images(count);
        if (getImages(dev, sc, &count, images.data()) != VK_SUCCESS) return false;

        // render pass: one color attachment kept in PRESENT_SRC layout
        VkAttachmentDescription att{};
        att.format = info.fmt; att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sub.colorAttachmentCount = 1; sub.pColorAttachments = &ref;
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; deps[1].dstAccessMask = 0;
        VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        rpci.attachmentCount = 1; rpci.pAttachments = &att; rpci.subpassCount = 1; rpci.pSubpasses = &sub;
        rpci.dependencyCount = 2; rpci.pDependencies = deps;
        if (createRP(dev, &rpci, nullptr, &st.rp) != VK_SUCCESS) return false;

        VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (createPL(dev, &plci, nullptr, &st.layout) != VK_SUCCESS) return false;

        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smv{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smv.codeSize = sizeof(kGainVert); smv.pCode = kGainVert;
        VkShaderModuleCreateInfo smf{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smf.codeSize = sizeof(kGainFrag); smf.pCode = kGainFrag;
        if (createSM(dev, &smv, nullptr, &vs) != VK_SUCCESS || createSM(dev, &smf, nullptr, &fs) != VK_SUCCESS) return false;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport vp{ 0.f, 0.f, static_cast<float>(info.ext.width), static_cast<float>(info.ext.height), 0.f, 1.f };
        VkRect2D sc2{ { 0, 0 }, info.ext };
        VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc2;
        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR; ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; ba.alphaBlendOp = VK_BLEND_OP_ADD;
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1; cb.pAttachments = &ba;
        VkGraphicsPipelineCreateInfo gp{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        gp.stageCount = 2; gp.pStages = stages; gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vps;
        gp.pRasterizationState = &rs; gp.pMultisampleState = &ms; gp.pColorBlendState = &cb;
        gp.layout = st.layout; gp.renderPass = st.rp; gp.subpass = 0;
        const VkResult pr = createGP(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &st.pipe);
        destroySM(dev, vs, nullptr); destroySM(dev, fs, nullptr);
        if (pr != VK_SUCCESS) { LogLine("BRIGHTNESS: pipeline creation failed: %d", (int)pr); return false; }

        VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpi.queueFamilyIndex = family;
        if (createPool(dev, &cpi, nullptr, &st.pool) != VK_SUCCESS) return false;
        st.cbs.resize(count);
        VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = st.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = count;
        if (allocCB(dev, &cai, st.cbs.data()) != VK_SUCCESS) return false;

        // passes: gain = (1 + g)^N with g <= 1 (source color is clamped to [0,1] for UNORM targets)
        const float G = g_spvCfg.brightness;
        int passes = 1;
        while (passes < 4 && std::pow(2.0f, static_cast<float>(passes)) < G) ++passes;
        const float g = std::pow(G, 1.0f / static_cast<float>(passes)) - 1.0f;
        const float gClamped = g > 1.0f ? 1.0f : g;

        st.views.resize(count); st.fbs.resize(count); st.sems.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            vci.image = images[i]; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = info.fmt;
            vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            if (createView(dev, &vci, nullptr, &st.views[i]) != VK_SUCCESS) return false;
            VkFramebufferCreateInfo fci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            fci.renderPass = st.rp; fci.attachmentCount = 1; fci.pAttachments = &st.views[i];
            fci.width = info.ext.width; fci.height = info.ext.height; fci.layers = 1;
            if (createFB(dev, &fci, nullptr, &st.fbs[i]) != VK_SUCCESS) return false;
            VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            if (createSem(dev, &sci, nullptr, &st.sems[i]) != VK_SUCCESS) return false;

            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            if (beginCB(st.cbs[i], &bi) != VK_SUCCESS) return false;
            VkRenderPassBeginInfo rbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            rbi.renderPass = st.rp; rbi.framebuffer = st.fbs[i]; rbi.renderArea = { { 0, 0 }, info.ext };
            cmdBeginRP(st.cbs[i], &rbi, VK_SUBPASS_CONTENTS_INLINE);
            cmdBind(st.cbs[i], VK_PIPELINE_BIND_POINT_GRAPHICS, st.pipe);
            for (int pidx = 0; pidx < passes; ++pidx) {
                cmdPush(st.cbs[i], st.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &gClamped);
                cmdDraw(st.cbs[i], 3, 1, 0, 0);
            }
            cmdEndRP(st.cbs[i]);
            if (endCB(st.cbs[i]) != VK_SUCCESS) return false;
        }
        LogLine("BRIGHTNESS: post pass ready (%ux%u, format %d, %u images, gain %.2f = (1+%.3f)^%d, queue family %u)",
            info.ext.width, info.ext.height, (int)info.fmt, count, G, gClamped, passes, family);
        return true;
    }

    // Returns a semaphore the present must wait on (signalled by our submit), or VK_NULL_HANDLE to present unchanged.
    VkSemaphore Process(VkQueue queue, const VkPresentInfoKHR& pi) {
        if (pi.swapchainCount != 1 || !pi.pSwapchains || !pi.pImageIndices || pi.waitSemaphoreCount == 0 || !g_lastDevice) return VK_NULL_HANDLE;
        VkDevice dev = g_lastDevice;
        VkSwapchainKHR sc = pi.pSwapchains[0];
        VkSemaphore result = VK_NULL_HANDLE;
        AcquireSRWLockExclusive(&g_lock);
        auto iit = Infos().find(H(sc));
        if (iit != Infos().end()) {
            SwapState& st = States()[H(sc)];
            if (!st.failed && (!st.ready || st.generation != iit->second.generation)) {
                st = SwapState{};
                st.generation = iit->second.generation;
                if (Build(dev, queue, sc, iit->second, st)) st.ready = true;
                else { st.failed = true; LogLine("BRIGHTNESS: setup failed, feature disabled for this swapchain"); }
            }
            const uint32_t idx = *pi.pImageIndices;
            if (st.ready && idx < st.cbs.size()) {
                auto submit = Fn<PFN_vkQueueSubmit>(dev, "vkQueueSubmit");
                if (submit) {
                    std::vector<VkPipelineStageFlags> stageMask(pi.waitSemaphoreCount, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
                    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    si.waitSemaphoreCount = pi.waitSemaphoreCount; si.pWaitSemaphores = pi.pWaitSemaphores;
                    si.pWaitDstStageMask = stageMask.data();
                    si.commandBufferCount = 1; si.pCommandBuffers = &st.cbs[idx];
                    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &st.sems[idx];
                    if (submit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS) result = st.sems[idx];
                }
            }
        }
        ReleaseSRWLockExclusive(&g_lock);
        return result;
    }
}
}

// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    auto real = RealExport<PFN_vkCreateSwapchainKHR>("vkCreateSwapchainKHR");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    LoadSpvConfig();
    VkSwapchainCreateInfoKHR patchedInfo{};
    const VkSwapchainCreateInfoKHR* useInfo = pInfo;
    if (pInfo && g_spvCfg.brightness > 1.0f) {
        patchedInfo = *pInfo;
        patchedInfo.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;   // needed for the brightness pass
        useInfo = &patchedInfo;
    }
    VkResult r = real(device, useInfo, pAllocator, pSwapchain);
    if (r == VK_SUCCESS && pInfo && pSwapchain) post::Record(*pSwapchain, pInfo->imageFormat, pInfo->imageExtent);
    LogLine("vkCreateSwapchainKHR: format=%d colorspace=%d presentMode=%d minImg=%u %ux%u result=%d",
        pInfo ? (int)pInfo->imageFormat : -1,
        pInfo ? (int)pInfo->imageColorSpace : -1,
        pInfo ? (int)pInfo->presentMode : -1,
        pInfo ? pInfo->minImageCount : 0,
        pInfo ? pInfo->imageExtent.width : 0,
        pInfo ? pInfo->imageExtent.height : 0,
        (int)r);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule) {
    auto real = RealExport<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    if (!real) {
        LogLine("vkCreateShaderModule: RealExport NULL -> FAKE module");
        if (pShaderModule) *pShaderModule = MakeFake<VkShaderModule>();
        return VK_SUCCESS;
    }
    LoadSpvConfig();
    if (!pInfo || !pInfo->pCode || !pShaderModule) {
        if (pShaderModule) *pShaderModule = MakeFake<VkShaderModule>();
        return VK_SUCCESS;
    }

    const uint32_t* code = pInfo->pCode;
    const size_t nWords = pInfo->codeSize / 4;
    SpvInfo si = AnalyzeSpirv(code, nWords);
    si.origWords = nWords;
    static volatile LONG seen = 0;
    const LONG n = InterlockedIncrement(&seen);

    // Ray-query shaders: instead of denying them, remove the ray queries ("nothing was hit") so the rest of
    // the shader (lighting, materials) still runs on a GPU without ray tracing.
    std::vector<uint32_t> rewritten;
    bool didRewrite = false;
    // Allow RQ rewrite even when DenyAnyCaps also matched (e.g. Cap46) — strip RQ first.
    if (si.deny && si.rqOnly && g_spvCfg.rewriteRayQuery) {
        std::string err;
        static volatile LONG rwOk = 0, rwFail = 0;
        bool swDone = false;
        if (g_spvCfg.softwareRT && swpx::EnsureInit(g_lastDevice)) {      // [SWRT]
            std::string swErr;
            static volatile LONG swOk = 0, swFail = 0;
            if (spvrw::EmulateRayQuery(code, nWords, rewritten, swErr)) {
                swDone = true;
                const LONG k = InterlockedIncrement(&swOk);
                if (k <= 10 || k % 50 == 0) LogLine("SWRT: shader #%ld ray queries -> software BVH traversal (%zu -> %zu words), total=%ld", n, nWords, rewritten.size(), (long)k);
            } else {
                rewritten.clear();
                const LONG k = InterlockedIncrement(&swFail);
                if (k <= 20) LogLine("SWRT: shader #%ld cannot use software RT (%s) -> falling back to ray query stripping", n, swErr.c_str());
            }
        }
        if (swDone || spvrw::StripRayQuery(code, nWords, rewritten, err)) {
            // Only drop OpCapability 46/49 — never rewrite the body here (a broken mid-stream
            // word-count used to break the loop and truncate 11k words down to ~1827).
            {
                // [FIX] Начинаем с j = 5 (после 5-словного заголовка SPIR-V), а не с j = 0!
                // j = 0 трактовал magic 0x07230203 как инструкцию длиной 1827 слов и пропускал шапку.
                std::vector<uint32_t> stripped;
                stripped.reserve(rewritten.size());
                size_t j = 0;
                if (rewritten.size() >= 5 && rewritten[0] == 0x07230203u) {
                    for (j = 0; j < 5; ++j) stripped.push_back(rewritten[j]);
                }
                while (j < rewritten.size()) {
                    const uint32_t first = rewritten[j];
                    const uint32_t wc = first >> 16;
                    const uint32_t op = first & 0xFFFFu;
                    if (wc == 0 || j + wc > rewritten.size()) {
                        while (j < rewritten.size()) stripped.push_back(rewritten[j++]);
                        break;
                    }
                    if (op == 17 && wc >= 2) {
                        const uint32_t cap = rewritten[j + 1];
                        if (cap == 46 || cap == 49) { j += wc; continue; }
                    }
                    for (uint32_t t = 0; t < wc; ++t) stripped.push_back(rewritten[j + t]);
                    j += wc;
                }
                if (stripped.size() + 8 < rewritten.size())
                    LogLine("SPIRV: stripped Cap46/49 from RQ shader #%ld (%zu -> %zu)", n, rewritten.size(), stripped.size());
                rewritten.swap(stripped);
            }
            // Reject near-empty / over-stripped results (invalid SPIR-V → driver -3)
            if ((nWords > 4000 && rewritten.size() < 4000) ||
                (nWords > 3000 && rewritten.size() * 5 < nWords)) {
                LogLine("SPIRV: ray-query strip too aggressive #%ld (%zu -> %zu), discarding", n, nWords, rewritten.size());
                rewritten.clear();
                const LONG k = InterlockedIncrement(&rwFail);
                if (k <= 20) LogLine("SPIRV: ray-query rewrite FAILED for shader #%ld: over-stripped (shader stays denied)", n);
            } else {
                didRewrite = true;
                si.deny = false; si.reason.clear(); si.rewritten = true;
                if (g_spvCfg.promoteFloat16 && si.caps.find("Float16") != std::string::npos) {
                std::string err2;
                    std::vector<uint32_t> promoted;
                    if (spvrw::PromoteFloat16(rewritten.data(), rewritten.size(), promoted, err2) && promoted.size() > 100) {
                        rewritten.swap(promoted);
                        static volatile LONG fpRq = 0;
                        if (InterlockedIncrement(&fpRq) <= 15)
                            LogLine("SPIRV: Float16 promote after RQ rewrite shader #%ld (%zu words)", n, rewritten.size());
                    }
                }
                const LONG k = InterlockedIncrement(&rwOk);
                if (k <= 5 || k % 100 == 0) LogLine("SPIRV: ray-query shader #%ld rewritten (%zu -> %zu words), total rewritten=%ld", n, nWords, rewritten.size(), (long)k);
                if (k <= g_spvCfg.dumpRewrite) {
                    DumpSpirv("rq_in", n, code, pInfo->codeSize);
                    DumpSpirv("rq_out", n, rewritten.data(), rewritten.size() * 4);
                }
            }
        } else {
            rewritten.clear();
            const LONG k = InterlockedIncrement(&rwFail);
            if (k <= 20) LogLine("SPIRV: ray-query rewrite FAILED for shader #%ld: %s (shader stays denied)", n, err.c_str());
            if (k <= g_spvCfg.dumpRewriteFail) DumpSpirv("rqfail", n, code, pInfo->codeSize);
        }
    }
    // Promote Float16 ALU to Float32 so GCN2 can run menu/UI shaders that require shaderFloat16.
    // Also run when RQ rewrite did not already promote (RQ path promotes itself).
if (g_spvCfg.lowerFp16AndInt16 && (si.caps.find("Float16") != std::string::npos || si.caps.find("Int16") != std::string::npos)) {
    std::string err;
    std::vector<uint32_t> lowered;
    static volatile LONG loOk = 0, loFail = 0, loDump = 0, loBad = 0;
    const uint32_t* src = didRewrite ? rewritten.data() : code;
    size_t srcN = didRewrite ? rewritten.size() : nWords;
    bool ok = spvrw::LowerFp16AndInt16Alu(src, srcN, lowered, err, g_spvCfg.dropF16Cap != 0) && lowered.size() > 100;
    if (ok && !spvrw::CheckStructure(lowered.data(), lowered.size(), err)) {
        ok = false;
        const LONG b = InterlockedIncrement(&loBad);
        if (b <= 20) LogLine("SPIRV: fp16+int16 lower BAD MODULE #%ld: %s (rejected, original kept)", n, err.c_str());
    }
    if (ok) {
        const LONG d = InterlockedIncrement(&loDump);
        if (d <= g_spvCfg.dumpRewrite) {
            DumpSpirv("low_in", n, src, srcN * 4);
            DumpSpirv("low_out", n, lowered.data(), lowered.size() * 4);
        }
        rewritten = std::move(lowered);
        didRewrite = true;
        si.rewritten = true;
        if (si.reason.find("Float16") != std::string::npos || si.reason.find("Int16") != std::string::npos) {
            si.deny = false; si.reason.clear();
        }
        const LONG k = InterlockedIncrement(&loOk);
        if (k <= 10 || k % 50 == 0)
            LogLine("SPIRV: fp16+int16 ALU->32-bit shader #%ld (%zu -> %zu words), total=%ld",
                n, srcN, rewritten.size(), (long)k);
    } else {
        const LONG k = InterlockedIncrement(&loFail);
        if (k <= 15) LogLine("SPIRV: fp16+int16 lower FAILED #%ld: %s", n, err.c_str());
    }
}
    if (g_spvCfg.dumpSamples > 0 && si.model == 5) {
        static volatile LONG cubeSamples = 0, i64Samples = 0;
        const bool c45 = si.caps.find("cap45") != std::string::npos, i64 = si.caps.find("Int64") != std::string::npos;
        if (c45 && !i64 && InterlockedIncrement(&cubeSamples) <= g_spvCfg.dumpSamples) DumpSpirv("ok_cubearray", n, code, pInfo->codeSize);
        else if (i64 && !c45 && InterlockedIncrement(&i64Samples) <= g_spvCfg.dumpSamples) DumpSpirv("ok_int64", n, code, pInfo->codeSize);
    }
    static volatile LONG deniedSeen = 0;
    const LONG dn = si.deny ? InterlockedIncrement(&deniedSeen) : 0;
    const bool logIt = g_spvCfg.logAll || (si.deny && dn <= 30) || n <= 10;
    if (n % 200 == 0) LogLine("SHADER stats: total=%ld denied=%ld", n, (long)deniedSeen);
    if (logIt)
        LogLine("SHADER #%ld spv=%u.%u model=%u size=%zu caps=[%s]%s%s", n,
            (si.version >> 16) & 0xFF, (si.version >> 8) & 0xFF, si.model, pInfo->codeSize, si.caps.c_str(),
            si.deny ? " DENIED: " : "", si.deny ? si.reason.c_str() : "");

    // Pre-made replacement for a module that is known to crash this driver (keyed by hash of the original).
    std::vector<uint32_t> replacement;
    bool didReplace = false;
    if (g_replMap && !g_replMap->empty()) {
        const uint64_t h = Fnv1a64(code, pInfo->codeSize);
        auto it = g_replMap->find(h);
        if (it != g_replMap->end() && ReadReplacement(it->second, replacement)) {
            didReplace = true;
            si.deny = false; si.reason.clear();
            LogLine("SPIRV: shader #%ld REPLACED by spv_replace file (hash %016llX, %zu -> %zu words)", n,
                static_cast<unsigned long long>(h), nWords, replacement.size());
        }
    }

    // SafeModeAfterCrashes REMOVED — it FAKE'd menu/UI shaders after a few
    // recovered BindBufferMemory crashes → black menu with audio/cursor only.

    if (si.deny) {
        // The driver cannot run this module (GCN 2 has no fp16/int16/dot-product/RT/mesh).
        static volatile LONG fakeN = 0;
        LONG fn = InterlockedIncrement(&fakeN);
        if (fn <= 20 || fn % 50 == 0)
            LogLine("vkCreateShaderModule #%ld -> FAKE module (reason=%s) totalFake=%ld [rtfill зальёт его storage-выходы]", n, si.reason.c_str(), (long)fn);
        *pShaderModule = MakeFake<VkShaderModule>();
        PutSpvInfo(*pShaderModule, si);
        return VK_SUCCESS;
    }

    VkShaderModuleCreateInfo ci = *pInfo;
    std::vector<uint32_t> patched;
    const uint32_t* useCode = didReplace ? replacement.data() : (didRewrite ? rewritten.data() : code);
    size_t useWords = didReplace ? replacement.size() : (didRewrite ? rewritten.size() : nWords);
    if (g_spvCfg.patchVersion && si.version > 0x00010500u && si.version != 0) {
        patched.assign(useCode, useCode + useWords);
        patched[1] = 0x00010500u;                       // driver supports up to SPIR-V 1.5
        useCode = patched.data();
        if (FirstTime("spv:patchver")) LogLine("SPIRV: patching module version %x -> 1.5 (driver limit)", si.version);
    }
	
	
	// OpTerminateInvocation (4416) есть только в SPIR-V 1.6. После патча версии на 1.5
// модуль невалиден -> заменяем на OpKill (252), тот же эффект во фрагментном шейдере.
{
    bool need = false;
    for (size_t j = (useWords >= 5 && useCode[0] == 0x07230203u) ? 5 : 0; j < useWords;) { // [FIX] пропускаем 5-словный заголовок
        const uint32_t wc = useCode[j] >> 16, op = useCode[j] & 0xFFFFu;
        if (wc == 0 || j + wc > useWords) break;
        if (op == 4416) { need = true; break; }
        j += wc;
    }
    if (need) {
        std::vector<uint32_t> tk(useCode, useCode + useWords);
        int fixed = 0;
        for (size_t j = (tk.size() >= 5 && tk[0] == 0x07230203u) ? 5 : 0; j < tk.size();) {
            const uint32_t wc = tk[j] >> 16, op = tk[j] & 0xFFFFu;
            if (wc == 0 || j + wc > tk.size()) break;
            if (op == 4416) { tk[j] = (wc << 16) | 252u; ++fixed; }
            j += wc;
        }
        static volatile LONG tiN = 0;
        if (InterlockedIncrement(&tiN) <= 10)
            LogLine("SPIRV: OpTerminateInvocation -> OpKill x%d in shader #%ld", fixed, n);
        rewritten.swap(tk);
        useCode = rewritten.data();
        useWords = rewritten.size();
    }
}
	
	
    // v8: сначала понижаем неподдерживаемые инструкции.
{
    std::vector<uint32_t> low;
    std::string lerr;
    if (spvrw::LowerUnsupportedOps(useCode, useWords, low, lerr)) {
        static volatile LONG loN = 0;
        if (InterlockedIncrement(&loN) <= 20)
            LogLine("SPIRV: lowered clock/subgroup-ballot in shader #%ld (%zu -> %zu words)", n, useWords, low.size());
        rewritten.swap(low);
        useCode = rewritten.data();
        useWords = rewritten.size();
    }
}
// Удаляем ТОЛЬКО те capability, которые реально больше не нужны.
// Удаление cap при живых 16-битных инструкциях = невалидный SPIR-V = -13.
{
    bool has16f = false, has16i = false;
    for (size_t j = 0; j < useWords;) {
        const uint32_t wc = useCode[j] >> 16, op = useCode[j] & 0xFFFFu;
        if (wc == 0 || j + wc > useWords) break;
        if (op == 22 && wc >= 3 && useCode[j + 2] == 16) has16f = true;
        if (op == 21 && wc >= 3 && useCode[j + 2] == 16) has16i = true;
        if (op == 54) break;
        j += wc;
    }
    std::vector<uint32_t> out;
    out.reserve(useWords);
    size_t j = 0; int dropped = 0;
    while (j < useWords) {
        const uint32_t wc = useCode[j] >> 16, op = useCode[j] & 0xFFFFu;
        if (wc == 0 || j + wc > useWords) { while (j < useWords) out.push_back(useCode[j++]); break; }
        if (op == 17 && wc >= 2) {
            const uint32_t cap = useCode[j + 1];
            if ((cap == 9 && !has16f) || (cap == 22 && !has16i)) { ++dropped; j += wc; continue; }
        }
        for (uint32_t t = 0; t < wc; ++t) out.push_back(useCode[j + t]);
        j += wc;
    }
    if (dropped > 0) {
        rewritten.swap(out);
        useCode = rewritten.data();
        useWords = rewritten.size();
    }
}



    // OpTerminateInvocation (4416) есть только в SPIR-V 1.6. После патча версии на 1.5
// модуль невалиден -> OpKill (252), тот же эффект во фрагментном шейдере.
{
    bool need = false;
    for (size_t j = 0; j < useWords;) {
        const uint32_t wc = useCode[j] >> 16, op = useCode[j] & 0xFFFFu;
        if (wc == 0 || j + wc > useWords) break;
        if (op == 4416) { need = true; break; }
        j += wc;
    }
    if (need) {
        std::vector<uint32_t> tk(useCode, useCode + useWords);
        int fixed = 0;
        for (size_t j = 0; j < tk.size();) {
            const uint32_t wc = tk[j] >> 16, op = tk[j] & 0xFFFFu;
            if (wc == 0 || j + wc > tk.size()) break;
            if (op == 4416) { tk[j] = (wc << 16) | 252u; ++fixed; }
            j += wc;
        }
        static volatile LONG tiN = 0;
        if (InterlockedIncrement(&tiN) <= 10)
            LogLine("SPIRV: OpTerminateInvocation -> OpKill x%d in shader #%ld", fixed, n);
        rewritten.swap(tk);
        useCode = rewritten.data();
        useWords = rewritten.size();
    }
}


    ci.pCode = useCode;
    ci.codeSize = useWords * 4;
    Crumb("vkCreateShaderModule #%ld caps=[%s]", n, si.caps.c_str());
    SetJob("vkCreateShaderModule #%ld size=%zu caps=[%s]", n, pInfo->codeSize, si.caps.c_str());
    VkResult r = VK_ERROR_UNKNOWN;
    const bool callOk = DriverCall([&]() { r = real(device, &ci, pAllocator, pShaderModule); });
    if (!callOk) {
        LogLine("CAUGHT driver crash in vkCreateShaderModule #%ld (spv=%x size=%zu caps=[%s])", n, si.version, pInfo->codeSize, si.caps.c_str());
        DumpSpirv("crash", n, useCode, useWords * 4);
        r = VK_ERROR_UNKNOWN;
    }
    // The old driver's SPIR-V front end crashes on some constructs (e.g. OpConvertUToPtr). Try equivalent
    // rewrites of the module before giving up on it.
    std::vector<uint32_t> variantCode;
    if (r != VK_SUCCESS && r != static_cast<VkResult>(-13) && g_spvCfg.tryVariants) {
        for (int v = 1; v <= 2 && r != VK_SUCCESS; ++v) {
            std::vector<uint32_t> var;
            std::string verr;
            if (!spvrw::ApplyVariant(v, useCode, useWords, var, verr)) continue;
            VkShaderModuleCreateInfo ci2 = ci;
            ci2.pCode = var.data();
            ci2.codeSize = var.size() * 4;
            SetJob("vkCreateShaderModule #%ld VARIANT %d caps=[%s]", n, v, si.caps.c_str());
            VkResult r2 = VK_ERROR_UNKNOWN;
            const bool ok2 = DriverCall([&]() { r2 = real(device, &ci2, pAllocator, pShaderModule); });
            if (ok2 && r2 == VK_SUCCESS) {
                r = VK_SUCCESS;
                variantCode = std::move(var);
                static volatile LONG worked = 0;
                const LONG k = InterlockedIncrement(&worked);
                if (k <= 10) LogLine("VARIANT %d WORKED for shader #%ld (caps=[%s]); total variant successes=%ld", v, n, si.caps.c_str(), (long)k);
            } else {
                LogLine("shader #%ld variant %d %s", n, v, ok2 ? "rejected by driver" : "CRASHED the driver again");
            }
        }
    }
    if (r != VK_SUCCESS) {
        LogLine("vkCreateShaderModule #%ld FAILED: result=%d caps=[%s] -> FAKE module", n, (int)r, si.caps.c_str());
        *pShaderModule = MakeFake<VkShaderModule>();
        PutSpvInfo(*pShaderModule, si);
        return VK_SUCCESS;
    }
    if (g_spvCfg.keepCode) {
        if (!variantCode.empty()) si.code = std::make_shared<std::vector<uint32_t>>(variantCode);
        else si.code = std::make_shared<std::vector<uint32_t>>(useCode, useCode + useWords);
    }
    PutSpvInfo(*pShaderModule, si);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    auto real = RealExport<PFN_vkQueuePresentKHR>("vkQueuePresentKHR");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;

    LoadSpvConfig();
    VkPresentInfoKHR modifiedPresent{};
    const VkPresentInfoKHR* usePresent = pPresentInfo;
    VkSemaphore postSem = VK_NULL_HANDLE;
    if (g_spvCfg.brightness > 1.0f && pPresentInfo) {
        const VkSemaphore s = post::Process(queue, *pPresentInfo);
        if (s) {
            postSem = s;
            modifiedPresent = *pPresentInfo;
            modifiedPresent.waitSemaphoreCount = 1;
            modifiedPresent.pWaitSemaphores = &postSem;
            usePresent = &modifiedPresent;
        }
    }
    VkResult r = real(queue, usePresent);
    // [v3.2] frame counter + visible failures
    InterlockedIncrement(&g_presentCount);
    InterlockedExchange(&g_lastPresentRes, static_cast<LONG>(r));
    InterlockedExchange64(&g_lastPresentTick, static_cast<LONG64>(GetTickCount64()));
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        static volatile LONG badPresent = 0;
        const LONG n = InterlockedIncrement(&badPresent);
        if (n <= 20 || n % 500 == 0)
            LogLine("!!! vkQueuePresentKHR FAILED: result=%d%s (failures so far=%ld)", static_cast<int>(r),
                r == VK_ERROR_DEVICE_LOST ? " VK_ERROR_DEVICE_LOST (GPU hang / driver reset)" :
                r == VK_ERROR_OUT_OF_DATE_KHR ? " VK_ERROR_OUT_OF_DATE_KHR" :
                r == VK_ERROR_SURFACE_LOST_KHR ? " VK_ERROR_SURFACE_LOST_KHR" : "", static_cast<long>(n));
    }
    static bool firstPresent = true;
    if (firstPresent) {
        firstPresent = false;
        LoadSpvConfig();
        LogLine("vkQueuePresentKHR: first present done (result=%d)", (int)r);
        InstallCrashHandlerFirst();
        // Old debugging hack, now optional: VulkanProxy.ini [spirv] ExitAfterFirstPresent=1
        wchar_t ip[2048]{};
        GetLogPath(ip, sizeof(ip) / sizeof(ip[0]));
        wchar_t* sl = wcsrchr(ip, L'\\');
        if (sl) {
            wcscpy_s(sl + 1, (sizeof(ip) / sizeof(ip[0])) - static_cast<size_t>(sl + 1 - ip), L"VulkanProxy.ini");
            if (GetPrivateProfileIntW(L"spirv", L"ExitAfterFirstPresent", 0, ip)) {
                Sleep(500);
                ExitProcess(0);
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// v5: диагностика instance / surface / ресурсов
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateInstance(const VkInstanceCreateInfo* pInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    auto real = RealExport<PFN_vkCreateInstance>("vkCreateInstance");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (pInfo) {
        LogLine("vkCreateInstance: apiVersion=%u.%u extCount=%u layerCount=%u",
            VK_API_VERSION_MAJOR(pInfo->pApplicationInfo ? pInfo->pApplicationInfo->apiVersion : 0),
            VK_API_VERSION_MINOR(pInfo->pApplicationInfo ? pInfo->pApplicationInfo->apiVersion : 0),
            pInfo->enabledExtensionCount, pInfo->enabledLayerCount);
        for (uint32_t i = 0; i < pInfo->enabledExtensionCount; ++i)
            LogLine("  INSTANCE EXT: %s", pInfo->ppEnabledExtensionNames[i]);
        for (uint32_t i = 0; i < pInfo->enabledLayerCount; ++i)
            LogLine("  INSTANCE LAYER: %s", pInfo->ppEnabledLayerNames[i]);
    }
    VkResult r = real(pInfo, pAllocator, pInstance);
    LogLine("vkCreateInstance result=%d", (int)r);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pInfo,
    const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) {
    auto real = RealExport<PFN_vkCreateWin32SurfaceKHR>("vkCreateWin32SurfaceKHR");
    if (!real) return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkResult r = real(instance, pInfo, pAllocator, pSurface);
    LogLine("vkCreateWin32SurfaceKHR result=%d (hwnd=%p)", (int)r, pInfo ? (void*)pInfo->hwnd : nullptr);
    return r;
}


// Safe no-ops for optional debug/utils functions some engines call without null-checking.
static VKAPI_ATTR VkResult VKAPI_CALL Stub_CreateDebugUtilsMessengerEXT(
    VkInstance, const void*, const VkAllocationCallbacks*, void** pMessenger) {
    if (pMessenger) *pMessenger = (void*)(uintptr_t)0x1;
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL Stub_DestroyDebugUtilsMessengerEXT(VkInstance, void*, const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL Stub_SubmitDebugUtilsMessageEXT(VkInstance, int, int, const void*) {}
static VKAPI_ATTR VkResult VKAPI_CALL Stub_SetDebugUtilsObjectNameEXT(VkDevice, const void*) { return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL Stub_SetDebugUtilsObjectTagEXT(VkDevice, const void*) { return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL Stub_CmdBeginDebugUtilsLabelEXT(VkCommandBuffer, const void*) {}
static VKAPI_ATTR void VKAPI_CALL Stub_CmdEndDebugUtilsLabelEXT(VkCommandBuffer) {}
static VKAPI_ATTR void VKAPI_CALL Stub_CmdInsertDebugUtilsLabelEXT(VkCommandBuffer, const void*) {}
static VKAPI_ATTR void VKAPI_CALL Stub_QueueBeginDebugUtilsLabelEXT(VkQueue, const void*) {}
static VKAPI_ATTR void VKAPI_CALL Stub_QueueEndDebugUtilsLabelEXT(VkQueue) {}
static VKAPI_ATTR void VKAPI_CALL Stub_QueueInsertDebugUtilsLabelEXT(VkQueue, const void*) {}
static VKAPI_ATTR VkResult VKAPI_CALL Stub_CreateDebugReportCallbackEXT(VkInstance, const void*, const VkAllocationCallbacks*, void** p) {
    if (p) *p = (void*)(uintptr_t)0x1; return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL Stub_DestroyDebugReportCallbackEXT(VkInstance, void*, const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL Stub_DebugReportMessageEXT(VkInstance, int, int, uint64_t, size_t, int32_t, const char*, const char*) {}

// vkDestroySurfaceKHR / vkGetPhysicalDeviceSurfaceSupportKHR /
// vkGetPhysicalDeviceFormatProperties2 are provided by generated thunks
// (vk_proxy_thunks.generated.asm). Do NOT redefine them here (LNK2005).
// They are served to the game via GIPA by looking them up in our own module.

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    VkSurfaceCapabilitiesKHR* pCaps) {
    auto real = RealExport<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>("vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (!real) return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkResult r = real(pd, surface, pCaps);
    if (r == VK_SUCCESS && pCaps) {
        LogLine("vkGetPhysicalDeviceSurfaceCapabilitiesKHR: minImg=%u maxImg=%u extent=%ux%u curTransform=%d maxLayers=%u",
            pCaps->minImageCount, pCaps->maxImageCount,
            pCaps->currentExtent.width, pCaps->currentExtent.height,
            (int)pCaps->currentTransform, pCaps->maxImageArrayLayers);
    } else {
        LogLine("vkGetPhysicalDeviceSurfaceCapabilitiesKHR result=%d", (int)r);
    }
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    uint32_t* pCount, VkSurfaceFormatKHR* pFormats) {
    auto real = RealExport<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>("vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (!real) return VK_ERROR_EXTENSION_NOT_PRESENT;
    if (!pCount) return real(pd, surface, pCount, pFormats);

    // Спросим у драйвера реальный список.
    uint32_t driverCount = 0;
    VkResult r = real(pd, surface, &driverCount, nullptr);
    if (r != VK_SUCCESS || driverCount > 256) return r;
    std::vector<VkSurfaceFormatKHR> driverFormats(driverCount);
    if (driverCount) {
        r = real(pd, surface, &driverCount, driverFormats.data());
        if (r != VK_SUCCESS) return r;
        driverFormats.resize(driverCount);
    }

    // Оставляем только безопасные форматы (B8G8R8A8 и R8G8B8A8, UNORM/SRGB).
    std::vector<VkSurfaceFormatKHR> safe;
    auto pushUnique = [&](VkFormat f, VkColorSpaceKHR cs) {
        for (auto& s : safe) if (s.format == f && s.colorSpace == cs) return;
        VkSurfaceFormatKHR x{}; x.format = f; x.colorSpace = cs; safe.push_back(x);
    };
    // Сначала — то, что реально поддерживает драйвер и что входит в safe set.
    for (auto& f : driverFormats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_SRGB ||
             f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            pushUnique(f.format, f.colorSpace);
        }
    }
    // Если ни одного safe не нашлось — хотя бы SRGB_NONLINEAR + B8G8R8A8_UNORM.
    if (safe.empty()) pushUnique(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);

    static volatile LONG logged = 0;
    if (InterlockedExchange(&logged, 1) == 0) {
        LogLine("vkGetPhysicalDeviceSurfaceFormatsKHR: driver had %u formats, returning %u safe",
            driverCount, (unsigned)safe.size());
        for (auto& f : driverFormats) LogLine("  DRIVER FMT: format=%d colorspace=%d", (int)f.format, (int)f.colorSpace);
        for (auto& f : safe) LogLine("  SAFE   FMT: format=%d colorspace=%d", (int)f.format, (int)f.colorSpace);
    }

    const uint32_t total = (uint32_t)safe.size();
    if (!pFormats) { *pCount = total; return VK_SUCCESS; }
    const uint32_t cap = *pCount;
    const uint32_t n = cap < total ? cap : total;
    for (uint32_t i = 0; i < n; ++i) pFormats[i] = safe[i];
    *pCount = n;
    return n < total ? VK_INCOMPLETE : VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    uint32_t* pCount, VkPresentModeKHR* pModes) {
    auto real = RealExport<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>("vkGetPhysicalDeviceSurfacePresentModesKHR");
    if (!real) return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkResult r = real(pd, surface, pCount, pModes);
    // First call is the count query: pModes == nullptr. Never dereference it then.
    if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && pCount && pModes) {
        static volatile LONG logged = 0;
        if (InterlockedExchange(&logged, 1) == 0) {
            LogLine("vkGetPhysicalDeviceSurfacePresentModesKHR: count=%u", *pCount);
            for (uint32_t i = 0; i < *pCount && i < 16; ++i)
                LogLine("  PRESENT MODE: %d", (int)pModes[i]);
        }
    }
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImage(VkDevice device, const VkImageCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
    static volatile LONG cnt = 0;
    if (pInfo && InterlockedIncrement(&cnt) <= 60) {
        LogLine("vkCreateImage ENTER: format=%d %ux%ux%u mip=%u layers=%u usage=0x%X tiling=%d",
            (int)pInfo->format, pInfo->extent.width, pInfo->extent.height, pInfo->extent.depth,
            pInfo->mipLevels, pInfo->arrayLayers, (unsigned)pInfo->usage, (int)pInfo->tiling);
    }
    auto real = RealExport<PFN_vkCreateImage>("vkCreateImage");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pImage);
    if (r != VK_SUCCESS) {
        LogLine("vkCreateImage FAILED: result=%d format=%d type=%d %ux%ux%u mip=%u layers=%u samples=%u usage=0x%X tiling=%d",
            (int)r,
            pInfo ? (int)pInfo->format : -1,
            pInfo ? (int)pInfo->imageType : -1,
            pInfo ? pInfo->extent.width : 0,
            pInfo ? pInfo->extent.height : 0,
            pInfo ? pInfo->extent.depth : 0,
            pInfo ? pInfo->mipLevels : 0,
            pInfo ? pInfo->arrayLayers : 0,
            pInfo ? (unsigned)pInfo->samples : 0,
            pInfo ? (unsigned)pInfo->usage : 0,
            pInfo ? (int)pInfo->tiling : -1);
    } else {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 40) {
            LogLine("vkCreateImage: format=%d %ux%ux%u mip=%u layers=%u usage=0x%X tiling=%d mem=0x%X",
                pInfo ? (int)pInfo->format : -1,
                pInfo ? pInfo->extent.width : 0,
                pInfo ? pInfo->extent.height : 0,
                pInfo ? pInfo->extent.depth : 0,
                pInfo ? pInfo->mipLevels : 0,
                pInfo ? pInfo->arrayLayers : 0,
                pInfo ? (unsigned)pInfo->usage : 0,
                pInfo ? (int)pInfo->tiling : -1,
                pInfo ? (unsigned)pInfo->sharingMode : 0);
        }
    }
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemReq) {
    auto real = RealExport<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements");
    if (real) real(device, image, pMemReq);
    if (pMemReq) LogLine("vkGetImageMemoryRequirements: size=%llu align=%llu memTypeBits=0x%X",
        (unsigned long long)pMemReq->size, (unsigned long long)pMemReq->alignment, pMemReq->memoryTypeBits);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetImageMemoryRequirements2(VkDevice device, const VkImageMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemReq) {
    auto real = RealExport<PFN_vkGetImageMemoryRequirements2>("vkGetImageMemoryRequirements2");
    if (real) real(device, pInfo, pMemReq);
    if (pMemReq) LogLine("vkGetImageMemoryRequirements2: size=%llu align=%llu memTypeBits=0x%X",
        (unsigned long long)pMemReq->memoryRequirements.size,
        (unsigned long long)pMemReq->memoryRequirements.alignment,
        pMemReq->memoryRequirements.memoryTypeBits);
}

#include "swrt_proxy.h"   // [SWRT] software ray tracing: AS records, GPU BVH build, AS query pools

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) {
    auto real = RealExport<PFN_vkAllocateMemory>("vkAllocateMemory");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pMemory);
    if (r != VK_SUCCESS) {
        LogLine("vkAllocateMemory FAILED: result=%d size=%llu typeIdx=%u",
            (int)r, pInfo ? (unsigned long long)pInfo->allocationSize : 0,
            pInfo ? pInfo->memoryTypeIndex : 0);
    } else {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 30) {
            LogLine("vkAllocateMemory: size=%llu typeIdx=%u",
                pInfo ? (unsigned long long)pInfo->allocationSize : 0,
                pInfo ? pInfo->memoryTypeIndex : 0);
        }
    }
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
    auto real = RealExport<PFN_vkBindImageMemory>("vkBindImageMemory");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    // Never pass FAKE/null handles into the old AMD driver - it AVs at +0x8.
    if (!image || !memory || IsFake(image) || IsFake(memory)) {
        static volatile LONG n = 0;
        if (InterlockedIncrement(&n) <= 20)
            LogLine("vkBindImageMemory: skipped FAKE/null image=%p memory=%p", (void*)image, (void*)memory);
        return VK_SUCCESS;
    }
    VkResult r = VK_ERROR_UNKNOWN;
    if (!DriverCall([&]() { r = real(device, image, memory, offset); }, false)) {   // [v3.2] not gated by shader compiles
        LogLine("CAUGHT driver crash in vkBindImageMemory image=%p memory=%p offset=%llu",
            (void*)image, (void*)memory, (unsigned long long)offset);
        return VK_SUCCESS; // keep the game alive
    }
    if (r != VK_SUCCESS) LogLine("vkBindImageMemory FAILED: result=%d offset=%llu", (int)r, (unsigned long long)offset);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindImageMemory2(VkDevice device, uint32_t bindCount, const VkBindImageMemoryInfo* pBinds) {
    auto real = RealExport<PFN_vkBindImageMemory2>("vkBindImageMemory2");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pBinds || bindCount == 0) return VK_SUCCESS;
    // Filter out FAKE/null binds
    std::vector<VkBindImageMemoryInfo> safe;
    safe.reserve(bindCount);
    for (uint32_t i = 0; i < bindCount; ++i) {
        if (!pBinds[i].image || !pBinds[i].memory || IsFake(pBinds[i].image) || IsFake(pBinds[i].memory))
            continue;
        safe.push_back(pBinds[i]);
    }
    if (safe.empty()) return VK_SUCCESS;
    VkResult r = VK_ERROR_UNKNOWN;
    if (!DriverCall([&]() { r = real(device, (uint32_t)safe.size(), safe.data()); }, false)) {
        LogLine("CAUGHT driver crash in vkBindImageMemory2 count=%u", bindCount);
        return VK_SUCCESS;
    }
    if (r != VK_SUCCESS) LogLine("vkBindImageMemory2 FAILED: result=%d count=%u", (int)r, bindCount);
    return r;
}

// Use the same effective usage flags for actual creation and maintenance4 queries.
static bool PatchSwrtBufferCreateInfo(const VkBufferCreateInfo& source, VkBufferCreateInfo& patched) {
    constexpr VkBufferUsageFlags kRtBits = 0x00080000 | 0x00100000 | 0x00000400;
    LoadSpvConfig();
    if (!g_spvCfg.softwareRT || !g_spvCfg.swrtPatchBuffers || !swpx::Ready() || !(source.usage & kRtBits)) return false;
    patched = source;
    patched.usage = (source.usage & ~kRtBits) | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return true;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer) {
    auto real = RealExport<PFN_vkCreateBuffer>("vkCreateBuffer");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkBufferCreateInfo patched{};
    if (pInfo && PatchSwrtBufferCreateInfo(*pInfo, patched)) {
        static volatile LONG n = 0;
        if (InterlockedIncrement(&n) <= 8) LogLine("SWRT: vkCreateBuffer RT usage 0x%X -> 0x%X (size=%llu)", (unsigned)pInfo->usage, (unsigned)patched.usage, (unsigned long long)pInfo->size);
        pInfo = &patched;
    }
    if (pInfo && pInfo->size == 0) {
        static volatile LONG zeroCount = 0;
        if (InterlockedIncrement(&zeroCount) <= 20)
            LogLine("!!! vkCreateBuffer called with size=0, usage=0x%X (invalid Vulkan create info; check preceding EXEC@0)", (unsigned)pInfo->usage);
    }
    VkResult r = real(device, pInfo, pAllocator, pBuffer);
    if (r != VK_SUCCESS) {
        LogLine("vkCreateBuffer FAILED: result=%d size=%llu usage=0x%X",
            (int)r, pInfo ? (unsigned long long)pInfo->size : 0, pInfo ? (unsigned)pInfo->usage : 0);
    } else {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 30) {
            LogLine("vkCreateBuffer: size=%llu usage=0x%X",
                pInfo ? (unsigned long long)pInfo->size : 0, pInfo ? (unsigned)pInfo->usage : 0);
        }
    }
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemReq) {
    auto real = RealExport<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    if (real) real(device, buffer, pMemReq);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetBufferMemoryRequirements2(VkDevice device, const VkBufferMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemReq) {
    auto real = RealExport<PFN_vkGetBufferMemoryRequirements2>("vkGetBufferMemoryRequirements2");
    if (real) real(device, pInfo, pMemReq);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    auto real = RealExport<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    // Old AMDVLK AVs at +0x8 when given a null/FAKE buffer or memory object.
    if (!buffer || !memory || IsFake(buffer) || IsFake(memory)) {
        static volatile LONG n = 0;
        if (InterlockedIncrement(&n) <= 20)
            LogLine("vkBindBufferMemory: skipped FAKE/null buffer=%p memory=%p", (void*)buffer, (void*)memory);
        return VK_SUCCESS;
    }
    VkResult r = VK_ERROR_UNKNOWN;
    if (!DriverCall([&]() { r = real(device, buffer, memory, offset); }, false)) {   // [v3.2] not gated by shader compiles
        LogLine("CAUGHT driver crash in vkBindBufferMemory buffer=%p memory=%p offset=%llu",
            (void*)buffer, (void*)memory, (unsigned long long)offset);
        return VK_SUCCESS; // keep the game alive - menu is more important than this bind
    }
    if (r != VK_SUCCESS) LogLine("vkBindBufferMemory FAILED: result=%d offset=%llu", (int)r, (unsigned long long)offset);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindBufferMemory2(VkDevice device, uint32_t bindCount, const VkBindBufferMemoryInfo* pBinds) {
    auto real = RealExport<PFN_vkBindBufferMemory2>("vkBindBufferMemory2");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pBinds || bindCount == 0) return VK_SUCCESS;
    std::vector<VkBindBufferMemoryInfo> safe;
    safe.reserve(bindCount);
    for (uint32_t i = 0; i < bindCount; ++i) {
        if (!pBinds[i].buffer || !pBinds[i].memory || IsFake(pBinds[i].buffer) || IsFake(pBinds[i].memory))
            continue;
        safe.push_back(pBinds[i]);
    }
    if (safe.empty()) return VK_SUCCESS;
    VkResult r = VK_ERROR_UNKNOWN;
    if (!DriverCall([&]() { r = real(device, (uint32_t)safe.size(), safe.data()); }, false)) {
        LogLine("CAUGHT driver crash in vkBindBufferMemory2 count=%u", bindCount);
        return VK_SUCCESS;
    }
    if (r != VK_SUCCESS) LogLine("vkBindBufferMemory2 FAILED: result=%d count=%u", (int)r, bindCount);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
    auto real = RealExport<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    // [SWRT] acceleration-structure bindings become read-only storage buffers that hold the (software) TLAS blob
    std::vector<VkDescriptorSetLayoutBinding> swBind; VkDescriptorSetLayoutCreateInfo swInfo;
    if (pInfo && g_spvCfg.softwareRT && pInfo->pBindings) {
        bool any = false; for (uint32_t i = 0; i < pInfo->bindingCount; ++i) if ((int)pInfo->pBindings[i].descriptorType == 1000150000) any = true;
        if (any) {
            swBind.assign(pInfo->pBindings, pInfo->pBindings + pInfo->bindingCount);
            for (auto& b : swBind) if ((int)b.descriptorType == 1000150000) b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            swInfo = *pInfo; swInfo.pBindings = swBind.data(); pInfo = &swInfo;
            static volatile LONG n = 0; if (InterlockedIncrement(&n) <= 5) LogLine("SWRT: descriptor set layout: AS binding -> STORAGE_BUFFER");
        }
    }
    VkResult r = real(device, pInfo, pAllocator, pSetLayout);
    if (r != VK_SUCCESS) {
        LogLine("vkCreateDescriptorSetLayout FAILED: result=%d bindingCount=%u flags=0x%X",
            (int)r, pInfo ? pInfo->bindingCount : 0, pInfo ? (unsigned)pInfo->flags : 0);
        if (pInfo) for (uint32_t i = 0; i < pInfo->bindingCount && i < 32; ++i)
            LogLine("  BINDING: %u type=%d count=%u stages=0x%X",
                pInfo->pBindings[i].binding, (int)pInfo->pBindings[i].descriptorType,
                pInfo->pBindings[i].descriptorCount, (unsigned)pInfo->pBindings[i].stageFlags);
    } else {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 10)
            LogLine("vkCreateDescriptorSetLayout: bindingCount=%u", pInfo ? pInfo->bindingCount : 0);
    }
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout) {
    auto real = RealExport<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pPipelineLayout);
    if (r != VK_SUCCESS) {
        LogLine("vkCreatePipelineLayout FAILED: result=%d setLayouts=%u pushRanges=%u",
            (int)r, pInfo ? pInfo->setLayoutCount : 0, pInfo ? pInfo->pushConstantRangeCount : 0);
    } else {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 10)
            LogLine("vkCreatePipelineLayout: setLayouts=%u pushRanges=%u",
                pInfo ? pInfo->setLayoutCount : 0, pInfo ? pInfo->pushConstantRangeCount : 0);
    }
    return r;
}

// ---------------------------------------------------------------------------
// v6: расширенная локализация краша
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
    if (pInfo) {
        LogLine("vkCreateSampler ENTER: magFilter=%d minFilter=%d mipmap=%d addrU/V/W=%d/%d/%d aniso=%f compare=%u minLod=%f maxLod=%f",
            (int)pInfo->magFilter, (int)pInfo->minFilter, (int)pInfo->mipmapMode,
            (int)pInfo->addressModeU, (int)pInfo->addressModeV, (int)pInfo->addressModeW,
            pInfo->maxAnisotropy, (unsigned)pInfo->compareEnable,
            pInfo->minLod, pInfo->maxLod);
    }
    auto real = RealExport<PFN_vkCreateSampler>("vkCreateSampler");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pSampler);
    if (r != VK_SUCCESS) LogLine("vkCreateSampler FAILED: result=%d", (int)r);
    else LogLine("vkCreateSampler OK");
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool) {
    if (pInfo) {
        LogLine("vkCreateDescriptorPool ENTER: maxSets=%u poolSizes=%u flags=0x%X",
            pInfo->maxSets, pInfo->poolSizeCount, (unsigned)pInfo->flags);
    }
    auto real = RealExport<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    std::vector<VkDescriptorPoolSize> swSizes; VkDescriptorPoolCreateInfo swPool;     // [SWRT] AS pool sizes -> storage buffers
    if (pInfo && g_spvCfg.softwareRT && pInfo->pPoolSizes) {
        bool any = false; for (uint32_t i = 0; i < pInfo->poolSizeCount; ++i) if ((int)pInfo->pPoolSizes[i].type == 1000150000) any = true;
        if (any) {
            swSizes.assign(pInfo->pPoolSizes, pInfo->pPoolSizes + pInfo->poolSizeCount);
            for (auto& ps : swSizes) if ((int)ps.type == 1000150000) ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            swPool = *pInfo; swPool.pPoolSizes = swSizes.data(); pInfo = &swPool;
        }
    }
    VkResult r = real(device, pInfo, pAllocator, pDescriptorPool);
    if (r != VK_SUCCESS) LogLine("vkCreateDescriptorPool FAILED: result=%d", (int)r);
    else LogLine("vkCreateDescriptorPool OK");
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pInfo,
    VkDescriptorSet* pDescriptorSets) {
    if (pInfo) LogLine("vkAllocateDescriptorSets ENTER: count=%u", pInfo->descriptorSetCount);
    auto real = RealExport<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pDescriptorSets);
    if (r != VK_SUCCESS) LogLine("vkAllocateDescriptorSets FAILED: result=%d", (int)r);
    else LogLine("vkAllocateDescriptorSets OK");
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkUpdateDescriptorSets(VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* pWrites,
    uint32_t copyCount, const VkCopyDescriptorSet* pCopies) {
    static volatile LONG cnt = 0;
    if (InterlockedIncrement(&cnt) <= 20)
        LogLine("vkUpdateDescriptorSets ENTER: writes=%u copies=%u", writeCount, copyCount);
    auto real = RealExport<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    if (!real) return;
    rtfill::TrackWrites(writeCount, pWrites); // [RTFILL] запоминаем storage-image привязки ДО фильтрации AS-дескрипторов

    // [SWRT] write the software TLAS blob (a region of the AS' backing buffer) as a storage-buffer descriptor
    std::vector<VkWriteDescriptorSet> swWrites; std::vector<std::vector<VkDescriptorBufferInfo>> swInfos;
    if (g_spvCfg.softwareRT && pWrites) {
        bool any = false; for (uint32_t i = 0; i < writeCount; ++i) if ((int)pWrites[i].descriptorType == 1000150000) any = true;
        if (any) {
            swInfos.reserve(writeCount);
            for (uint32_t i = 0; i < writeCount; ++i) {
                VkWriteDescriptorSet w = pWrites[i];
                if ((int)w.descriptorType == 1000150000) {
                    const VkWriteDescriptorSetAccelerationStructureKHR* as = nullptr;
                    for (const VkBaseInStructure* p = reinterpret_cast<const VkBaseInStructure*>(w.pNext); p; p = p->pNext)
                        if ((int)p->sType == 1000150007) as = reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(p);
                    std::vector<VkDescriptorBufferInfo> bi; bool ok = as && as->pAccelerationStructures;
                    for (uint32_t k = 0; ok && k < w.descriptorCount && k < as->accelerationStructureCount; ++k) {
                        swpx::AsRec* rec = IsFake(as->pAccelerationStructures[k]) ? nullptr : swpx::Lookup(as->pAccelerationStructures[k]);
                        if (!rec) { ok = false; break; }
                        bi.push_back({ rec->buf, rec->off, rec->size });
                    }
                    if (!ok) continue;                      // unknown / fake AS: drop the write like before
                    swInfos.push_back(std::move(bi));
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pNext = nullptr; w.pBufferInfo = swInfos.back().data();
                }
                swWrites.push_back(w);
            }
            pWrites = swWrites.data(); writeCount = (uint32_t)swWrites.size();
        }
    }

    // The old driver does not know acceleration-structure descriptors (ray tracing): it treats them as
    // images and dereferences the NULL pImageInfo (crash at READ 0x0 in amdvlk64). Drop such writes, and
    // any write whose mandatory info pointer / destination set is NULL.
    auto shouldDrop = [](const VkWriteDescriptorSet& w) -> bool {
        const int t = static_cast<int>(w.descriptorType);
        if (t == 1000150000 || t == 1000165000) return true;       // ACCELERATION_STRUCTURE_KHR / _NV
        if (!w.dstSet) return true;
        if (w.descriptorCount == 0) return false;
        switch (w.descriptorType) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return w.pImageInfo == nullptr;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return w.pBufferInfo == nullptr;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            return w.pTexelBufferView == nullptr;
        default: return false;
        }
    };
    bool anyDrop = false;
    for (uint32_t i = 0; i < writeCount && pWrites; ++i) if (shouldDrop(pWrites[i])) { anyDrop = true; break; }
    if (!anyDrop) { real(device, writeCount, pWrites, copyCount, pCopies); return; }

    std::vector<VkWriteDescriptorSet> kept;
    kept.reserve(writeCount);
    for (uint32_t i = 0; i < writeCount; ++i) {
        if (shouldDrop(pWrites[i])) {
            char key[64];
            _snprintf_s(key, sizeof(key), _TRUNCATE, "dropdesc:%d", static_cast<int>(pWrites[i].descriptorType));
            if (FirstTime(key))
                LogLine("vkUpdateDescriptorSets: DROPPING writes of descriptor type %d (unsupported by driver / NULL data)",
                    static_cast<int>(pWrites[i].descriptorType));
            continue;
        }
        kept.push_back(pWrites[i]);
    }
    real(device, static_cast<uint32_t>(kept.size()), kept.empty() ? nullptr : kept.data(), copyCount, pCopies);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool) {
    if (pInfo) LogLine("vkCreateCommandPool ENTER: queueFamilyIdx=%u flags=0x%X",
        pInfo->queueFamilyIndex, (unsigned)pInfo->flags);
    auto real = RealExport<PFN_vkCreateCommandPool>("vkCreateCommandPool");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pCommandPool);
    if (r != VK_SUCCESS) LogLine("vkCreateCommandPool FAILED: result=%d", (int)r);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateFence(VkDevice device, const VkFenceCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkFence* pFence) {
    auto real = RealExport<PFN_vkCreateFence>("vkCreateFence");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pFence);
    if (r != VK_SUCCESS) LogLine("vkCreateFence FAILED: result=%d", (int)r);
    return r;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass) {
    static volatile LONG cnt = 0;
    if (InterlockedIncrement(&cnt) <= 5 && pInfo) {
        LogLine("vkCreateRenderPass ENTER: attachments=%u subpasses=%u deps=%u",
            pInfo->attachmentCount, pInfo->subpassCount, pInfo->dependencyCount);
    }
    auto real = RealExport<PFN_vkCreateRenderPass>("vkCreateRenderPass");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pAllocator, pRenderPass);
    if (r != VK_SUCCESS) LogLine("vkCreateRenderPass FAILED: result=%d", (int)r);
    return r;
}



    // -----------------------------------------------------------------------
    // RT stubs (unchanged from v2, except vkCmdBindPipeline/vkDestroyPipeline
    // hooks are removed - they are now C++ exports below).
    // -----------------------------------------------------------------------
    namespace stub {
        VKAPI_ATTR VkResult VKAPI_CALL CreateAS(VkDevice d, const VkAccelerationStructureCreateInfoKHR* ci,
            const VkAllocationCallbacks*, VkAccelerationStructureKHR* p) {
            if (!p) return VK_ERROR_INITIALIZATION_FAILED;
            return swpx::CreateAS(d, ci, p);   // [SWRT] real record backed by the game's buffer (falls back to FAKE if the builder is unavailable)
        }
        VKAPI_ATTR void VKAPI_CALL DestroyAS(VkDevice, VkAccelerationStructureKHR h, const VkAllocationCallbacks*) { if (!IsFake(h)) swpx::DestroyAS(h); }
        VKAPI_ATTR void VKAPI_CALL CmdBuildAS(VkCommandBuffer cb, uint32_t n,
            const VkAccelerationStructureBuildGeometryInfoKHR* infos, const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
            swpx::CmdBuild(cb, n, infos, ranges);   // [SWRT] BVH build as compute work in the game's command buffer
        }
        VKAPI_ATTR void VKAPI_CALL CmdBuildASIndirect(VkCommandBuffer, uint32_t,
            const VkAccelerationStructureBuildGeometryInfoKHR*, const VkDeviceAddress*, const uint32_t*,
            const uint32_t* const*) {}
        VKAPI_ATTR VkResult VKAPI_CALL BuildAS(VkDevice, VkDeferredOperationKHR, uint32_t,
            const VkAccelerationStructureBuildGeometryInfoKHR*, const VkAccelerationStructureBuildRangeInfoKHR* const*) {
            return VK_SUCCESS;
        }
        VKAPI_ATTR VkResult VKAPI_CALL CopyAS(VkDevice, VkDeferredOperationKHR, const VkCopyAccelerationStructureInfoKHR*) { return VK_SUCCESS; }
        VKAPI_ATTR VkResult VKAPI_CALL CopyASToMem(VkDevice, VkDeferredOperationKHR, const VkCopyAccelerationStructureToMemoryInfoKHR*) { return VK_SUCCESS; }
        VKAPI_ATTR VkResult VKAPI_CALL CopyMemToAS(VkDevice, VkDeferredOperationKHR, const VkCopyMemoryToAccelerationStructureInfoKHR*) { return VK_SUCCESS; }
        VKAPI_ATTR VkResult VKAPI_CALL WriteASProps(VkDevice, uint32_t n, const VkAccelerationStructureKHR* as,
            VkQueryType, size_t dataSize, void* pData, size_t stride) {
            if (pData && dataSize) memset(pData, 0, dataSize);
            if (pData && as && stride >= 8) for (uint32_t i = 0; i < n && (size_t)(i + 1) * stride <= dataSize; ++i) {
                swpx::AsRec* r = IsFake(as[i]) ? nullptr : swpx::Lookup(as[i]);
                uint64_t v = r ? (r->used ? r->used : r->size) : 0; memcpy(static_cast<uint8_t*>(pData) + i * stride, &v, 8);
            }
            return VK_SUCCESS;
        }
        VKAPI_ATTR void VKAPI_CALL CmdCopyAS(VkCommandBuffer cb, const VkCopyAccelerationStructureInfoKHR* ci) { swpx::CmdCopy(cb, ci); }
        VKAPI_ATTR void VKAPI_CALL CmdCopyASToMem(VkCommandBuffer, const VkCopyAccelerationStructureToMemoryInfoKHR*) {}
        VKAPI_ATTR void VKAPI_CALL CmdCopyMemToAS(VkCommandBuffer, const VkCopyMemoryToAccelerationStructureInfoKHR*) {}
        VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetASAddr(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* ai) {
            if (ai && !IsFake(ai->accelerationStructure)) { VkDeviceAddress a = swpx::GetAddr(ai->accelerationStructure); if (a) return a; }   // [SWRT] real address
            static volatile LONG64 n = 0;
            return 0x200000000ull + static_cast<uint64_t>(InterlockedIncrement64(&n)) * 256ull;
        }
        VKAPI_ATTR void VKAPI_CALL CmdWriteASProps(VkCommandBuffer, uint32_t n, const VkAccelerationStructureKHR* as,
            VkQueryType, VkQueryPool pool, uint32_t first) { swpx::CmdWriteProps(n, as, pool, first); }
        VKAPI_ATTR void VKAPI_CALL GetASCompat(VkDevice, const VkAccelerationStructureVersionInfoKHR*,
            VkAccelerationStructureCompatibilityKHR* p) {
            if (p) *p = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR;
        }
        VKAPI_ATTR void VKAPI_CALL GetASBuildSizes(VkDevice, VkAccelerationStructureBuildTypeKHR,
            const VkAccelerationStructureBuildGeometryInfoKHR* bi, const uint32_t* maxPrims,
            VkAccelerationStructureBuildSizesInfoKHR* s) {
            if (!s) return;
            if (swpx::EnsureInit(g_lastDevice)) { swpx::GetBuildSizes(bi, maxPrims, s); return; }   // [SWRT] sizes of OUR BVH format (must match what CmdBuild writes)
            s->accelerationStructureSize = 256; s->updateScratchSize = 256; s->buildScratchSize = 256;   // old stub behaviour
        }

        // [SWRT] acceleration-structure query pools: the driver has no such query types, emulate them on the CPU.
        VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPoolHook(VkDevice d, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkQueryPool* p) {
            if (swpx::MaybeCreatePool(ci, p)) return VK_SUCCESS;
            auto real = swpx::RF<PFN_vkCreateQueryPool>(d, "vkCreateQueryPool");
            return real ? real(d, ci, a, p) : VK_ERROR_INITIALIZATION_FAILED;
        }
        VKAPI_ATTR void VKAPI_CALL DestroyQueryPoolHook(VkDevice d, VkQueryPool q, const VkAllocationCallbacks* a) {
            if (swpx::MaybeDestroyPool(q)) return;
            auto real = swpx::RF<PFN_vkDestroyQueryPool>(d, "vkDestroyQueryPool"); if (real) real(d, q, a);
        }
        VKAPI_ATTR void VKAPI_CALL CmdResetQueryPoolHook(VkCommandBuffer cb, VkQueryPool q, uint32_t first, uint32_t count) {
            if (swpx::FindPool(q)) return;
            auto real = swpx::RF<PFN_vkCmdResetQueryPool>(g_lastDevice, "vkCmdResetQueryPool"); if (real) real(cb, q, first, count);
        }
        VKAPI_ATTR void VKAPI_CALL ResetQueryPoolHook(VkDevice d, VkQueryPool q, uint32_t first, uint32_t count) {
            if (swpx::FindPool(q)) return;
            auto real = swpx::RF<PFN_vkResetQueryPool>(d, "vkResetQueryPool"); if (!real) real = swpx::RF<PFN_vkResetQueryPool>(d, "vkResetQueryPoolEXT"); if (real) real(d, q, first, count);
        }
        VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResultsHook(VkDevice d, VkQueryPool q, uint32_t first, uint32_t count, size_t size, void* data, VkDeviceSize stride, VkQueryResultFlags flags) {
            if (swpx::FindPool(q)) return swpx::GetResults(q, first, count, size, data, stride, flags);
            auto real = swpx::RF<PFN_vkGetQueryPoolResults>(d, "vkGetQueryPoolResults");
            return real ? real(d, q, first, count, size, data, stride, flags) : VK_ERROR_INITIALIZATION_FAILED;
        }
        VKAPI_ATTR void VKAPI_CALL CmdCopyQueryPoolResultsHook(VkCommandBuffer cb, VkQueryPool q, uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize off, VkDeviceSize stride, VkQueryResultFlags flags) {
            if (swpx::FindPool(q)) { swpx::CmdCopyResults(cb, q, first, count, dst, off, stride, flags); return; }
            auto real = swpx::RF<PFN_vkCmdCopyQueryPoolResults>(g_lastDevice, "vkCmdCopyQueryPoolResults"); if (real) real(cb, q, first, count, dst, off, stride, flags);
        }
        VKAPI_ATTR void VKAPI_CALL CmdTraceRays(VkCommandBuffer cb, const VkStridedDeviceAddressRegionKHR*,
            const VkStridedDeviceAddressRegionKHR*, const VkStridedDeviceAddressRegionKHR*,
            const VkStridedDeviceAddressRegionKHR*, uint32_t w, uint32_t h, uint32_t) {
            rtfill::OnTraceRays(cb, w, h); // [RTFILL] вместо пустого no-op
        }
        VKAPI_ATTR void VKAPI_CALL CmdTraceRaysIndirect(VkCommandBuffer cb, const VkStridedDeviceAddressRegionKHR*,
            const VkStridedDeviceAddressRegionKHR*, const VkStridedDeviceAddressRegionKHR*,
            const VkStridedDeviceAddressRegionKHR*, VkDeviceAddress) {
            rtfill::OnTraceRays(cb, 0, 0); // [RTFILL]
        }
        VKAPI_ATTR void VKAPI_CALL CmdTraceRaysIndirect2(VkCommandBuffer cb, VkDeviceAddress) {
            rtfill::OnTraceRays(cb, 0, 0); // [RTFILL]
        }
        VKAPI_ATTR VkResult VKAPI_CALL CreateRTPipelines(VkDevice, VkDeferredOperationKHR, VkPipelineCache,
            uint32_t count, const VkRayTracingPipelineCreateInfoKHR*, const VkAllocationCallbacks*, VkPipeline* p) {
            if (p) for (uint32_t i = 0; i < count; ++i) p[i] = MakeFake<VkPipeline>();
            return VK_SUCCESS;
        }
        VKAPI_ATTR VkResult VKAPI_CALL GetRTHandles(VkDevice, VkPipeline, uint32_t, uint32_t,
            size_t dataSize, void* pData) {
            if (pData && dataSize) memset(pData, 0, dataSize);
            return VK_SUCCESS;
        }
        VKAPI_ATTR VkDeviceSize VKAPI_CALL GetRTStack(VkDevice, VkPipeline, uint32_t, VkShaderGroupShaderKHR) { return 0; }
        VKAPI_ATTR void VKAPI_CALL CmdSetRTStack(VkCommandBuffer, uint32_t) {}

        VKAPI_ATTR VkResult VKAPI_CALL CreateDefOp(VkDevice, const VkAllocationCallbacks*, VkDeferredOperationKHR* p) {
            if (!p) return VK_ERROR_INITIALIZATION_FAILED;
            *p = MakeFake<VkDeferredOperationKHR>();
            return VK_SUCCESS;
        }
        VKAPI_ATTR void VKAPI_CALL DestroyDefOp(VkDevice, VkDeferredOperationKHR, const VkAllocationCallbacks*) {}
        VKAPI_ATTR uint32_t VKAPI_CALL DefOpConcurrency(VkDevice, VkDeferredOperationKHR) { return 1; }
        VKAPI_ATTR VkResult VKAPI_CALL DefOpResult(VkDevice, VkDeferredOperationKHR) { return VK_SUCCESS; }
        VKAPI_ATTR VkResult VKAPI_CALL DefOpJoin(VkDevice, VkDeferredOperationKHR) { return VK_SUCCESS; }

        struct Entry { const char* name; PFN_vkVoidFunction fn; };
#define VKSTUB(n, f) { n, reinterpret_cast<PFN_vkVoidFunction>(&f) }

        // Always-active: dynamic rendering emulation, memory-requirements stubs,
        // and the pipeline bind/destroy hooks (which used to be RT-only).
        const Entry kAlwaysTable[] = {
            VKSTUB("vkQueueSubmit", vkQueueSubmit),
            VKSTUB("vkQueueWaitIdle", vkQueueWaitIdle),
            VKSTUB("vkWaitForFences", vkWaitForFences),
            VKSTUB("vkCreateGraphicsPipelines", vkCreateGraphicsPipelines),
			VKSTUB("vkCreateComputePipelines",  vkCreateComputePipelines),
            VKSTUB("vkDestroyPipeline",         vkDestroyPipeline),
            VKSTUB("vkCmdBindPipeline",         vkCmdBindPipeline),
            // draws / dispatches: a game that loads functions through vkGet*ProcAddr would otherwise call the
            // driver directly and crash on a FAKE or NULL pipeline
            VKSTUB("vkCmdDraw",                         vkCmdDraw),
            VKSTUB("vkCmdDrawIndexed",                  vkCmdDrawIndexed),
            VKSTUB("vkCmdDrawIndirect",                 vkCmdDrawIndirect),
            VKSTUB("vkCmdDrawIndexedIndirect",          vkCmdDrawIndexedIndirect),
            VKSTUB("vkCmdDrawIndirectCount",            vkCmdDrawIndirectCount),
            VKSTUB("vkCmdDrawIndirectCountKHR",         vkCmdDrawIndirectCount),
            VKSTUB("vkCmdDrawIndexedIndirectCount",     vkCmdDrawIndexedIndirectCount),
            VKSTUB("vkCmdDrawIndexedIndirectCountKHR",  vkCmdDrawIndexedIndirectCount),
            VKSTUB("vkCmdDispatch",                     vkCmdDispatch),
            VKSTUB("vkCmdDispatchIndirect",             vkCmdDispatchIndirect),
            VKSTUB("vkCmdBeginRendering",       vkCmdBeginRendering),
            VKSTUB("vkCmdBeginRenderingKHR",    vkCmdBeginRenderingKHR),
            VKSTUB("vkCmdEndRendering",         vkCmdEndRendering),
            VKSTUB("vkCmdEndRenderingKHR",      vkCmdEndRenderingKHR),
            VKSTUB("vkCreateImageView",         vkCreateImageView),
            VKSTUB("vkDestroyImageView",        vkDestroyImageView),
            VKSTUB("vkAllocateCommandBuffers",  vkAllocateCommandBuffers),
            VKSTUB("vkFreeCommandBuffers",      vkFreeCommandBuffers),
            VKSTUB("vkGetDeviceBufferMemoryRequirements",         vkGetDeviceBufferMemoryRequirements),
            VKSTUB("vkGetDeviceBufferMemoryRequirementsKHR", vkGetDeviceBufferMemoryRequirements),
            VKSTUB("vkGetDeviceImageMemoryRequirements",          vkGetDeviceImageMemoryRequirements),
            VKSTUB("vkGetDeviceImageMemoryRequirementsKHR", vkGetDeviceImageMemoryRequirements),
            VKSTUB("vkGetDeviceImageSparseMemoryRequirements",    vkGetDeviceImageSparseMemoryRequirements),
            VKSTUB("vkGetDeviceImageSparseMemoryRequirementsKHR", vkGetDeviceImageSparseMemoryRequirements),
			            // v4: sync2 stubs (always active - translate to sync1)
            VKSTUB("vkQueueSubmit2",           vkQueueSubmit2),
            VKSTUB("vkQueueSubmit2KHR",        vkQueueSubmit2KHR),
            VKSTUB("vkCmdPipelineBarrier2",    vkCmdPipelineBarrier2),
            VKSTUB("vkCmdPipelineBarrier2KHR", vkCmdPipelineBarrier2KHR),
            VKSTUB("vkCmdWriteTimestamp2",     vkCmdWriteTimestamp2),
            VKSTUB("vkCmdWriteTimestamp2KHR",  vkCmdWriteTimestamp2KHR),
        VKSTUB("vkCmdCopyBuffer2", vkCmdCopyBuffer2),           VKSTUB("vkCmdCopyBuffer2KHR", vkCmdCopyBuffer2KHR),
        VKSTUB("vkCmdCopyImage2", vkCmdCopyImage2),             VKSTUB("vkCmdCopyImage2KHR", vkCmdCopyImage2KHR),
        VKSTUB("vkCmdCopyBufferToImage2", vkCmdCopyBufferToImage2), VKSTUB("vkCmdCopyBufferToImage2KHR", vkCmdCopyBufferToImage2KHR),
        VKSTUB("vkCmdCopyImageToBuffer2", vkCmdCopyImageToBuffer2), VKSTUB("vkCmdCopyImageToBuffer2KHR", vkCmdCopyImageToBuffer2KHR),
        VKSTUB("vkCmdBlitImage2", vkCmdBlitImage2),             VKSTUB("vkCmdBlitImage2KHR", vkCmdBlitImage2KHR),
        VKSTUB("vkCmdResolveImage2", vkCmdResolveImage2),       VKSTUB("vkCmdResolveImage2KHR", vkCmdResolveImage2KHR),
        VKSTUB("vkCmdSetEvent2", vkCmdSetEvent2),               VKSTUB("vkCmdSetEvent2KHR", vkCmdSetEvent2KHR),
        VKSTUB("vkCmdResetEvent2", vkCmdResetEvent2),           VKSTUB("vkCmdResetEvent2KHR", vkCmdResetEvent2KHR),
        VKSTUB("vkCmdWaitEvents2", vkCmdWaitEvents2),           VKSTUB("vkCmdWaitEvents2KHR", vkCmdWaitEvents2KHR),
        VKSTUB("vkCmdBindVertexBuffers2", vkCmdBindVertexBuffers2),
        VKSTUB("vkCmdPipelineBarrier", vkCmdPipelineBarrier),
        VKSTUB("vkCreatePrivateDataSlot", vkCreatePrivateDataSlot), VKSTUB("vkDestroyPrivateDataSlot", vkDestroyPrivateDataSlot),
        VKSTUB("vkSetPrivateData", vkSetPrivateData),           VKSTUB("vkGetPrivateData", vkGetPrivateData),
            // v4: diagnostic wrappers
            VKSTUB("vkCreateSwapchainKHR",     vkCreateSwapchainKHR),
            VKSTUB("vkCreateShaderModule",     vkCreateShaderModule),
			VKSTUB("vkDestroyShaderModule",    vkDestroyShaderModule),
            VKSTUB("vkQueuePresentKHR",        vkQueuePresentKHR),
			            // v5: диагностика
            VKSTUB("vkCreateInstance",                        vkCreateInstance),
            VKSTUB("vkCreateWin32SurfaceKHR",                 vkCreateWin32SurfaceKHR),
            VKSTUB("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", vkGetPhysicalDeviceSurfaceCapabilitiesKHR),
            VKSTUB("vkGetPhysicalDeviceSurfaceFormatsKHR",    vkGetPhysicalDeviceSurfaceFormatsKHR),
            VKSTUB("vkGetPhysicalDeviceSurfacePresentModesKHR", vkGetPhysicalDeviceSurfacePresentModesKHR),
            VKSTUB("vkCreateImage",                           vkCreateImage),
            VKSTUB("vkGetImageMemoryRequirements",            vkGetImageMemoryRequirements),
            VKSTUB("vkGetImageMemoryRequirements2",           vkGetImageMemoryRequirements2),
            VKSTUB("vkAllocateMemory",                        vkAllocateMemory),
            VKSTUB("vkBindImageMemory",                       vkBindImageMemory),
            VKSTUB("vkBindImageMemory2",                      vkBindImageMemory2),
            VKSTUB("vkCreateBuffer",                          vkCreateBuffer),
            VKSTUB("vkGetBufferMemoryRequirements",           vkGetBufferMemoryRequirements),
            VKSTUB("vkGetBufferMemoryRequirements2",          vkGetBufferMemoryRequirements2),
            VKSTUB("vkBindBufferMemory",                      vkBindBufferMemory),
            VKSTUB("vkBindBufferMemory2",                     vkBindBufferMemory2),
            VKSTUB("vkCreateDescriptorSetLayout",             vkCreateDescriptorSetLayout),
            VKSTUB("vkCreatePipelineLayout",                  vkCreatePipelineLayout),
			            // v6
            VKSTUB("vkCreateSampler",            vkCreateSampler),
            VKSTUB("vkCreateDescriptorPool",     vkCreateDescriptorPool),
            VKSTUB("vkAllocateDescriptorSets",   vkAllocateDescriptorSets),
            VKSTUB("vkUpdateDescriptorSets",     vkUpdateDescriptorSets),
            VKSTUB("vkCreateCommandPool",        vkCreateCommandPool),
            VKSTUB("vkCreateFence",              vkCreateFence),
            VKSTUB("vkCreateRenderPass",         vkCreateRenderPass),
			
        };

        // RT-only stubs (served when g_rtSpoof != 0).
        const Entry kTable[] = {
            VKSTUB("vkCreateQueryPool",                 CreateQueryPoolHook),            // [SWRT]
            VKSTUB("vkDestroyQueryPool",                DestroyQueryPoolHook),
            VKSTUB("vkCmdResetQueryPool",               CmdResetQueryPoolHook),
            VKSTUB("vkResetQueryPool",                  ResetQueryPoolHook),
            VKSTUB("vkGetQueryPoolResults",             GetQueryPoolResultsHook),
            VKSTUB("vkCmdCopyQueryPoolResults",         CmdCopyQueryPoolResultsHook),
            VKSTUB("vkCreateAccelerationStructureKHR", CreateAS),
            VKSTUB("vkDestroyAccelerationStructureKHR", DestroyAS),
            VKSTUB("vkCmdBuildAccelerationStructuresKHR", CmdBuildAS),
            VKSTUB("vkCmdBuildAccelerationStructuresIndirectKHR", CmdBuildASIndirect),
            VKSTUB("vkBuildAccelerationStructuresKHR", BuildAS),
            VKSTUB("vkCopyAccelerationStructureKHR", CopyAS),
            VKSTUB("vkCopyAccelerationStructureToMemoryKHR", CopyASToMem),
            VKSTUB("vkCopyMemoryToAccelerationStructureKHR", CopyMemToAS),
            VKSTUB("vkWriteAccelerationStructuresPropertiesKHR", WriteASProps),
            VKSTUB("vkCmdCopyAccelerationStructureKHR", CmdCopyAS),
            VKSTUB("vkCmdCopyAccelerationStructureToMemoryKHR", CmdCopyASToMem),
            VKSTUB("vkCmdCopyMemoryToAccelerationStructureKHR", CmdCopyMemToAS),
            VKSTUB("vkGetAccelerationStructureDeviceAddressKHR", GetASAddr),
            VKSTUB("vkCmdWriteAccelerationStructuresPropertiesKHR", CmdWriteASProps),
            VKSTUB("vkGetDeviceAccelerationStructureCompatibilityKHR", GetASCompat),
            VKSTUB("vkGetAccelerationStructureBuildSizesKHR", GetASBuildSizes),
            VKSTUB("vkCmdTraceRaysKHR", CmdTraceRays),
            VKSTUB("vkCmdTraceRaysIndirectKHR", CmdTraceRaysIndirect),
            VKSTUB("vkCmdTraceRaysIndirect2KHR", CmdTraceRaysIndirect2),
            VKSTUB("vkCreateRayTracingPipelinesKHR", CreateRTPipelines),
            VKSTUB("vkGetRayTracingShaderGroupHandlesKHR", GetRTHandles),
            VKSTUB("vkGetRayTracingCaptureReplayShaderGroupHandlesKHR", GetRTHandles),
            VKSTUB("vkGetRayTracingShaderGroupStackSizeKHR", GetRTStack),
            VKSTUB("vkCmdSetRayTracingPipelineStackSizeKHR", CmdSetRTStack),
            VKSTUB("vkCreateDeferredOperationKHR", CreateDefOp),
            VKSTUB("vkDestroyDeferredOperationKHR", DestroyDefOp),
            VKSTUB("vkGetDeferredOperationMaxConcurrencyKHR", DefOpConcurrency),
            VKSTUB("vkGetDeferredOperationResultKHR", DefOpResult),
            VKSTUB("vkDeferredOperationJoinKHR", DefOpJoin),
        };
#undef VKSTUB

        PFN_vkVoidFunction Find(const char* name) {
            if (!name) return nullptr;
            for (const Entry& e : kAlwaysTable) if (strcmp(e.name, name) == 0) return e.fn;
            if (!g_rtSpoof) return nullptr;
            for (const Entry& e : kTable) if (strcmp(e.name, name) == 0) return e.fn;
            return nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic-rendering emulation hooks (definitions)
// ---------------------------------------------------------------------------

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    static volatile LONG enterCnt = 0;
    if (InterlockedIncrement(&enterCnt) <= 30) {
        LogLine("vkCreateGraphicsPipelines ENTER: count=%u", count);
        if (pCreateInfos) for (uint32_t i = 0; i < count && i < 4; ++i) {
            LogLine("  pipeline[%u]: renderPass=%p layout=%p stages=%u",
                i, pCreateInfos[i].renderPass, pCreateInfos[i].layout, pCreateInfos[i].stageCount);
        }
    }
    auto real = RealExport<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
    if (!real) {
        LogLine("vkCreateGraphicsPipelines: RealExport returned NULL -> FAKE pipelines");
        if (pPipelines) for (uint32_t i = 0; i < count; ++i) pPipelines[i] = MakeFake<VkPipeline>();
        return VK_SUCCESS;
    }
    if (!pCreateInfos || count == 0) {
        if (pPipelines && count) for (uint32_t i = 0; i < count; ++i) pPipelines[i] = MakeFake<VkPipeline>();
        return VK_SUCCESS;
    }
    cache = pcache::Get(device, cache);   // persistent proxy cache when the game passes none

    // Pipelines that use a FAKE (denied / failed) shader module become FAKE pipelines;
    // the remaining pipelines of the batch are still compiled by the driver.
    {
        std::vector<char> isFakePipe(count, 0);
        bool anyFake = false;
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < pCreateInfos[i].stageCount && pCreateInfos[i].pStages; ++j)
                if (IsFake(pCreateInfos[i].pStages[j].module)) { isFakePipe[i] = 1; anyFake = true; break; }
        }
        if (anyFake) {
            static volatile LONG fakeBatches = 0;
            if (InterlockedIncrement(&fakeBatches) <= 40)
                LogLine("vkCreateGraphicsPipelines: batch of %u contains FAKE-shader pipelines", count);
            std::vector<VkGraphicsPipelineCreateInfo> sub;
            for (uint32_t i = 0; i < count; ++i) if (!isFakePipe[i]) sub.push_back(pCreateInfos[i]);
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            VkResult sr = VK_SUCCESS;
            if (!sub.empty())
                sr = vkCreateGraphicsPipelines(device, cache, static_cast<uint32_t>(sub.size()), sub.data(), pAllocator, subOut.data());
            size_t k = 0;
            for (uint32_t i = 0; i < count; ++i)
                if (pPipelines) pPipelines[i] = isFakePipe[i] ? MakeFake<VkPipeline>() : subOut[k++];
            (void)sr;
            return VK_SUCCESS;
        }
    }
    // [HANGFIX] quarantined (hung before / too heavy) pipelines are never sent to the driver
    {
        std::vector<char> q(count, 0);
        bool anyQ = false;
        for (uint32_t i = 0; i < count; ++i) {
            const char* why = nullptr;
            if (pCreateInfos[i].pStages && PipelineQuarantined(pCreateInfos[i].pStages, pCreateInfos[i].stageCount, &why)) {
                q[i] = 1; anyQ = true;
                static volatile LONG qn = 0;
                const LONG t = InterlockedIncrement(&qn);
                if (t <= 20 || t % 50 == 0) LogLine("QUARANTINE graphics pipeline -> FAKE (%s) total=%ld", why ? why : "?", (long)t);
            }
        }
        if (anyQ) {
            std::vector<VkGraphicsPipelineCreateInfo> sub;
            std::vector<uint32_t> subIdx;
            for (uint32_t i = 0; i < count; ++i) {
                if (q[i]) { if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>(); }
                else { sub.push_back(pCreateInfos[i]); subIdx.push_back(i); }
            }
            if (sub.empty()) return VK_SUCCESS;
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            vkCreateGraphicsPipelines(device, cache, static_cast<uint32_t>(sub.size()), sub.data(), pAllocator, subOut.data());
            for (size_t k = 0; k < sub.size(); ++k) if (pPipelines) pPipelines[subIdx[k]] = subOut[k];
            return VK_SUCCESS;
        }
    }
    // Fast-fail: GCN2 AMDVLK spends ~5s then returns -13 on shaders with GroupNonUniform
    // + Float16/Int16/cap45. Don't call the driver — FAKE immediately so menu can load.
    {
        std::vector<char> toxic(count, 0);
        bool anyToxic = false;
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < pCreateInfos[i].stageCount && pCreateInfos[i].pStages; ++j) {
                const std::string sig = SigOfModule(pCreateInfos[i].pStages[j].module, static_cast<uint32_t>(pCreateInfos[i].pStages[j].stage));
                if (SigToxic(sig)) { toxic[i] = 1; anyToxic = true; break; }
            }
        }
        if (anyToxic) {
            static volatile LONG toxN = 0;
            std::vector<VkGraphicsPipelineCreateInfo> sub;
            std::vector<uint32_t> subIdx;
            for (uint32_t i = 0; i < count; ++i) {
                if (toxic[i]) {
                    if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>();
                    LONG t = InterlockedIncrement(&toxN);
                    if (t <= 15 || t % 50 == 0)
                        LogLine("FAST-FAKE graphics pipeline (learned toxic signature) total=%ld", (long)t);
                } else {
                    sub.push_back(pCreateInfos[i]);
                    subIdx.push_back(i);
                }
            }
            if (sub.empty()) return VK_SUCCESS;
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            VkResult sr = vkCreateGraphicsPipelines(device, cache, static_cast<uint32_t>(sub.size()),
                sub.data(), pAllocator, subOut.data());
            for (size_t k = 0; k < sub.size(); ++k)
                if (pPipelines) pPipelines[subIdx[k]] = subOut[k];
            (void)sr;
            return VK_SUCCESS;
        }
    }

    // =====================================================================
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = 0; j < pCreateInfos[i].stageCount && j < 6; ++j) {
            char d[180];
            DescribeModule(pCreateInfos[i].pStages[j].module, d, sizeof(d));
            Crumb("vkCreateGraphicsPipelines[%u/%u] stage%u=0x%X %s", i, count, j,
                (unsigned)pCreateInfos[i].pStages[j].stage, d);
            SetJob("vkCreateGraphicsPipelines[%u/%u] stage%u=0x%X %s", i, count, j,
                (unsigned)pCreateInfos[i].pStages[j].stage, d);
        }
    }

    std::vector<VkGraphicsPipelineCreateInfo> mods(count);
    for (uint32_t i = 0; i < count; ++i) {
        mods[i] = pCreateInfos[i];
        if (mods[i].renderPass != VK_NULL_HANDLE) continue;

        const VkBaseInStructure* n = reinterpret_cast<const VkBaseInStructure*>(mods[i].pNext);
        const VkPipelineRenderingCreateInfo* ri = nullptr;
        while (n) {
            if (static_cast<uint32_t>(n->sType) == dynrend::ST_PIPELINE_RENDERING_CREATE_INFO) {
                ri = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(n);
                break;
            }
            n = n->pNext;
        }
        if (!ri) continue;

        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        if (mods[i].pMultisampleState) samples = mods[i].pMultisampleState->rasterizationSamples;

        dynrend::RPKey k;
        for (uint32_t j = 0; j < ri->colorAttachmentCount; ++j) {
            k.colorFormats.push_back(ri->pColorAttachmentFormats[j]);
            k.colorLoadOps.push_back(VK_ATTACHMENT_LOAD_OP_LOAD);
        k.colorLayouts.push_back(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
        k.depthFormat = ri->depthAttachmentFormat;
        k.stencilFormat = ri->stencilAttachmentFormat;
        k.samples = samples;
        k.viewMask = ri->viewMask;
        k.depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        k.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

        VkRenderPass rp = dynrend::CreateOrGetRenderPass(device, k);
        if (rp == VK_NULL_HANDLE) continue;
        mods[i].renderPass = rp;
        mods[i].subpass = 0;
        LogLine("dynrend: pipeline[%u] renderPass substituted (colors=%u depth=0x%X stencil=0x%X samples=%u)",
            i, static_cast<unsigned>(k.colorFormats.size()), k.depthFormat, k.stencilFormat,
            static_cast<unsigned>(samples));
    }

    VkResult r = VK_ERROR_UNKNOWN;
    {
        // [HANGFIX] the driver works on private copies so the call can be abandoned if it never returns
        auto outP  = std::make_shared<std::vector<VkPipeline>>(count, VK_NULL_HANDLE);
        auto resP  = std::make_shared<VkResult>(VK_ERROR_UNKNOWN);
        auto modsP = std::make_shared<std::vector<VkGraphicsPipelineCreateInfo>>(mods);
        bool timedOut = false;
        const bool callOk = DriverCallTimed([=]() { *resP = real(device, cache, count, modsP->data(), pAllocator, outP->data()); }, timedOut);
        if (callOk) {
            r = *resP;
            if (pPipelines) for (uint32_t i = 0; i < count; ++i) pPipelines[i] = (*outP)[i];
            if (r == VK_SUCCESS) pcache::Dirty();
        } else if (timedOut) {
            LogLine("vkCreateGraphicsPipelines TIMED OUT (count=%u) -> FAKE pipelines", count);
            for (uint32_t i = 0; i < count; ++i) if (pCreateInfos[i].pStages) PipelineMarkHung(pCreateInfos[i].pStages, pCreateInfos[i].stageCount);
            r = VK_ERROR_UNKNOWN;
        } else {
            LogLine("CAUGHT driver crash in vkCreateGraphicsPipelines (count=%u) -> FAKE pipelines", count);
            r = VK_ERROR_UNKNOWN;
        }
    }
    
    // Если реальный драйвер по каким-то причинам все равно вернул ошибку - подсовываем фейк
    if (r != VK_SUCCESS) {
        LogLine("vkCreateGraphicsPipelines FAILED: result=%d, creating FAKE pipelines", (int)r);
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < pCreateInfos[i].stageCount && pCreateInfos[i].pStages; ++j)
                SigRecord(SigOfModule(pCreateInfos[i].pStages[j].module, static_cast<uint32_t>(pCreateInfos[i].pStages[j].stage)), false);
        static volatile LONG gpFails = 0;
        const LONG gf = InterlockedIncrement(&gpFails);
        if (gf <= 5) for (uint32_t i = 0; i < count; ++i) DescribeFailedGraphicsPipeline(pCreateInfos[i], gf * 10 + i);
        // Last resort: retry with texel fetches cut out. Skip when driver already spent
        // ~5s and returned -13 (VK_ERROR_UNKNOWN) — retry almost never helps and freezes menu load.
        bool recovered = false;
        if (g_spvCfg.tryVariants && count == 1 && pPipelines && r != static_cast<VkResult>(-13)) {
            std::vector<VkPipelineShaderStageCreateInfo> stages(mods[0].pStages, mods[0].pStages + mods[0].stageCount);
            std::vector<VkShaderModule> temps;
            auto createSM = RealExport<PFN_vkCreateShaderModule>("vkCreateShaderModule");
            for (uint32_t j = 0; j < stages.size() && createSM; ++j) {
                SpvInfo si;
                if (IsFake(stages[j].module) || !GetSpvInfo(stages[j].module, si) || !si.code || si.code->empty()) continue;
                std::vector<uint32_t> var;
                std::string verr;
                if (!spvrw::StripImageFetch(si.code->data(), si.code->size(), var, verr)) continue;
                if (g_spvCfg.patchVersion && var.size() > 1 && var[1] > 0x00010500u) var[1] = 0x00010500u;
                VkShaderModuleCreateInfo mci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                mci.codeSize = var.size() * 4;
                mci.pCode = var.data();
                VkShaderModule m = VK_NULL_HANDLE;
                VkResult mr = VK_ERROR_UNKNOWN;
                DriverCall([&]() { mr = createSM(device, &mci, pAllocator, &m); });
                if (mr == VK_SUCCESS && m) { stages[j].module = m; temps.push_back(m); }
            }
            if (!temps.empty()) {
                VkGraphicsPipelineCreateInfo ci2 = mods[0];
                ci2.pStages = stages.data();
                VkPipeline np = VK_NULL_HANDLE;
                VkResult rr = VK_ERROR_UNKNOWN;
                SetJob("vkCreateGraphicsPipelines RETRY without texel fetch");
                const bool ok = DriverCall([&]() { rr = real(device, cache, 1, &ci2, pAllocator, &np); });
                if (ok && rr == VK_SUCCESS && np) {
                    pPipelines[0] = np;
                    recovered = true;
                    LogLine("RETRY WORKED: pipeline accepted after removing texel fetches from %zu shader(s) (their fetched data is lost)", temps.size());
                } else {
                    LogLine("RETRY failed (%s, result=%d): pipeline stays FAKE", ok ? "rejected" : "crash", (int)rr);
                }
                auto destroySM = RealExport<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
                if (destroySM) for (VkShaderModule m : temps) destroySM(device, m, pAllocator);
            }
        }
        if (!recovered) {
            for (uint32_t i = 0; i < count; ++i) {
                if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>();
            }
        }
        r = VK_SUCCESS; 
    }

    if ((r == VK_SUCCESS || r == VK_PIPELINE_COMPILE_REQUIRED_EXT) && pPipelines) {
        for (uint32_t i = 0; i < count; ++i) {
            if (!pPipelines[i]) continue;
            
            if (IsFake(pPipelines[i])) continue; 
            for (uint32_t j = 0; j < pCreateInfos[i].stageCount && pCreateInfos[i].pStages; ++j)
                SigRecord(SigOfModule(pCreateInfos[i].pStages[j].module, static_cast<uint32_t>(pCreateInfos[i].pStages[j].stage)), true);

            const VkBaseInStructure* n = reinterpret_cast<const VkBaseInStructure*>(pCreateInfos[i].pNext);
            while (n) {
                if (static_cast<uint32_t>(n->sType) == dynrend::ST_PIPELINE_RENDERING_CREATE_INFO) {
                    const auto* ri = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(n);
                    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
                    if (pCreateInfos[i].pMultisampleState)
                        samples = pCreateInfos[i].pMultisampleState->rasterizationSamples;
                    dynrend::RegisterPipeline(pPipelines[i], device, ri, samples);
                    break;
                }
                n = n->pNext;
            }
        }
    }
    return r;
}


extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    auto real = RealExport<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pCreateInfos || createInfoCount == 0)
        return real(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    pipelineCache = pcache::Get(device, pipelineCache);   // persistent proxy cache when the game passes none

    {
        std::vector<char> isFakePipe(createInfoCount, 0);
        bool anyFake = false;
        for (uint32_t i = 0; i < createInfoCount; ++i)
            if (IsFake(pCreateInfos[i].stage.module)) { isFakePipe[i] = 1; anyFake = true; }
        if (anyFake) {
            static volatile LONG fakeBatches = 0;
            if (InterlockedIncrement(&fakeBatches) <= 40)
                LogLine("vkCreateComputePipelines: batch of %u contains FAKE-shader pipelines", createInfoCount);
            std::vector<VkComputePipelineCreateInfo> sub;
            for (uint32_t i = 0; i < createInfoCount; ++i) if (!isFakePipe[i]) sub.push_back(pCreateInfos[i]);
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            if (!sub.empty())
                vkCreateComputePipelines(device, pipelineCache, static_cast<uint32_t>(sub.size()), sub.data(), pAllocator, subOut.data());
            size_t k = 0;
            for (uint32_t i = 0; i < createInfoCount; ++i)
                if (pPipelines) pPipelines[i] = isFakePipe[i] ? MakeFake<VkPipeline>() : subOut[k++];
            return VK_SUCCESS;
        }
    }
    // [HANGFIX] quarantined (hung before / too heavy) compute pipelines are never sent to the driver
    {
        std::vector<char> q(createInfoCount, 0);
        bool anyQ = false;
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            const char* why = nullptr;
            if (PipelineQuarantined(&pCreateInfos[i].stage, 1, &why)) {
                q[i] = 1; anyQ = true;
                static volatile LONG qn = 0;
                const LONG t = InterlockedIncrement(&qn);
                if (t <= 20 || t % 50 == 0) LogLine("QUARANTINE compute pipeline -> FAKE (%s) total=%ld", why ? why : "?", (long)t);
            }
        }
        if (anyQ) {
            std::vector<VkComputePipelineCreateInfo> sub;
            std::vector<uint32_t> subIdx;
            for (uint32_t i = 0; i < createInfoCount; ++i) {
                if (q[i]) { if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>(); }
                else { sub.push_back(pCreateInfos[i]); subIdx.push_back(i); }
            }
            if (sub.empty()) return VK_SUCCESS;
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            vkCreateComputePipelines(device, pipelineCache, static_cast<uint32_t>(sub.size()), sub.data(), pAllocator, subOut.data());
            for (size_t k = 0; k < sub.size(); ++k) if (pPipelines) pPipelines[subIdx[k]] = subOut[k];
            return VK_SUCCESS;
        }
    }
    // [v3.2] Huge compute shaders never come back from amdvlk64 (LLPC/LLVM) on GCN2: in the log all four compute
    // pipelines with 81655..99743 SPIR-V words ran for 2600+ s, while every smaller compute pipeline took < 3 s.
    // Fake them at once instead of blocking the game for 60 s each and leaving a core spinning for ever.
    if (g_spvCfg.heavyComputeWords > 0) {
        std::vector<char> heavy(createInfoCount, 0);
        bool anyHeavy = false;
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            SpvInfo hsi;
            size_t words = 0;
            if (GetSpvInfo(pCreateInfos[i].stage.module, hsi) && hsi.code) words = hsi.origWords ? hsi.origWords : hsi.code->size();   // [SWRT] compare the original size
            if (words >= static_cast<size_t>(g_spvCfg.heavyComputeWords)) {
                heavy[i] = 1; anyHeavy = true;
                static volatile LONG heavyC = 0;
                const LONG hc = InterlockedIncrement(&heavyC);
                if (hc <= 20 || hc % 50 == 0)
                    LogLine("HEAVY-FAKE compute pipeline: stage has %zu SPIR-V words (limit %d) -> FAKE without calling the driver (total=%ld)",
                        words, g_spvCfg.heavyComputeWords, static_cast<long>(hc));
            }
        }
        if (anyHeavy) {
            std::vector<VkComputePipelineCreateInfo> sub;
            std::vector<uint32_t> subIdx;
            for (uint32_t i = 0; i < createInfoCount; ++i) {
                if (heavy[i]) { if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>(); }
                else { sub.push_back(pCreateInfos[i]); subIdx.push_back(i); }
            }
            if (sub.empty()) return VK_SUCCESS;
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            vkCreateComputePipelines(device, pipelineCache, static_cast<uint32_t>(sub.size()),
                sub.data(), pAllocator, subOut.data());
            for (size_t k = 0; k < sub.size(); ++k)
                if (pPipelines) pPipelines[subIdx[k]] = subOut[k];
            return VK_SUCCESS;
        }
    }

    // Fast-fail heavy compute shaders (same 5s -13 problem)
    {
        std::vector<char> toxic(createInfoCount, 0);
        bool anyToxic = false;
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            if (SigToxic(SigOfModule(pCreateInfos[i].stage.module, static_cast<uint32_t>(pCreateInfos[i].stage.stage)))) { toxic[i] = 1; anyToxic = true; }
        }
        if (anyToxic) {
            static volatile LONG toxC = 0;
            std::vector<VkComputePipelineCreateInfo> sub;
            std::vector<uint32_t> subIdx;
            for (uint32_t i = 0; i < createInfoCount; ++i) {
                if (toxic[i]) {
                    if (pPipelines) pPipelines[i] = MakeFake<VkPipeline>();
                    LONG t = InterlockedIncrement(&toxC);
                    if (t <= 10 || t % 50 == 0)
                        LogLine("FAST-FAKE compute pipeline (learned toxic signature) total=%ld", (long)t);
                } else {
                    sub.push_back(pCreateInfos[i]);
                    subIdx.push_back(i);
                }
            }
            if (sub.empty()) return VK_SUCCESS;
            std::vector<VkPipeline> subOut(sub.size(), VK_NULL_HANDLE);
            vkCreateComputePipelines(device, pipelineCache, static_cast<uint32_t>(sub.size()),
                sub.data(), pAllocator, subOut.data());
            for (size_t k = 0; k < sub.size(); ++k)
                if (pPipelines) pPipelines[subIdx[k]] = subOut[k];
            return VK_SUCCESS;
        }
    }

    for (uint32_t i = 0; i < createInfoCount; ++i) {
        char d[180];
        DescribeModule(pCreateInfos[i].stage.module, d, sizeof(d));
        const auto& st = pCreateInfos[i].stage;
        const VkBaseInStructure* sp = reinterpret_cast<const VkBaseInStructure*>(st.pNext);
        Crumb("vkCreateComputePipelines[%u/%u] flags=0x%X stageFlags=0x%X stagePNext=%u spec=%u cache=%p %s", i, createInfoCount,
            (unsigned)pCreateInfos[i].flags, (unsigned)st.flags, sp ? (unsigned)sp->sType : 0u,
            st.pSpecializationInfo ? st.pSpecializationInfo->mapEntryCount : 0u, (void*)pipelineCache, d);
        SetJob("vkCreateComputePipelines[%u/%u] layout=%p %s", i, createInfoCount, (void*)pCreateInfos[i].layout, d);
    }
    VkResult cr = VK_ERROR_UNKNOWN;
    bool cTimedOut = false;
    bool cCallOk = false;
    {
        auto outP = std::make_shared<std::vector<VkPipeline>>(createInfoCount, VK_NULL_HANDLE);
        auto resP = std::make_shared<VkResult>(VK_ERROR_UNKNOWN);
        auto ciP  = std::make_shared<std::vector<VkComputePipelineCreateInfo>>(pCreateInfos, pCreateInfos + createInfoCount);
        t_timeoutOverrideSec = g_spvCfg.computeTimeoutSec;   // [v3.2]
        cCallOk = DriverCallTimed([=]() { *resP = real(device, pipelineCache, createInfoCount, ciP->data(), pAllocator, outP->data()); }, cTimedOut);
        t_timeoutOverrideSec = 0;
        if (cCallOk) {
            cr = *resP;
            if (pPipelines) for (uint32_t i = 0; i < createInfoCount; ++i) pPipelines[i] = (*outP)[i];
            if (cr == VK_SUCCESS) pcache::Dirty();
        }
    }
    if (!cCallOk) {
        if (cTimedOut) {
            LogLine("vkCreateComputePipelines TIMED OUT (count=%u) -> FAKE pipelines", createInfoCount);
            for (uint32_t i = 0; i < createInfoCount; ++i) PipelineMarkHung(&pCreateInfos[i].stage, 1);
        }
        else LogLine("CAUGHT driver crash in vkCreateComputePipelines (count=%u) -> FAKE pipelines", createInfoCount);
        for (uint32_t i = 0; i < createInfoCount && i < 4; ++i) {
            char d[180];
            DescribeModule(pCreateInfos[i].stage.module, d, sizeof(d));
            LogLine("  crashed compute pipeline[%u]: flags=0x%X stageFlags=0x%X spec=%u name=%s layout=%p %s", i,
                (unsigned)pCreateInfos[i].flags, (unsigned)pCreateInfos[i].stage.flags,
                pCreateInfos[i].stage.pSpecializationInfo ? pCreateInfos[i].stage.pSpecializationInfo->mapEntryCount : 0u,
                pCreateInfos[i].stage.pName ? pCreateInfos[i].stage.pName : "(null)", (void*)pCreateInfos[i].layout, d);
        }
        for (uint32_t k = 0; k < createInfoCount; ++k) if (pPipelines) pPipelines[k] = MakeFake<VkPipeline>();
        return VK_SUCCESS;
    }
    for (uint32_t i = 0; i < createInfoCount; ++i)
        SigRecord(SigOfModule(pCreateInfos[i].stage.module, static_cast<uint32_t>(pCreateInfos[i].stage.stage)), cr == VK_SUCCESS);
    if (cr != VK_SUCCESS) {
        LogLine("vkCreateComputePipelines FAILED: result=%d (count=%u) -> FAKE pipelines", (int)cr, createInfoCount);
        for (uint32_t k = 0; k < createInfoCount; ++k) if (pPipelines) pPipelines[k] = MakeFake<VkPipeline>();
        return VK_SUCCESS;
    }
    return cr;
}




// Pass-through wrappers. The build script lists these three names as local exports (so that they are NOT MASM thunks);
// they simply call the real driver. They are the natural place for submit/wait diagnostics (e.g. to find which submit
// ends in VK_ERROR_DEVICE_LOST).
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
    auto real = RealExport<PFN_vkQueueSubmit>("vkQueueSubmit");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(queue, submitCount, pSubmits, fence);
    InterlockedIncrement(&g_submitCount);
    InterlockedExchange(&g_lastSubmitRes, static_cast<LONG>(r));
    InterlockedExchange64(&g_lastSubmitTick, static_cast<LONG64>(GetTickCount64()));
    if (r != VK_SUCCESS) { static volatile LONG n = 0; if (InterlockedIncrement(&n) <= 20) LogLine("vkQueueSubmit FAILED: result=%d (submits=%u)%s", (int)r, submitCount, r == VK_ERROR_DEVICE_LOST ? " VK_ERROR_DEVICE_LOST" : ""); }
    return r;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueWaitIdle(VkQueue queue) {
    auto real = RealExport<PFN_vkQueueWaitIdle>("vkQueueWaitIdle");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(queue);
    if (r != VK_SUCCESS) { static volatile LONG n = 0; if (InterlockedIncrement(&n) <= 20) LogLine("vkQueueWaitIdle FAILED: result=%d", (int)r); }
    return r;
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout) {
    auto real = RealExport<PFN_vkWaitForFences>("vkWaitForFences");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, fenceCount, pFences, waitAll, timeout);
    if (r != VK_SUCCESS && r != VK_TIMEOUT) { static volatile LONG n = 0; if (InterlockedIncrement(&n) <= 20) LogLine("vkWaitForFences FAILED: result=%d", (int)r); }
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator) {
    if (IsFake(pipeline)) return;  // don't leak RT stub pipelines into the driver
    dynrend::UnregisterPipeline(pipeline);
    auto real = RealExport<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    if (real) real(device, pipeline, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator) {
    EraseSpvInfo(shaderModule);
    if (IsFake(shaderModule)) return; // Не передаем фейковый хэндл в драйвер!
    if (ModulePinned(shaderModule)) return;   // [HANGFIX] still referenced by an abandoned (stuck) compile
    auto real = RealExport<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    if (real) real(device, shaderModule, pAllocator);
}



extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline) {
    // Всегда сохраняем пайплайн в мапу (даже если он фейковый)
    // [v3.2] graphics and compute are separate bind points: one shared slot made a FAKE compute pipeline suppress
    // every later draw of the command buffer (and a real compute pipeline hid a FAKE graphics one).
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        AcquireSRWLockExclusive(&dynrend::g_lock);
        if (!dynrend::g_boundPipes) dynrend::g_boundPipes = new std::map<VkCommandBuffer, VkPipeline>();
        (*dynrend::g_boundPipes)[cb] = pipeline;
        ReleaseSRWLockExclusive(&dynrend::g_lock);
    } else if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) {
        AcquireSRWLockExclusive(&dynrend::g_lock);
        if (!dynrend::g_boundCompute) dynrend::g_boundCompute = new std::map<VkCommandBuffer, VkPipeline>();
        (*dynrend::g_boundCompute)[cb] = pipeline;
        ReleaseSRWLockExclusive(&dynrend::g_lock);
    }
    
    // Fake or NULL pipeline: never hand it to the driver (a NULL object crashes amdvlk64 at [obj+0x2A0])
    if (!pipeline || IsFake(pipeline)) return;
    
    auto real = RealExport<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    if (real) real(cb, bp, pipeline);
}
// [v3.2] same check for the COMPUTE bind point
static bool IsBoundComputeFake(VkCommandBuffer cb) {
    AcquireSRWLockShared(&dynrend::g_lock);
    bool isFake = false;
    if (dynrend::g_boundCompute) {
        auto it = dynrend::g_boundCompute->find(cb);
        if (it != dynrend::g_boundCompute->end() && (!it->second || IsFake(it->second))) isFake = true;
    }
    ReleaseSRWLockShared(&dynrend::g_lock);
    return isFake;
}
// Вспомогательная функция для проверки
static bool IsBoundPipelineFake(VkCommandBuffer cb) {
    AcquireSRWLockShared(&dynrend::g_lock);
    bool isFake = false;
    if (dynrend::g_boundPipes) {
        auto it = dynrend::g_boundPipes->find(cb);
        if (it != dynrend::g_boundPipes->end() && (!it->second || IsFake(it->second))) {
            isFake = true;
        }
    }
    ReleaseSRWLockShared(&dynrend::g_lock);
    return isFake;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDraw(VkCommandBuffer cb, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
    if (IsBoundPipelineFake(cb)) {
        static volatile LONG skipped = 0;
        LONG n = InterlockedIncrement(&skipped);
        if (n <= 5 || n % 500 == 0)
            LogLine("SKIP draw on FAKE pipeline (total skipped draws~%ld)", (long)n);
        return;
    }
    auto real = RealExport<PFN_vkCmdDraw>("vkCmdDraw");
    if (real) real(cb, vertexCount, instanceCount, firstVertex, firstInstance);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    if (IsBoundComputeFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect");
    if (real) real(cb, buffer, offset);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDrawIndirectCount>("vkCmdDrawIndirectCount");
    if (real) real(cb, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDrawIndexedIndirectCount>("vkCmdDrawIndexedIndirectCount");
    if (real) real(cb, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets) {
    rtfill::TrackBind(cb, firstSet, setCount, pDescriptorSets); // [RTFILL] ДО проверки FAKE
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE ? IsBoundComputeFake(cb) : IsBoundPipelineFake(cb)) return;   // [v3.2] per bind point
    auto real = RealExport<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    if (real) real(cb, bp, layout, firstSet, setCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues) {
    if (stageFlags == VK_SHADER_STAGE_COMPUTE_BIT ? IsBoundComputeFake(cb) : IsBoundPipelineFake(cb)) return;   // [v3.2]
    swpx::NotePush(cb, stageFlags, offset, size, pValues);   // [SWRT]
    auto real = RealExport<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    if (real) real(cb, layout, stageFlags, offset, size, pValues);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    if (real) real(cb, firstBinding, bindingCount, pBuffers, pOffsets);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    if (real) real(cb, buffer, offset, indexType);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
if (IsBoundPipelineFake(cb)) {
    static volatile LONG skippedIdx = 0;
    LONG k = InterlockedIncrement(&skippedIdx);
    if (k <= 5 || k % 2000 == 0)
        LogLine("SKIP indexed draw on FAKE pipeline (total=%ld)", (long)k);
    return;
}
    auto real = RealExport<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
    if (real) real(cb, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDispatch(VkCommandBuffer cb, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    if (IsBoundComputeFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDispatch>("vkCmdDispatch");
    if (real) real(cb, groupCountX, groupCountY, groupCountZ);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDrawIndirect>("vkCmdDrawIndirect");
    if (real) real(cb, buffer, offset, drawCount, stride);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    if (IsBoundPipelineFake(cb)) return;
    auto real = RealExport<PFN_vkCmdDrawIndexedIndirect>("vkCmdDrawIndexedIndirect");
    if (real) real(cb, buffer, offset, drawCount, stride);
}



extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImageView* pView) {
        static volatile LONG cnt = 0;
    if (pCreateInfo && InterlockedIncrement(&cnt) <= 40) {
        LogLine("vkCreateImageView ENTER: format=%d viewType=%d aspect=0x%X mip=%u..%u layer=%u..%u",
            (int)pCreateInfo->format, (int)pCreateInfo->viewType,
            (unsigned)pCreateInfo->subresourceRange.aspectMask,
            pCreateInfo->subresourceRange.baseMipLevel, pCreateInfo->subresourceRange.levelCount,
            pCreateInfo->subresourceRange.baseArrayLayer, pCreateInfo->subresourceRange.layerCount);
    }
    auto real = RealExport<PFN_vkCreateImageView>("vkCreateImageView");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pCreateInfo, pAllocator, pView);
    if (r == VK_SUCCESS && pView && *pView && pCreateInfo) {
        AcquireSRWLockExclusive(&dynrend::g_lock);
        if (!dynrend::g_imageViews) dynrend::g_imageViews = new std::map<VkImageView, dynrend::ImageViewInfo>();
        dynrend::ImageViewInfo info;
        info.format = pCreateInfo->format;
        info.aspect = pCreateInfo->subresourceRange.aspectMask;
info.image = pCreateInfo->image;
        (*dynrend::g_imageViews)[*pView] = info;
        ReleaseSRWLockExclusive(&dynrend::g_lock);
        rtfill::TrackView(*pView, pCreateInfo); // [RTFILL]
    }
	
	    if (r == VK_SUCCESS && pCreateInfo) {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 20)
            LogLine("vkCreateImageView: format=%d aspect=0x%X",
                (int)pCreateInfo->format, (unsigned)pCreateInfo->subresourceRange.aspectMask);
    } else if (r != VK_SUCCESS) {
        LogLine("vkCreateImageView FAILED: result=%d format=%d",
            (int)r, pCreateInfo ? (int)pCreateInfo->format : -1);
    }
	
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkDestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* pAllocator) {
    // [FIX] Framebuffers cached by dynamic-rendering emulation reference image views by HANDLE. AMDVLK reuses handle
    // values, so after the app destroyed a view, a NEW view with the same handle matched the stale cache entry and the
    // frame was rendered through a framebuffer that points to a destroyed view (black / garbage image, driver crashes).
    std::vector<VkFramebuffer> deadFbs;
    AcquireSRWLockExclusive(&dynrend::g_lock);
    if (dynrend::g_imageViews) dynrend::g_imageViews->erase(view);
    if (dynrend::g_fbCache) {
        for (auto it = dynrend::g_fbCache->begin(); it != dynrend::g_fbCache->end();) {
            bool uses = false;
            for (VkImageView v : it->first.views) if (v == view) { uses = true; break; }
            if (uses) { deadFbs.push_back(it->second); it = dynrend::g_fbCache->erase(it); }
            else ++it;
        }
    }
    ReleaseSRWLockExclusive(&dynrend::g_lock);
    if (!deadFbs.empty()) {
        auto destroyFb = RealExport<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer");
        if (destroyFb) for (VkFramebuffer fb : deadFbs) destroyFb(device, fb, nullptr);
    }
    auto real = RealExport<PFN_vkDestroyImageView>("vkDestroyImageView");
    if (real) real(device, view, pAllocator);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pInfo,
    VkCommandBuffer* pBuffers) {
    auto real = RealExport<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = real(device, pInfo, pBuffers);
    if (r == VK_SUCCESS && pInfo && pBuffers) {
        AcquireSRWLockExclusive(&dynrend::g_lock);
        if (!dynrend::g_cbDevices) dynrend::g_cbDevices = new std::map<VkCommandBuffer, VkDevice>();
        for (uint32_t i = 0; i < pInfo->commandBufferCount; ++i)
            (*dynrend::g_cbDevices)[pBuffers[i]] = device;
        ReleaseSRWLockExclusive(&dynrend::g_lock);
    }
	
	    if (r == VK_SUCCESS && pInfo) {
        static volatile LONG count = 0;
        if (InterlockedIncrement(&count) <= 5)
            LogLine("vkAllocateCommandBuffers: count=%u", pInfo->commandBufferCount);
    }
	
    return r;
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkFreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
    const VkCommandBuffer* pBuffers) {
    AcquireSRWLockExclusive(&dynrend::g_lock);
    if (dynrend::g_cbDevices && pBuffers)
        for (uint32_t i = 0; i < count; ++i) dynrend::g_cbDevices->erase(pBuffers[i]);
    if (dynrend::g_boundPipes && pBuffers)
        for (uint32_t i = 0; i < count; ++i) dynrend::g_boundPipes->erase(pBuffers[i]);
    if (dynrend::g_boundCompute && pBuffers)
        for (uint32_t i = 0; i < count; ++i) dynrend::g_boundCompute->erase(pBuffers[i]);
    if (dynrend::g_active && pBuffers)
        for (uint32_t i = 0; i < count; ++i) dynrend::g_active->erase(pBuffers[i]);
    ReleaseSRWLockExclusive(&dynrend::g_lock);
    auto real = RealExport<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers");
    if (real) real(device, pool, count, pBuffers);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* pInfo) {
    if (!pInfo) return;
    const VkDevice device = dynrend::LookupCBDevice(cb);
    if (device == VK_NULL_HANDLE) { LogLine("dynrend: BeginRendering without device cb=%p", cb); return; }
    dynrend::PipeInfo boundPipe{};
    const bool haveBoundPipe = dynrend::LookupBoundPipeline(cb, &boundPipe);
    dynrend::RPKey k;
    k.samples = haveBoundPipe ? boundPipe.samples : VK_SAMPLE_COUNT_1_BIT;
    k.viewMask = pInfo->viewMask;
    auto fmtHasStencil = [](VkFormat f) {
        return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT ||
               f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
    };
    // [FIX] Use the layout the APPLICATION declares in VkRenderingAttachmentInfo::imageLayout (the spec requires the
    // image to be in exactly that layout). The old code guessed from a global per-image barrier history, which is wrong
    // for command buffers recorded on several threads / out of order and for reused image handles.
    auto attLayout = [](VkImageLayout app, VkImage img, VkImageLayout def, VkImageAspectFlags aspect) -> VkImageLayout {
        VkImageLayout l = MapLayout2(app, aspect);
        if (l == VK_IMAGE_LAYOUT_UNDEFINED) l = dynrend::LayoutOf(img, def);
        switch (static_cast<int>(l)) {
        case 1000241000: case 1000241002: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case 1000241001: case 1000241003: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        default: return l;
        }
    };
    for (uint32_t i = 0; i < pInfo->colorAttachmentCount; ++i) {
        const auto& ca = pInfo->pColorAttachments[i];
        if (!ca.imageView) {
            k.colorFormats.push_back(VK_FORMAT_UNDEFINED);
            k.colorLoadOps.push_back(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
            k.colorLayouts.push_back(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            continue;
        }
        dynrend::ImageViewInfo vi{};
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        if (dynrend::LookupImageView(ca.imageView, &vi)) fmt = vi.format;
        if (fmt == VK_FORMAT_UNDEFINED && haveBoundPipe && i < boundPipe.colorFormats.size()) fmt = boundPipe.colorFormats[i];
        if (fmt == VK_FORMAT_UNDEFINED) fmt = VK_FORMAT_B8G8R8A8_UNORM;
        k.colorFormats.push_back(fmt);
        k.colorLoadOps.push_back(ca.loadOp);
        k.colorLayouts.push_back(attLayout(ca.imageLayout, vi.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT));
    }
    const VkRenderingAttachmentInfo* dA = (pInfo->pDepthAttachment && pInfo->pDepthAttachment->imageView) ? pInfo->pDepthAttachment : nullptr;
    const VkRenderingAttachmentInfo* sA = (pInfo->pStencilAttachment && pInfo->pStencilAttachment->imageView) ? pInfo->pStencilAttachment : nullptr;
    // The emulated render pass has ONE depth/stencil attachment: depth view if present, otherwise the stencil view.
    const VkImageView dsView = dA ? dA->imageView : (sA ? sA->imageView : VK_NULL_HANDLE);
    if (dA) {
        dynrend::ImageViewInfo vi{};
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        if (dynrend::LookupImageView(dA->imageView, &vi)) fmt = vi.format;
        if (fmt == VK_FORMAT_UNDEFINED && haveBoundPipe) fmt = boundPipe.depthFormat;
        if (fmt == VK_FORMAT_UNDEFINED) fmt = VK_FORMAT_D32_SFLOAT;
        k.depthFormat = fmt;
        k.depthLoadOp = dA->loadOp;
        k.depthLayout = attLayout(dA->imageLayout, vi.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        // [FIX] combined depth/stencil image: the stencil aspect of the SAME attachment used to get loadOp=DONT_CARE /
        // storeOp=DONT_CARE (stencil buffer lost). Carry the stencil load op over.
        if (sA && fmtHasStencil(fmt)) {
            k.stencilFormat = fmt;
            k.stencilLoadOp = sA->loadOp;
            k.stencilLayout = k.depthLayout;
        }
    } else if (sA) {
        dynrend::ImageViewInfo vi{};
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        if (dynrend::LookupImageView(sA->imageView, &vi)) fmt = vi.format;
        if (fmt == VK_FORMAT_UNDEFINED && haveBoundPipe) fmt = boundPipe.stencilFormat;
        if (fmt == VK_FORMAT_UNDEFINED) fmt = VK_FORMAT_S8_UINT;
        k.stencilFormat = fmt;
        k.stencilLoadOp = sA->loadOp;
        k.stencilLayout = attLayout(sA->imageLayout, vi.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_STENCIL_BIT);
    }
    VkRenderPass rp = dynrend::CreateOrGetRenderPass(device, k);
    if (rp == VK_NULL_HANDLE) return;
    std::vector<VkImageView> views;
    for (uint32_t i = 0; i < pInfo->colorAttachmentCount; ++i)
        if (pInfo->pColorAttachments[i].imageView) views.push_back(pInfo->pColorAttachments[i].imageView);
    // [FIX] a separate stencil view used to be appended as an extra framebuffer attachment although the render pass
    // only has one depth/stencil attachment (invalid framebuffer).
    if (dsView) views.push_back(dsView);
    const uint32_t fbW = pInfo->renderArea.offset.x + pInfo->renderArea.extent.width;
    const uint32_t fbH = pInfo->renderArea.offset.y + pInfo->renderArea.extent.height;
    const uint32_t layers = pInfo->layerCount ? pInfo->layerCount : 1;
    dynrend::FBKey fbk; fbk.rp = rp; fbk.views = views; fbk.width = fbW; fbk.height = fbH; fbk.layers = layers;
    VkFramebuffer fb = VK_NULL_HANDLE;
    AcquireSRWLockExclusive(&dynrend::g_lock);
    if (!dynrend::g_fbCache) dynrend::g_fbCache = new std::map<dynrend::FBKey, VkFramebuffer>();
    auto fbit = dynrend::g_fbCache->find(fbk);
    if (fbit != dynrend::g_fbCache->end()) fb = fbit->second;
    ReleaseSRWLockExclusive(&dynrend::g_lock);
    if (fb == VK_NULL_HANDLE) {
        VkFramebufferCreateInfo fbci{};
        fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbci.renderPass = rp;
        fbci.attachmentCount = (uint32_t)views.size();
        fbci.pAttachments = views.empty() ? nullptr : views.data();
        fbci.width = fbW ? fbW : 1; fbci.height = fbH ? fbH : 1; fbci.layers = layers ? layers : 1;
        auto createFb = RealExport<PFN_vkCreateFramebuffer>("vkCreateFramebuffer");
        if (!createFb) return;
        VkResult r = createFb(device, &fbci, nullptr, &fb);
        if (r != VK_SUCCESS) { LogLine("dynrend: CreateFramebuffer FAILED %d", (int)r); return; }
        AcquireSRWLockExclusive(&dynrend::g_lock);
        (*dynrend::g_fbCache)[fbk] = fb;
        ReleaseSRWLockExclusive(&dynrend::g_lock);
    }
    std::vector<VkClearValue> clears(views.size());
    uint32_t ci2 = 0;
    for (uint32_t i = 0; i < pInfo->colorAttachmentCount && ci2 < clears.size(); ++i) {
        if (!pInfo->pColorAttachments[i].imageView) continue;
        if (pInfo->pColorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
            clears[ci2].color = pInfo->pColorAttachments[i].clearValue.color;
        ++ci2;
    }
    if (dsView && ci2 < clears.size()) {
        if (dA) clears[ci2].depthStencil.depth = dA->clearValue.depthStencil.depth;
        if (sA) clears[ci2].depthStencil.stencil = sA->clearValue.depthStencil.stencil;
    }
    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = pInfo->renderArea;
    rpbi.clearValueCount = (uint32_t)clears.size();
    rpbi.pClearValues = clears.empty() ? nullptr : clears.data();
    auto cmdBegin = RealExport<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass");
    if (cmdBegin) cmdBegin(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    AcquireSRWLockExclusive(&dynrend::g_lock);
    if (!dynrend::g_active) dynrend::g_active = new std::map<VkCommandBuffer, dynrend::ActiveState>();
    dynrend::ActiveState as; as.renderPass = rp; as.framebuffer = fb;
    (*dynrend::g_active)[cb] = as;
    ReleaseSRWLockExclusive(&dynrend::g_lock);
}
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdEndRendering(VkCommandBuffer cb) {
    bool have = false;
    AcquireSRWLockExclusive(&dynrend::g_lock);
    if (dynrend::g_active) {
        auto it = dynrend::g_active->find(cb);
        if (it != dynrend::g_active->end()) {
            dynrend::g_active->erase(it);
            have = true;
        }
    }
    ReleaseSRWLockExclusive(&dynrend::g_lock);

    if (!have) {
        static volatile LONG n = 0;
        if (InterlockedIncrement(&n) <= 8)
            LogLine("dynrend: vkCmdEndRendering without active state for cb=%p", cb);
        return;
    }
    auto cmdEnd = RealExport<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass");
    if (cmdEnd) cmdEnd(cb);
    // Framebuffer stays in g_fbCache (reused next frame). Destroy on process exit.
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* pInfo) {
    vkCmdBeginRendering(cb, pInfo);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdEndRenderingKHR(VkCommandBuffer cb) {
    vkCmdEndRendering(cb);
}

// ---------------------------------------------------------------------------
// maintenance4 stubs (translated to legacy vkGet*MemoryRequirements via a
// temporary resource)
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceBufferMemoryRequirements(VkDevice device, const VkDeviceBufferMemoryRequirements* pInfo,
    VkMemoryRequirements2* pMemoryRequirements) {
    if (!pInfo || !pInfo->pCreateInfo || !pMemoryRequirements) return;
    auto create = RealExport<PFN_vkCreateBuffer>("vkCreateBuffer");
    auto getReq = RealExport<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    auto destroy = RealExport<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    if (!create || !getReq || !destroy) return;
    VkBuffer tmp = VK_NULL_HANDLE;
    VkBufferCreateInfo patched{};
    const VkBufferCreateInfo* ci = pInfo->pCreateInfo;
    if (PatchSwrtBufferCreateInfo(*ci, patched)) ci = &patched;
    VkResult r = create(device, ci, nullptr, &tmp);
    if (r == VK_SUCCESS && tmp) {
        // Preserve output pNext, including dedicated allocation requirements.
        auto getReq2 = RealExport<PFN_vkGetBufferMemoryRequirements2>("vkGetBufferMemoryRequirements2");
        if (getReq2) {
            VkBufferMemoryRequirementsInfo2 ri{};
            ri.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            ri.buffer = tmp;
            getReq2(device, &ri, pMemoryRequirements);
        } else {
            getReq(device, tmp, &pMemoryRequirements->memoryRequirements);
            if (pMemoryRequirements->pNext && FirstTime("maintenance4:buffer-output-pnext"))
                LogLine("!!! maintenance4 buffer requirements: output pNext unavailable on this driver");
        }
        destroy(device, tmp, nullptr);
    } else {
        LogLine("dynrend: temp vkCreateBuffer for memory reqs failed result=%d", static_cast<int>(r));
        VkMemoryRequirements& mr = pMemoryRequirements->memoryRequirements;
        mr.size = 1024; mr.alignment = 256; mr.memoryTypeBits = 0xFFFFFFFFu;
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceImageMemoryRequirements(VkDevice device, const VkDeviceImageMemoryRequirements* pInfo,
    VkMemoryRequirements2* pMemoryRequirements) {
    if (!pInfo || !pInfo->pCreateInfo || !pMemoryRequirements) return;
    auto create = RealExport<PFN_vkCreateImage>("vkCreateImage");
    auto getReq = RealExport<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements");
    auto destroy = RealExport<PFN_vkDestroyImage>("vkDestroyImage");
    if (!create || !getReq || !destroy) return;
    VkImage tmp = VK_NULL_HANDLE;
    VkResult r = create(device, pInfo->pCreateInfo, nullptr, &tmp);
    if (r == VK_SUCCESS && tmp) {
        getReq(device, tmp, &pMemoryRequirements->memoryRequirements);
        destroy(device, tmp, nullptr);
    } else {
        LogLine("dynrend: temp vkCreateImage for memory reqs failed result=%d", static_cast<int>(r));
        VkMemoryRequirements& mr = pMemoryRequirements->memoryRequirements;
        mr.size = 4096; mr.alignment = 256; mr.memoryTypeBits = 0xFFFFFFFFu;
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetDeviceImageSparseMemoryRequirements(VkDevice, const VkDeviceImageMemoryRequirements*,
    uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements2*) {
    if (pSparseMemoryRequirementCount) *pSparseMemoryRequirementCount = 0;
}



// v4 additions
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit2KHR(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* pInfo);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* pInfo);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t query);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdWriteTimestamp2KHR(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t query);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo);

// v5 additions
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    VkSurfaceCapabilitiesKHR* pCapabilities);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    uint32_t* pCount, VkSurfaceFormatKHR* pFormats);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice pd, VkSurfaceKHR surface,
    uint32_t* pCount, VkPresentModeKHR* pModes);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImage* pImage);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemReq);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetImageMemoryRequirements2(VkDevice device, const VkImageMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemReq);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindImageMemory2(VkDevice device, uint32_t bindCount, const VkBindImageMemoryInfo* pBinds);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemReq);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetBufferMemoryRequirements2(VkDevice device, const VkBufferMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemReq);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkBindBufferMemory2(VkDevice device, uint32_t bindCount, const VkBindBufferMemoryInfo* pBinds);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout);
	
// v6 additions
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSampler* pSampler);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
    VkDescriptorSet* pDescriptorSets);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet* pDescriptorCopies);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkFence* pFence);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass);
	
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDraw(VkCommandBuffer cb, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDispatch(VkCommandBuffer cb, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);

extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets);
extern "C" VKAPI_ATTR void VKAPI_CALL
vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType);
	
extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);	
	
// ---------------------------------------------------------------------------
// Loader entry points
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName);
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName);

extern "C" FARPROC ResolveVkExport(uint32_t index) noexcept {
    if (index >= VK_PROXY_EXPORT_COUNT || !EnsureSystemLoader()) return nullptr;
    PVOID cached = InterlockedCompareExchangePointer(&g_vkSlots[index], nullptr, nullptr);
    if (cached) return reinterpret_cast<FARPROC>(cached);
    FARPROC resolved = GetProcAddress(g_realLoader, kVkProxyExportNames[index]);
    if (!resolved) {
        LogLine("System Loader does not export %s", kVkProxyExportNames[index]);
        return nullptr;
    }
    PVOID previous = InterlockedCompareExchangePointer(
        &g_vkSlots[index], reinterpret_cast<PVOID>(resolved), nullptr);
    return reinterpret_cast<FARPROC>(previous ? previous : reinterpret_cast<PVOID>(resolved));
}

// ---------------------------------------------------------------------------
// Interposed Vulkan commands (properties / features / extensions / device)
// ---------------------------------------------------------------------------
extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* pProperties) {
    auto real = RealExport<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
    if (!real) return;
    real(pd, pProperties);
    FixupProperties(pd, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* pProperties) {
    using Fn = PFN_vkGetPhysicalDeviceProperties2;
    Fn real = RealExport<Fn>("vkGetPhysicalDeviceProperties2");
    if (!real) real = RealExport<Fn>("vkGetPhysicalDeviceProperties2KHR");
    if (!real || !pProperties) return;
    LogChainOnce("Properties2", pProperties->pNext);
    real(pd, pProperties);
    FixupProperties(pd, &pProperties->properties);
    FixupPropertiesChain(pd, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceProperties2KHR(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
    vkGetPhysicalDeviceProperties2(pd, p);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* pFeatures) {
    using Fn = PFN_vkGetPhysicalDeviceFeatures2;
    Fn real = RealExport<Fn>("vkGetPhysicalDeviceFeatures2");
    if (!real) real = RealExport<Fn>("vkGetPhysicalDeviceFeatures2KHR");
    if (!real || !pFeatures) return;
    LogChainOnce("Features2", pFeatures->pNext);
    real(pd, pFeatures);
    FixupFeaturesChain(pd, pFeatures);
}

extern "C" VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* p) {
    vkGetPhysicalDeviceFeatures2(pd, p);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char* pLayerName,
    uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;
    std::vector<VkExtensionProperties> realExts;
    if (!QueryRealExtensions(pd, pLayerName, realExts)) return VK_ERROR_INITIALIZATION_FAILED;

    if (pLayerName == nullptr) {
        char key[64];
        sprintf_s(key, sizeof(key), "extlist:%p", static_cast<void*>(pd));
        if (FirstTime(key)) {
            LogLine("REAL device extensions (%u):", static_cast<unsigned>(realExts.size()));
            for (const auto& e : realExts) LogLine("    %s v%u", e.extensionName, e.specVersion);
        }
    }

    std::vector<VkExtensionProperties> all = realExts;
    const bool advertise = (pLayerName == nullptr) && g_rtSpoof && !DeviceHasRealRT(pd);
    if (advertise) {
        for (const char* name : kFakeExtensions) {
            if (HasExt(realExts, name)) continue;
            VkExtensionProperties ep{};
            strncpy_s(ep.extensionName, VK_MAX_EXTENSION_NAME_SIZE, name, _TRUNCATE);
            ep.specVersion = 1;
            all.push_back(ep);
        }
    }

    const uint32_t total = static_cast<uint32_t>(all.size());
    if (!pProperties) { *pPropertyCount = total; return VK_SUCCESS; }
    const uint32_t cap = *pPropertyCount;
    const uint32_t n = cap < total ? cap : total;
    if (n) memcpy(pProperties, all.data(), n * sizeof(VkExtensionProperties));
    *pPropertyCount = n;
    return n < total ? VK_INCOMPLETE : VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    auto real = RealExport<PFN_vkCreateDevice>("vkCreateDevice");
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pCreateInfo) return VK_ERROR_INITIALIZATION_FAILED;

    LogLine("vkCreateDevice: app requests %u extensions", pCreateInfo->enabledExtensionCount);
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
        LogLine("  REQUESTED EXT: %s", pCreateInfo->ppEnabledExtensionNames[i]);
    LogChainOnce("DeviceCreateInfo", pCreateInfo->pNext);

    const bool realRt = DeviceHasRealRT(pd);
    if (realRt) {
        InterlockedExchange(&g_rtSpoof, 0);
        LogLine("Real RT hardware present -> RT spoofing OFF, pass-through");
        return real(pd, pCreateInfo, pAllocator, pDevice);
    }

    std::vector<VkExtensionProperties> realExts;
    QueryRealExtensions(pd, nullptr, realExts);

    VkDeviceCreateInfo ci = *pCreateInfo;
    std::vector<const char*> exts;
    for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i) {
        const char* n = ci.ppEnabledExtensionNames[i];
        if (IsFakeExtension(n) && !HasExt(realExts, n)) {
            LogLine("  STRIPPED (not in real driver): %s", n);
            continue;
        }
        exts.push_back(n);
    }
    // [FIX] The 1.2 driver exposes VK_EXT_extended_dynamic_state, but the game (written for Vulkan 1.3) never enables it,
    // so every vkCmdSetCullMode / SetFrontFace / SetDepthTestEnable / SetViewportWithCount ... was a NO-OP or a NULL
    // trampoline ("MISSING dynamic-state ... NO-OP" in the log). Enable it and route the 1.3 core names to it.
    bool edsAvail = false, edsAdded = false;
    {
        bool have = false;
        for (const char* n : exts) if (!strcmp(n, "VK_EXT_extended_dynamic_state")) have = true;
        edsAvail = have || HasExt(realExts, "VK_EXT_extended_dynamic_state");
        if (!have && edsAvail) {
            exts.push_back("VK_EXT_extended_dynamic_state");
            edsAdded = true;
            LogLine("  ADDED EXT: VK_EXT_extended_dynamic_state (backs the Vulkan 1.3 dynamic-state commands)");
        }
    }
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    struct Unlink { VkBaseOutStructure** link; VkBaseOutStructure* node; };
    struct Saved  { VkBaseOutStructure* node; std::vector<uint8_t> bytes; };
    std::vector<Unlink> unlinked;
    std::vector<Saved> saved;

    VkPhysicalDeviceFeatures localCore{};
    if (ci.pEnabledFeatures) {
        localCore = *ci.pEnabledFeatures;
        ClampCoreFeatures(pd, &localCore, "pEnabledFeatures");
        ci.pEnabledFeatures = &localCore;
    }

    VkBaseOutStructure** link = reinterpret_cast<VkBaseOutStructure**>(const_cast<void**>(&ci.pNext));
    unsigned guard = 0;
    while (*link && guard++ < 128) {
        VkBaseOutStructure* node = *link;
        const uint32_t s = static_cast<uint32_t>(node->sType);

        bool drop = (s == ST_AS_FEATURES || s == ST_RTP_FEATURES || s == ST_RAY_QUERY_FEATURES ||
                     s == ST_RT_MAINTENANCE1_FEATURES);
        for (const DropInfo& d : kDropIfExtMissing)
            if (d.sType == s && !HasExt(realExts, d.ext)) drop = true;
        if (drop) {
            LogLine("  UNLINKED feature struct sType=%u %s (extension not in real driver)", s, KnownSTypeName(s));
            unlinked.push_back({ link, node });
            *link = node->pNext;
            continue;
        }

        // Preserve proxy-emulated feature structs: dynamicRendering, sync2,
        // maintenance4 are implemented in this DLL, so do NOT clamp them to
        // the real driver's values.
        if (s == ST_VULKAN13_FEATURES || s == ST_DYNAMIC_RENDERING_FEAT ||
            s == ST_SYNCHRONIZATION2_FEAT || s == ST_MAINTENANCE4_FEAT) {
            LogLine("  PRESERVED feature struct sType=%u %s (proxy-emulated)", s, KnownSTypeName(s));
            link = &node->pNext;
            continue;
        }

        if (s == static_cast<uint32_t>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)) {
            auto* f2 = reinterpret_cast<VkPhysicalDeviceFeatures2*>(node);
            Saved sv;
            sv.node = node;
            sv.bytes.assign(reinterpret_cast<uint8_t*>(node), reinterpret_cast<uint8_t*>(node) + sizeof(VkPhysicalDeviceFeatures2));
            saved.push_back(std::move(sv));
            ClampCoreFeatures(pd, &f2->features, "Features2.features");
        } else {
            for (const ClampInfo& c : kClampable) {
                if (c.sType == s) {
                    Saved sv;
                    sv.node = node;
                    sv.bytes.assign(reinterpret_cast<uint8_t*>(node), reinterpret_cast<uint8_t*>(node) + c.size);
                    saved.push_back(std::move(sv));
                    ClampToReal(pd, node, c);
                    // NEVER re-enable shaderFloat16 for the real CreateDevice call — AMDVLK
                    // returns VK_ERROR_FEATURE_NOT_PRESENT (-8) and the game AVs on null device.
                    // Float16 is only spoofed in GetPhysicalDeviceFeatures2; shaders are
                    // promoted to fp32 in the proxy before they reach the driver.
                    // dynamicRendering / sync2 / maintenance4: keep enabled only if the
                    // structure was already requested; driver may ignore unknown bits on
                    // older ICD paths, but forcing TRUE on a clamped-false bit caused -8.
                    break;
                }
            }
        }
        link = &node->pNext;
    }

    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT edsFeat{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT };
    const void* pNextBeforeEds = ci.pNext;
    if (edsAvail) {
        bool inChain = false;
        for (const VkBaseInStructure* n = reinterpret_cast<const VkBaseInStructure*>(ci.pNext); n; n = n->pNext)
            if (static_cast<uint32_t>(n->sType) == static_cast<uint32_t>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT)) {
                inChain = true;
                const_cast<VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(
                    reinterpret_cast<const VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(n))->extendedDynamicState = VK_TRUE;
            }
        if (!inChain) {
            edsFeat.extendedDynamicState = VK_TRUE;
            edsFeat.pNext = const_cast<void*>(ci.pNext);
            ci.pNext = &edsFeat;
        }
    }
    // [SWRT] BVH builder kernels need 64-bit ints, buffer device address and scalar block layout.
    LoadSpvConfig();
    if (g_spvCfg.softwareRT && g_spvCfg.swrtForceFeatures) {
        bool got12 = false;
        for (VkBaseOutStructure* n = const_cast<VkBaseOutStructure*>(reinterpret_cast<const VkBaseOutStructure*>(ci.pNext)); n; n = n->pNext) {
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
                auto* f = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(n);
                f->bufferDeviceAddress = VK_TRUE; f->scalarBlockLayout = VK_TRUE; got12 = true;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                reinterpret_cast<VkPhysicalDeviceFeatures2*>(n)->features.shaderInt64 = VK_TRUE;
            }
        }
        if (ci.pEnabledFeatures) localCore.shaderInt64 = VK_TRUE;
        if (!got12) LogLine("SWRT: no Vulkan12Features in the device chain - BVH builder may be unavailable");
    }
    VkResult r = real(pd, &ci, pAllocator, pDevice);
    if (r != VK_SUCCESS && edsAdded) {
        // The driver refused the extra extension: retry exactly as the game asked.
        LogLine("vkCreateDevice failed (%d) with the extra VK_EXT_extended_dynamic_state -> retrying without it", (int)r);
        ci.pNext = pNextBeforeEds;
        exts.pop_back();
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
        edsAvail = false;
        r = real(pd, &ci, pAllocator, pDevice);
    }
    if (r == VK_SUCCESS && edsAvail) InterlockedExchange(&g_edsEnabled, 1);

    for (size_t i = unlinked.size(); i-- > 0;) *unlinked[i].link = unlinked[i].node;
    for (auto& sv : saved) {
        VkBaseOutStructure* keepNext = sv.node->pNext;
        memcpy(sv.node, sv.bytes.data(), sv.bytes.size());
        sv.node->pNext = keepNext;
    }

    LogLine("vkCreateDevice (sanitized) result=%d", static_cast<int>(r));
    if (r == VK_SUCCESS && pDevice) {
        g_lastDevice = *pDevice;
        InstallCrashHandlerFirst();
        if (g_spvCfg.softwareRT) swpx::EnsureInit(*pDevice);   // [SWRT] compile the BVH kernels now, before any buffers exist (a failure only disables SWRT)
        g_createdFamilies.clear();
        for (uint32_t qi = 0; qi < pCreateInfo->queueCreateInfoCount; ++qi)
            g_createdFamilies.push_back({ pCreateInfo->pQueueCreateInfos[qi].queueFamilyIndex, pCreateInfo->pQueueCreateInfos[qi].queueCount });
    }
    return r;
}

// ---------------------------------------------------------------------------
// Loader entry points (definitions)
// ---------------------------------------------------------------------------
namespace {
    // Expanded: every intercepted function must be returned via GIPA/GDPA.
    // Returning nullptr was a primary cause of the EXEC@0 BagSplat crash.
    PFN_vkVoidFunction LocalOverride(const char* name) {
        if (!name) return nullptr;
        if (!strcmp(name, "vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateInstance);
        if (!strcmp(name, "vkGetInstanceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(&vkGetInstanceProcAddr);
        if (!strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(&vkGetDeviceProcAddr);
        if (!strcmp(name, "vkEnumerateDeviceExtensionProperties")) return reinterpret_cast<PFN_vkVoidFunction>(&vkEnumerateDeviceExtensionProperties);
        if (!strcmp(name, "vkGetPhysicalDeviceProperties")) return reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceProperties);
        if (!strcmp(name, "vkGetPhysicalDeviceProperties2") || !strcmp(name, "vkGetPhysicalDeviceProperties2KHR"))
            return reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceProperties2);
        if (!strcmp(name, "vkGetPhysicalDeviceFeatures2") || !strcmp(name, "vkGetPhysicalDeviceFeatures2KHR"))
            return reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceFeatures2);
        if (!strcmp(name, "vkCreateDevice")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateDevice);
        if (!strcmp(name, "vkDestroyShaderModule")) return reinterpret_cast<PFN_vkVoidFunction>(&vkDestroyShaderModule);
        if (!strcmp(name, "vkCreateShaderModule")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateShaderModule);
        if (!strcmp(name, "vkCreateGraphicsPipelines")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateGraphicsPipelines);
        if (!strcmp(name, "vkCreateComputePipelines")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateComputePipelines);
        if (!strcmp(name, "vkDestroyPipeline")) return reinterpret_cast<PFN_vkVoidFunction>(&vkDestroyPipeline);
        if (!strcmp(name, "vkCmdBindPipeline")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdBindPipeline);
        if (!strcmp(name, "vkCmdDraw")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDraw);
        if (!strcmp(name, "vkCmdDrawIndexed")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndexed);
        if (!strcmp(name, "vkCmdDrawIndirect")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndirect);
        if (!strcmp(name, "vkCmdDrawIndexedIndirect")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndexedIndirect);
        if (!strcmp(name, "vkCmdDispatch")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDispatch);
        if (!strcmp(name, "vkCmdDispatchIndirect")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDispatchIndirect);
        if (!strcmp(name, "vkQueuePresentKHR")) return reinterpret_cast<PFN_vkVoidFunction>(&vkQueuePresentKHR);
        if (!strcmp(name, "vkCreateSwapchainKHR")) return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateSwapchainKHR);
        if (!strcmp(name, "vkCmdBeginRendering") || !strcmp(name, "vkCmdBeginRenderingKHR"))
            return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdBeginRendering);
        if (!strcmp(name, "vkCmdEndRendering") || !strcmp(name, "vkCmdEndRenderingKHR"))
            return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdEndRendering);
        // Optional debug utils - engines sometimes call without null-check
        if (!strcmp(name, "vkCreateDebugUtilsMessengerEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_CreateDebugUtilsMessengerEXT);
        if (!strcmp(name, "vkDestroyDebugUtilsMessengerEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_DestroyDebugUtilsMessengerEXT);
        if (!strcmp(name, "vkSubmitDebugUtilsMessageEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_SubmitDebugUtilsMessageEXT);
        if (!strcmp(name, "vkSetDebugUtilsObjectNameEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_SetDebugUtilsObjectNameEXT);
        if (!strcmp(name, "vkSetDebugUtilsObjectTagEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_SetDebugUtilsObjectTagEXT);
        if (!strcmp(name, "vkCmdBeginDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_CmdBeginDebugUtilsLabelEXT);
        if (!strcmp(name, "vkCmdEndDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_CmdEndDebugUtilsLabelEXT);
        if (!strcmp(name, "vkCmdInsertDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_CmdInsertDebugUtilsLabelEXT);
        if (!strcmp(name, "vkQueueBeginDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_QueueBeginDebugUtilsLabelEXT);
        if (!strcmp(name, "vkQueueEndDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_QueueEndDebugUtilsLabelEXT);
        if (!strcmp(name, "vkQueueInsertDebugUtilsLabelEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_QueueInsertDebugUtilsLabelEXT);
        if (!strcmp(name, "vkCreateDebugReportCallbackEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_CreateDebugReportCallbackEXT);
        if (!strcmp(name, "vkDestroyDebugReportCallbackEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_DestroyDebugReportCallbackEXT);
        if (!strcmp(name, "vkDebugReportMessageEXT")) return reinterpret_cast<PFN_vkVoidFunction>(&Stub_DebugReportMessageEXT);
        return nullptr;
    }
}

namespace {
    // Vulkan 1.3 core dynamic-state commands that a 1.2 driver does not have. They map to VK_EXT_extended_dynamic_state
    // (enabled in vkCreateDevice) when possible, otherwise they are harmless no-ops (the pipeline's static state applies).
    struct DynStateMap { const char* core; const char* ext; bool inExt1; };
    const DynStateMap kDynState[] = {
        { "vkCmdSetCullMode",               "vkCmdSetCullModeEXT",               true  },
        { "vkCmdSetFrontFace",              "vkCmdSetFrontFaceEXT",              true  },
        { "vkCmdSetPrimitiveTopology",      "vkCmdSetPrimitiveTopologyEXT",      true  },
        { "vkCmdSetViewportWithCount",      "vkCmdSetViewportWithCountEXT",      true  },
        { "vkCmdSetScissorWithCount",       "vkCmdSetScissorWithCountEXT",       true  },
        { "vkCmdSetDepthTestEnable",        "vkCmdSetDepthTestEnableEXT",        true  },
        { "vkCmdSetDepthWriteEnable",       "vkCmdSetDepthWriteEnableEXT",       true  },
        { "vkCmdSetDepthCompareOp",         "vkCmdSetDepthCompareOpEXT",         true  },
        { "vkCmdSetDepthBoundsTestEnable",  "vkCmdSetDepthBoundsTestEnableEXT",  true  },
        { "vkCmdSetStencilTestEnable",      "vkCmdSetStencilTestEnableEXT",      true  },
        { "vkCmdSetStencilOp",              "vkCmdSetStencilOpEXT",              true  },
        // VK_EXT_extended_dynamic_state2 (not offered by this driver)
        { "vkCmdSetRasterizerDiscardEnable","vkCmdSetRasterizerDiscardEnableEXT",false },
        { "vkCmdSetDepthBiasEnable",        "vkCmdSetDepthBiasEnableEXT",        false },
        { "vkCmdSetPrimitiveRestartEnable", "vkCmdSetPrimitiveRestartEnableEXT", false },
    };

    PFN_vkVoidFunction ResolveDynState(VkDevice device, const char* name) {
        for (const DynStateMap& e : kDynState) {
            if (strcmp(name, e.core) && strcmp(name, e.ext)) continue;
            PFN_vkVoidFunction f = nullptr;
            if (g_realGdpa && device) {
                if (e.inExt1) { if (g_edsEnabled) f = g_realGdpa(device, e.ext); }
                else { f = g_realGdpa(device, e.ext); if (!f) f = g_realGdpa(device, e.core); }
            }
            if (!f) {
                if (FirstTime(std::string("dynstate-noop:") + name))
                    LogLine("!!! dynamic-state %s unavailable on the real driver -> NO-OP (pipeline static state is used)", name);
                f = reinterpret_cast<PFN_vkVoidFunction>(&Stub_NoopCmd);
            } else if (FirstTime(std::string("dynstate-real:") + name)) {
                LogLine("dynamic-state %s -> real driver function", name);
            }
            return f;
        }
        return nullptr;
    }

    // KHR/EXT name -> promoted core name ("vkGetBufferDeviceAddressKHR" -> "vkGetBufferDeviceAddress", ...). The real
    // loader returns NULL for the KHR/EXT name when the game did not enable the extension (it uses the 1.x core API
    // through a different name table); the game then jumped to NULL ("EXEC@0 skipped null call" in the log).
    PFN_vkVoidFunction ResolveBySuffixStrip(VkDevice device, const char* name) {
        const size_t len = strlen(name);
        if (len < 8 || len - 3 >= 160) return nullptr;
        if (strcmp(name + len - 3, "KHR") && strcmp(name + len - 3, "EXT")) return nullptr;
        char core[160];
        memcpy(core, name, len - 3);
        core[len - 3] = 0;
        PFN_vkVoidFunction f = stub::Find(core);
        if (!f) f = LocalOverride(core);
        if (!f && g_realGdpa && device) f = g_realGdpa(device, core);
        if (f && FirstTime(std::string("gdpa-strip:") + name))
            LogLine("PROXY served via suffix-strip %s -> %s", name, core);
        return f;
    }
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName || !EnsureSystemLoader()) return nullptr;

    // Prefer our stubs / overrides first so the game never stores a null FP.
    if (PFN_vkVoidFunction s = stub::Find(pName)) {
        if (FirstTime(std::string("stub:") + pName)) LogLine("PROXY served via GIPA: %s", pName);
        return s;
    }
    if (PFN_vkVoidFunction o = LocalOverride(pName)) return o;

    PFN_vkVoidFunction realFunction = g_realGipa(instance, pName);
    // Fallback 1: many core/instance functions are also exported by the system loader DLL.
    if (!realFunction && g_realLoader) {
        realFunction = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_realLoader, pName));
    }
    // Fallback 2: our own module exports (generated thunks).
    if (!realFunction && g_thisModule) {
        realFunction = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_thisModule, pName));
        if (realFunction && FirstTime(std::string("gipa-self:") + pName))
            LogLine("PROXY served via own export (GIPA fallback): %s", pName);
    }
    // Fallback 3: KHR/EXT alias -> core name (e.g. vkGetPhysicalDeviceFormatProperties2KHR
    // -> vkGetPhysicalDeviceFormatProperties2). Many games query the KHR name even when
    // the function is core in 1.1+.
    if (!realFunction && pName) {
        const char* aliases[][2] = {
            { "vkGetPhysicalDeviceFormatProperties2KHR", "vkGetPhysicalDeviceFormatProperties2" },
            { "vkGetPhysicalDeviceImageFormatProperties2KHR", "vkGetPhysicalDeviceImageFormatProperties2" },
            { "vkGetPhysicalDeviceQueueFamilyProperties2KHR", "vkGetPhysicalDeviceQueueFamilyProperties2" },
            { "vkGetPhysicalDeviceMemoryProperties2KHR", "vkGetPhysicalDeviceMemoryProperties2" },
            { "vkGetPhysicalDeviceSparseImageFormatProperties2KHR", "vkGetPhysicalDeviceSparseImageFormatProperties2" },
            { "vkGetPhysicalDeviceProperties2KHR", "vkGetPhysicalDeviceProperties2" },
            { "vkGetPhysicalDeviceFeatures2KHR", "vkGetPhysicalDeviceFeatures2" },
            { "vkEnumeratePhysicalDeviceGroupsKHR", "vkEnumeratePhysicalDeviceGroups" },
            { "vkGetDeviceBufferMemoryRequirementsKHR", "vkGetDeviceBufferMemoryRequirements" },
            { "vkGetDeviceImageMemoryRequirementsKHR", "vkGetDeviceImageMemoryRequirements" },
            { "vkGetDeviceImageSparseMemoryRequirementsKHR", "vkGetDeviceImageSparseMemoryRequirements" },
            { "vkCmdBeginRenderingKHR", "vkCmdBeginRendering" },
            { "vkCmdEndRenderingKHR", "vkCmdEndRendering" },
            { "vkCmdPipelineBarrier2KHR", "vkCmdPipelineBarrier2" },
            { "vkCmdWriteTimestamp2KHR", "vkCmdWriteTimestamp2" },
            { "vkQueueSubmit2KHR", "vkQueueSubmit2" },
            { nullptr, nullptr }
        };
        for (int i = 0; aliases[i][0]; ++i) {
            if (!strcmp(pName, aliases[i][0])) {
                if (g_thisModule)
                    realFunction = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_thisModule, aliases[i][1]));
                if (!realFunction && g_realLoader)
                    realFunction = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_realLoader, aliases[i][1]));
                if (!realFunction)
                    realFunction = g_realGipa(instance, aliases[i][1]);
                if (realFunction && FirstTime(std::string("gipa-alias:") + pName))
                    LogLine("PROXY served via alias %s -> %s", pName, aliases[i][1]);
                break;
            }
        }
    }
    if (!realFunction && !strncmp(pName, "vkCmdSet", 8)) {
    if (FirstTime(std::string("noop-set:") + pName))
        LogLine("!!! MISSING dynamic-state %s -> NO-OP (state NOT applied!)", pName);
    return reinterpret_cast<PFN_vkVoidFunction>(&Stub_NoopCmd);
}
if (!realFunction) {
        if (FirstTime(std::string("gipa-null:") + pName))
            LogLine("GIPA returned NULL for %s", pName);
        return nullptr;
    }
    return realFunction;
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName || !EnsureSystemLoader()) return nullptr;

    // Prefer our stubs / overrides first. Never let the game receive nullptr
    // for a function we intercept — that was the EXEC@0 crash root cause.
    if (PFN_vkVoidFunction s = stub::Find(pName)) {
        if (FirstTime(std::string("stub:") + pName)) LogLine("PROXY served via GDPA: %s", pName);
        return s;
    }
    if (PFN_vkVoidFunction o = LocalOverride(pName)) return o;
    if (PFN_vkVoidFunction d = ResolveDynState(device, pName)) return d;

    PFN_vkVoidFunction result = g_realGdpa(device, pName);
    // Fallback via loader export (helps for some core functions).
    if (!result && g_realLoader) {
        result = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_realLoader, pName));
    }
    // Fallback via our own exports (generated thunks).
    if (!result && g_thisModule) {
        result = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_thisModule, pName));
        if (result && FirstTime(std::string("gdpa-self:") + pName))
            LogLine("PROXY served via own export (GDPA fallback): %s", pName);
    }
    // KHR/EXT -> core alias (+ поиск сначала среди наших stub/override!)
    if (!result && pName) {
        if (!strcmp(pName, "vkCmdBindIndexBuffer2KHR")) {
            return reinterpret_cast<PFN_vkVoidFunction>(&vkCmdBindIndexBuffer2Adapter);
        }
        if (!strcmp(pName, "vkCmdWriteBufferMarkerAMD") || !strcmp(pName, "vkCmdWriteBufferMarker2AMD")) {
            return reinterpret_cast<PFN_vkVoidFunction>(&Stub_NoopCmd);
        }
        const char* aliases[][2] = {
            { "vkCmdBeginRenderingKHR", "vkCmdBeginRendering" },
            { "vkCmdEndRenderingKHR", "vkCmdEndRendering" },
            { "vkCmdPipelineBarrier2KHR", "vkCmdPipelineBarrier2" },
            { "vkCmdWriteTimestamp2KHR", "vkCmdWriteTimestamp2" },
            { "vkQueueSubmit2KHR", "vkQueueSubmit2" },
            { "vkGetDeviceBufferMemoryRequirementsKHR", "vkGetDeviceBufferMemoryRequirements" },
            { "vkGetDeviceImageMemoryRequirementsKHR", "vkGetDeviceImageMemoryRequirements" },
            { "vkGetDeviceImageSparseMemoryRequirementsKHR", "vkGetDeviceImageSparseMemoryRequirements" },
            { "vkCmdDrawIndirectCountKHR", "vkCmdDrawIndirectCount" },
            { "vkCmdDrawIndexedIndirectCountKHR", "vkCmdDrawIndexedIndirectCount" },
            // [FIX] Недостающие алиасы, из-за которых GDPA возвращал NULL -> EXEC@0 на +0xC84B65:
            { "vkBindBufferMemory2KHR", "vkBindBufferMemory2" },
            { "vkBindImageMemory2KHR", "vkBindImageMemory2" },
            { "vkCreateRenderPass2KHR", "vkCreateRenderPass2" },
            { "vkCmdBeginRenderPass2KHR", "vkCmdBeginRenderPass2" },
            { "vkCmdNextSubpass2KHR", "vkCmdNextSubpass2" },
            { "vkCmdEndRenderPass2KHR", "vkCmdEndRenderPass2" },
            { "vkResetQueryPoolEXT", "vkResetQueryPool" },
            { "vkGetSemaphoreCounterValueKHR", "vkGetSemaphoreCounterValue" },
            { "vkWaitSemaphoresKHR", "vkWaitSemaphores" },
            { "vkSignalSemaphoreKHR", "vkSignalSemaphore" },
            { "vkGetBufferOpaqueCaptureAddressKHR", "vkGetBufferOpaqueCaptureAddress" },
            { "vkGetDeviceMemoryOpaqueCaptureAddressKHR", "vkGetDeviceMemoryOpaqueCaptureAddress" },
            { "vkGetDescriptorSetLayoutSupportKHR", "vkGetDescriptorSetLayoutSupport" },
            { "vkCmdBindVertexBuffers2EXT", "vkCmdBindVertexBuffers2" },
            { "vkCmdSetDeviceMaskKHR", "vkCmdSetDeviceMask" },
            { "vkCreatePrivateDataSlotEXT", "vkCreatePrivateDataSlot" },
            { "vkDestroyPrivateDataSlotEXT", "vkDestroyPrivateDataSlot" },
            { "vkSetPrivateDataEXT", "vkSetPrivateData" },
            { "vkGetPrivateDataEXT", "vkGetPrivateData" },
            { nullptr, nullptr }
        };
        for (int i = 0; aliases[i][0]; ++i) {
            if (!strcmp(pName, aliases[i][0])) {
                result = stub::Find(aliases[i][1]);
                if (!result) result = LocalOverride(aliases[i][1]);
                if (!result && g_thisModule)
                    result = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_thisModule, aliases[i][1]));
                if (!result && g_realLoader)
                    result = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_realLoader, aliases[i][1]));
                if (!result)
                    result = g_realGdpa(device, aliases[i][1]);
                if (result && FirstTime(std::string("gdpa-alias:") + pName))
                    LogLine("PROXY served via alias %s -> %s", pName, aliases[i][1]);
                break;
            }
        }
    }
    if (!result) result = ResolveBySuffixStrip(device, pName);
    // [v3.2] VK_EXT_debug_marker is not offered by this driver, yet the game fetches these and the log shows it calling
    // null pointers (EXEC@0). They are pure debug helpers, so a harmless stub cannot change rendering.
    if (!result && !strncmp(pName, "vkDebugMarkerSet", 16)) {
        if (FirstTime(std::string("dbgmarker:") + pName)) LogLine("[v3.2] %s -> stub returning VK_SUCCESS", pName);
        return reinterpret_cast<PFN_vkVoidFunction>(&Stub_ReturnSuccess);
    }
    if (!result && !strncmp(pName, "vkCmdDebugMarker", 16)) {
        if (FirstTime(std::string("dbgmarker:") + pName)) LogLine("[v3.2] %s -> no-op stub", pName);
        return reinterpret_cast<PFN_vkVoidFunction>(&Stub_NoopCmd);
    }
    if (!result && !strncmp(pName, "vkCmdSet", 8)) {
    if (FirstTime(std::string("noop-set:") + pName))
        LogLine("!!! MISSING dynamic-state %s -> NO-OP (state NOT applied!)", pName);
    return reinterpret_cast<PFN_vkVoidFunction>(&Stub_NoopCmd);
}
if (!result && FirstTime(std::string("gdpa-null:") + pName))
        LogLine("GDPA returned NULL for %s (real driver lacks it)", pName);
    return result;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_DETACH && reserved != nullptr) {
        // Process is terminating: show what the game was doing last.
        LogLine("DLL_PROCESS_DETACH: process is exiting. Last API breadcrumbs:");
        const LONG end = g_crumbIdx;
        for (int k = kCrumbCount - 16; k < kCrumbCount; ++k) {
            const char* c = g_crumbs[(end + k) & (kCrumbCount - 1)];
            if (c[0]) LogLine("  %s", c);
        }
    }
    if (reason == DLL_PROCESS_ATTACH) {
        g_thisModule = instance;
        DisableThreadLibraryCalls(instance);
        InstallCrashHandlerFirst();
    }
    return TRUE;
}
// ===========================================================================
// [RTFILL v1] Заполнение выходов ray tracing на GPU без RT (R7 260X).
// Настройка без перекомпиляции — VulkanProxy.ini, секция [rt]:
//   Fill=1  FillMode=0  Ambient=25  EveryFrame=1  Log=1
// ===========================================================================
namespace rtfill {
    static SRWLOCK s_lock = SRWLOCK_INIT;
    static volatile LONG s_init = 0;
    static volatile LONG s_enabled = 1;
    static volatile LONG s_everyFrame = 1;
    static volatile LONG s_log = 1;
    static float s_clear[4] = { 0.05f, 0.05f, 0.056f, 1.0f };

    struct ViewInfo {
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;
        VkImageLayout declared;
    };
    static std::unordered_map<uint64_t, ViewInfo>* s_views = nullptr;
    static std::unordered_map<uint64_t, std::vector<uint64_t>>* s_sets = nullptr;
    static std::unordered_map<uint64_t, std::vector<uint64_t>>* s_cbViews = nullptr;
    static std::unordered_map<uint64_t, VkImageLayout>* s_layouts = nullptr;
    static std::unordered_set<uint64_t>* s_filled = nullptr;

    static inline uint64_t K(const void* h) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h)); }

    static void IniPath(wchar_t* out, size_t cap) {
        out[0] = 0;
        HMODULE self = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&IniPath), &self);
        wchar_t path[2048]{};
        const DWORD n = self ? GetModuleFileNameW(self, path, 2048) : 0;
        if (n == 0 || n >= 2048) { wcscpy_s(out, cap, L"VulkanProxy.ini"); return; }
        wchar_t* slash = wcsrchr(path, L'\\');
        if (slash) *(slash + 1) = 0;
        wcscpy_s(out, cap, path);
        wcscat_s(out, cap, L"VulkanProxy.ini");
    }

    static void InitOnce() {
        if (InterlockedCompareExchange(&s_init, 1, 0) != 0) return;
        wchar_t ini[2048]{};
        IniPath(ini, 2048);
        s_enabled = GetPrivateProfileIntW(L"rt", L"Fill", 1, ini) ? 1 : 0;
        const int mode = GetPrivateProfileIntW(L"rt", L"FillMode", 0, ini);
        int amb = GetPrivateProfileIntW(L"rt", L"Ambient", 25, ini);
        if (amb < 0) amb = 0;
        if (amb > 100) amb = 100;
        s_everyFrame = GetPrivateProfileIntW(L"rt", L"EveryFrame", 1, ini) ? 1 : 0;
        s_log = GetPrivateProfileIntW(L"rt", L"Log", 1, ini) ? 1 : 0;
        const float v = (static_cast<float>(amb) / 100.0f) * 0.2f;
        if (mode == 1)      { s_clear[0] = 0.0f;        s_clear[1] = 0.0f;        s_clear[2] = 0.0f; }
        else if (mode == 2) { s_clear[0] = v * 0.65f;   s_clear[1] = v * 0.85f;   s_clear[2] = v * 1.5f; }
        else                { s_clear[0] = v;           s_clear[1] = v;           s_clear[2] = v * 1.12f; }
        s_clear[3] = 1.0f;
        LogLine("RTFILL: init Fill=%d mode=%d ambient=%.4f everyFrame=%d log=%d",
            (int)s_enabled, mode, s_clear[0], (int)s_everyFrame, (int)s_log);
    }

    static void EnsureMaps() {
        if (!s_views)   s_views = new (std::nothrow) std::unordered_map<uint64_t, ViewInfo>();
        if (!s_sets)    s_sets = new (std::nothrow) std::unordered_map<uint64_t, std::vector<uint64_t>>();
        if (!s_cbViews) s_cbViews = new (std::nothrow) std::unordered_map<uint64_t, std::vector<uint64_t>>();
        if (!s_layouts) s_layouts = new (std::nothrow) std::unordered_map<uint64_t, VkImageLayout>();
        if (!s_filled)  s_filled = new (std::nothrow) std::unordered_set<uint64_t>();
    }

    static bool IsClearColorFormat(VkFormat f) {
        switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
            return true;
        default:
            return false;
        }
    }

    void TrackView(VkImageView view, const VkImageViewCreateInfo* ci) {
        InitOnce();
        if (!s_enabled || !view || !ci || IsFake(view)) return;
        AcquireSRWLockExclusive(&s_lock);
        EnsureMaps();
        if (s_views) {
            ViewInfo vi{};
            vi.image = ci->image;
            vi.format = ci->format;
            vi.range = ci->subresourceRange;
            if (vi.range.aspectMask == 0) vi.range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.declared = VK_IMAGE_LAYOUT_GENERAL;
            (*s_views)[K(view)] = vi;
        }
        ReleaseSRWLockExclusive(&s_lock);
    }

    void TrackWrites(uint32_t count, const VkWriteDescriptorSet* writes) {
        InitOnce();
        if (!s_enabled || !writes) return;
        AcquireSRWLockExclusive(&s_lock);
        EnsureMaps();
        if (!s_sets || !s_views || !s_layouts) { ReleaseSRWLockExclusive(&s_lock); return; }
        for (uint32_t i = 0; i < count; ++i) {
            const VkWriteDescriptorSet& w = writes[i];
            if (w.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || !w.pImageInfo || !w.dstSet) continue;
            auto& dst = (*s_sets)[K(w.dstSet)];
            for (uint32_t k = 0; k < w.descriptorCount; ++k) {
                const VkImageView v = w.pImageInfo[k].imageView;
                if (!v || IsFake(v)) continue;
                if (dst.size() < 256) dst.push_back(K(v));
                auto it = s_views->find(K(v));
                if (it != s_views->end()) {
                    it->second.declared = w.pImageInfo[k].imageLayout;
                    (*s_layouts)[K(it->second.image)] = w.pImageInfo[k].imageLayout;
                }
            }
        }
        ReleaseSRWLockExclusive(&s_lock);
    }

    void TrackBind(VkCommandBuffer cb, uint32_t firstSet, uint32_t count, const VkDescriptorSet* sets) {
        InitOnce();
        if (!s_enabled || !cb || !sets) return;
        AcquireSRWLockExclusive(&s_lock);
        EnsureMaps();
        if (s_cbViews && s_sets) {
            auto& dst = (*s_cbViews)[K(cb)];
            if (firstSet == 0) dst.clear();
            for (uint32_t i = 0; i < count; ++i) {
                auto it = s_sets->find(K(sets[i]));
                if (it == s_sets->end()) continue;
                for (uint64_t v : it->second) {
                    if (dst.size() >= 128) break;
                    dst.push_back(v);
                }
            }
        }
        ReleaseSRWLockExclusive(&s_lock);
    }

    void TrackBarriers(uint32_t count, const VkImageMemoryBarrier* barriers) {
        if (!s_init || !s_enabled || !barriers) return;
        AcquireSRWLockExclusive(&s_lock);
        if (s_layouts) {
            for (uint32_t i = 0; i < count; ++i) {
                if (barriers[i].image && !IsFake(barriers[i].image))
                    (*s_layouts)[K(barriers[i].image)] = barriers[i].newLayout;
            }
        }
        ReleaseSRWLockExclusive(&s_lock);
    }

    void OnTraceRays(VkCommandBuffer cb, uint32_t width, uint32_t height) {
        InitOnce();
        if (!s_enabled || !cb) return;
        std::vector<uint64_t> views;
        AcquireSRWLockShared(&s_lock);
        if (s_cbViews) {
            auto it = s_cbViews->find(K(cb));
            if (it != s_cbViews->end()) views = it->second;
        }
        ReleaseSRWLockShared(&s_lock);
        if (views.empty()) return;

        auto clear = RealExport<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
        auto barrier = RealExport<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        if (!clear || !barrier) {
            if (FirstTime("rtfill:nofn")) LogLine("RTFILL: driver missing vkCmdClearColorImage/vkCmdPipelineBarrier");
            return;
        }

        VkClearColorValue col{};
        col.float32[0] = s_clear[0]; col.float32[1] = s_clear[1];
        col.float32[2] = s_clear[2]; col.float32[3] = s_clear[3];

        uint32_t done = 0;
        for (uint64_t vk : views) {
            ViewInfo vi{};
            VkImageLayout cur = VK_IMAGE_LAYOUT_GENERAL;
            bool skip = false;
            AcquireSRWLockShared(&s_lock);
            if (!s_views) skip = true;
            else {
                auto it = s_views->find(vk);
                if (it == s_views->end()) skip = true;
                else {
                    vi = it->second;
                    cur = vi.declared;
                    if (s_layouts) { auto l = s_layouts->find(K(vi.image)); if (l != s_layouts->end()) cur = l->second; }
                    if (!s_everyFrame && s_filled && s_filled->count(K(vi.image))) skip = true;
                }
            }
            ReleaseSRWLockShared(&s_lock);
            if (skip || !vi.image || IsFake(vi.image)) continue;
            if (!IsClearColorFormat(vi.format)) continue;
            if (cur != VK_IMAGE_LAYOUT_GENERAL && cur != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                cur != VK_IMAGE_LAYOUT_UNDEFINED && cur != VK_IMAGE_LAYOUT_PREINITIALIZED) continue;

            VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = cur;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = vi.image;
            b.subresourceRange = vi.range;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

            clear(cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &col, 1, &vi.range);

            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = (cur == VK_IMAGE_LAYOUT_UNDEFINED || cur == VK_IMAGE_LAYOUT_PREINITIALIZED)
                ? VK_IMAGE_LAYOUT_GENERAL : cur;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);

            AcquireSRWLockExclusive(&s_lock);
            if (s_filled) s_filled->insert(K(vi.image));
            if (s_layouts) (*s_layouts)[K(vi.image)] = b.newLayout;
            ReleaseSRWLockExclusive(&s_lock);
            ++done;
        }

        if (s_log && done) {
            static volatile LONG n = 0;
            const LONG k = InterlockedIncrement(&n);
            if (k <= 40 || k % 600 == 0)
                LogLine("RTFILL: TraceRays %ux%u -> cleared %u RT output(s) (ambient=%.4f), total=%ld",
                    width, height, done, s_clear[0], (long)k);
        }
    }
} // namespace rtfill
