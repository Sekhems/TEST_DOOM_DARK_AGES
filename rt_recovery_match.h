// Classification of the two observed null indirect calls. Kept independent of
// Windows/Vulkan so rejection cases can be tested without a GPU or the game.
#pragma once
#include <cstdint>

namespace rt_recovery {
enum class Target { None, BuildSizes, CreateAS };
struct Evidence {
    bool enabled = false;
    bool spoof = false;
    bool sameDevice = false;
    bool gameCallIndirectNull = false;
    uint64_t arg2 = 0;
    uint32_t inputType = 0;
    uint32_t outputType = 0;
    bool argumentsReadable = false;
    bool outputWritable = false;
    bool validPayload = false;
};
inline Target Classify(const Evidence& e) {
    if (!e.enabled || !e.spoof || !e.sameDevice || !e.gameCallIndirectNull ||
        !e.argumentsReadable || !e.outputWritable || !e.validPayload) return Target::None;
    // vkGetAccelerationStructureBuildSizesKHR: DEVICE=1, geometry info, size info.
    if (e.arg2 == 1 && e.inputType == 1000150000u && e.outputType == 1000150020u)
        return Target::BuildSizes;
    // vkCreateAccelerationStructureKHR: allocator must also have been checked NULL.
    if (e.inputType == 1000150017u && e.outputType == 0)
        return Target::CreateAS;
    return Target::None;
}
}
