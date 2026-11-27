#pragma once
// Narrow, opt-in experiment. These structural checks do not replace spirv-val.
#include "optimized_compute.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace graphicscap49 {
inline bool AddForSelectedModule(const uint32_t* code, size_t words, uint32_t model,
                                 uint64_t selectedHash, bool featureEnabled,
                                 std::vector<uint32_t>& out) {
    out.clear();
    if (!featureEnabled || !selectedHash || !code || words < 5 ||
        code[0] != 0x07230203u || code[4] != 0 || (model != 0 && model != 4) ||
        optcompute::Hash(code, words) != selectedHash) return false;
    bool entry = false;
    for (size_t i = 5; i < words;) {
        const uint32_t len = code[i] >> 16, op = code[i] & 0xffffu;
        if (!len || len > words - i) return false;
        if (op == 17 && (len < 2 || code[i + 1] == 49)) return false;
        if (op == 15) {
            if (len < 4 || code[i + 1] != model) return false;
            entry = true;
        }
        i += len;
    }
    if (!entry) return false;
    out.assign(code, code + 5);
    out.push_back((2u << 16) | 17u); // OpCapability StorageImageExtendedFormats
    out.push_back(49);
    out.insert(out.end(), code + 5, code + words);
    return true;
}
}
