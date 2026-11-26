#pragma once
// Minimal admission checks, not a substitute for spirv-val or GPU testing.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace optcompute {
inline uint64_t Hash(const uint32_t* code, size_t words) {
    uint64_t h = 0xcbf29ce484222325ull;
    // SPIR-V files are little-endian; Windows Vulkan targets here are little-endian.
    const auto* p = reinterpret_cast<const uint8_t*>(code);
    for (size_t i = 0; i < words * sizeof(uint32_t); ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
inline bool EntryPoints(const uint32_t* code, size_t words, std::vector<std::string>& names) {
    if (!code || words < 5 || code[0] != 0x07230203u || code[4] != 0) return false;
    for (size_t i = 5; i < words;) {
        const uint32_t len = code[i] >> 16, op = code[i] & 0xffffu;
        if (!len || len > words - i) return false;
        if (op == 15) { // OpEntryPoint, execution model, id, NUL-terminated name
            if (len < 4 || code[i + 1] != 5) return false; // only compute entry points
            std::string name; bool ended = false;
            for (size_t j = i + 3; j < i + len && !ended; ++j)
                for (unsigned shift = 0; shift < 32; shift += 8) {
                    char c = static_cast<char>((code[j] >> shift) & 255u);
                    if (!c) { ended = true; break; }
                    name.push_back(c);
                }
            if (!ended || name.empty()) return false;
            names.push_back(name);
        }
        i += len;
    }
    std::sort(names.begin(), names.end());
    return !names.empty();
}
inline bool Accept(const uint32_t* source, size_t sourceWords, const uint32_t* candidate,
                   size_t candidateWords, size_t limit) {
    if (!source || !candidate || sourceWords < 5 || candidateWords < 5 || limit == 0 ||
        candidateWords >= sourceWords || candidateWords >= limit || source[1] != candidate[1]) return false;
    std::vector<std::string> a, b;
    return EntryPoints(source, sourceWords, a) && EntryPoints(candidate, candidateWords, b) && a == b;
}
}
