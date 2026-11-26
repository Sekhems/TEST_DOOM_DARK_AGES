// swrt_rq.h - SPIR-V rewriter: replaces OpRayQuery* (SPV_KHR_ray_query) by calls into the software traversal library.
// Included INSIDE namespace spvrw (needs KindsFor / ForEachId). No OS dependencies.
//   * OpTypeRayQueryKHR  -> struct RQ (from the library), same result id
//   * AS descriptor      -> read-only StorageBuffer (the TLAS blob); the TLAS address is read from its header
//   * OpRayQueryInitialize/Proceed/Get* -> OpFunctionCall into library functions merged into the module
// Returns false (and leaves `out` untouched) for shapes it does not understand, so the caller can fall back.
#include "swrt_rq_spv.h"
#include <map>
#include <set>
#include <unordered_map>
#include <functional>

struct RqInst { uint32_t op; std::vector<uint32_t> w; };   // w[0] = (wordcount << 16) | op

inline bool RqParse(const uint32_t* in, size_t n, std::vector<RqInst>& v, std::string& err) {
    if (!in || n < 5 || in[0] != 0x07230203u) { err = "not spir-v"; return false; }
    for (size_t i = 5; i < n;) {
        uint32_t wc = in[i] >> 16, op = in[i] & 0xFFFF;
        if (wc == 0 || i + wc > n) { err = "bad instruction length"; return false; }
        RqInst x; x.op = op; x.w.assign(in + i, in + i + wc); v.push_back(std::move(x));
        i += wc;
    }
    return true;
}
inline RqInst RqMake(uint32_t op, std::initializer_list<uint32_t> ops) {
    RqInst x; x.op = op; x.w.push_back(((uint32_t)(ops.size() + 1) << 16) | op); x.w.insert(x.w.end(), ops.begin(), ops.end()); return x;
}
inline std::string RqStr(const std::vector<uint32_t>& w, size_t from) {
    std::string s; for (size_t i = from; i < w.size(); ++i) for (int b = 0; b < 4; ++b) { char c = (char)((w[i] >> (8 * b)) & 0xFF); if (!c) return s; s.push_back(c); }
    return s;
}
inline bool RqIsTypeSection(uint32_t op) {
    return (op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59 || op == 1 || op == 4472 || op == 5341 || op == 6;  // (6 never matches; placeholder for clarity)
}
inline bool RqIsDedupOp(uint32_t op) { return op == 19 || op == 20 || op == 21 || op == 22 || op == 23 || op == 24 || op == 32 || op == 33 || op == 41 || op == 42 || op == 43 || op == 44; }
inline bool RqDefinesId(uint32_t op, uint32_t& idx) {      // index of the result id word for type/constant instructions
    if ((op >= 19 && op <= 30) || op == 32 || op == 33 || op == 31) { idx = 1; return true; }
    if ((op >= 41 && op <= 52) || op == 59 || op == 1) { idx = 2; return true; }
    return false;
}

inline bool EmulateRayQuery(const uint32_t* in, size_t n, std::vector<uint32_t>& out, std::string& err) {
    std::vector<RqInst> T, L;
    if (!RqParse(in, n, T, err)) return false;
    if (!RqParse(swrt_rq_lib_spv, sizeof(swrt_rq_lib_spv) / 4, L, err)) { err = "lib parse"; return false; }
    uint32_t bound = in[3];

    // ---- 1. find RQ / AS types and their dependants in the target -----------------------------------------------------
    uint32_t rqType = 0, asType = 0;
    for (auto& I : T) { if (I.op == 4472) rqType = I.w[1]; if (I.op == 5341) asType = I.w[1]; }
    if (!rqType) { err = "no ray query"; return false; }
    std::unordered_set<uint32_t> rqSet{ rqType }, asSet;
    if (asType) asSet.insert(asType);
    std::unordered_map<uint32_t, uint32_t> ptrClass;      // pointer-to-RQ type id -> storage class
    std::unordered_set<uint32_t> movedIds;           // type-section instructions (by result id) to be re-emitted after the library types
    std::vector<uint32_t> asVars;
    auto uses = [&](const RqInst& I, const std::unordered_set<uint32_t>& s) {
        bool u = false; ForEachId(I.w.data(), I.op, (uint32_t)I.w.size(), [&](size_t p, char r) { if (r == 'I' && s.count(I.w[p])) u = true; if (r == 'T' && s.count(I.w[p])) u = true; });
        return u; };
    for (auto& I : T) {
        if (I.op == 59 && I.w.size() >= 4) {          // global variable
            if (rqSet.count(I.w[1])) { movedIds.insert(I.w[2]); continue; }   // Private ray query variable: re-emitted after the RQ struct
            if (asSet.count(I.w[1])) { asVars.push_back(I.w[2]); continue; }
        }
        uint32_t idx; if (!RqDefinesId(I.op, idx) || I.op == 59) continue;
        const uint32_t id = I.w[idx];
        if (I.op == 32 && I.w.size() >= 4) {          // OpTypePointer
            if (rqSet.count(I.w[3])) { if (I.w[2] != 7 && I.w[2] != 6) { err = "pointer to ray query not in Function/Private storage"; return false; } rqSet.insert(id); movedIds.insert(id); ptrClass[id] = I.w[2]; }
            else if (asSet.count(I.w[3])) asSet.insert(id);
        } else if (I.op == 28 || I.op == 29) {
            if (rqSet.count(I.w[2]) ) { rqSet.insert(id); movedIds.insert(id); }
            else if (asSet.count(I.w[2])) { err = "array of acceleration structures"; return false; }
        } else if (I.op == 30 || I.op == 33) {
            bool r = false, a = false;
            for (size_t k = 2; k < I.w.size(); ++k) { if (rqSet.count(I.w[k])) r = true; if (asSet.count(I.w[k])) a = true; }
            if (a) { err = "AS inside struct/function type"; return false; }
            if (r) { rqSet.insert(id); movedIds.insert(id); }
        }
    }
    movedIds.insert(rqType);
    if (asVars.empty() && asType) { /* AS type but no variable: unusual */ }

    // typeOf for every id with a result type (needed for operand adaptation)
    std::unordered_map<uint32_t, uint32_t> typeOf;
    for (auto& I : T) { size_t t = 0, r = 0; ForEachId(I.w.data(), I.op, (uint32_t)I.w.size(), [&](size_t p, char role) { if (role == 'T') t = p; if (role == 'R') r = p; }); if (t && r) typeOf[I.w[r]] = I.w[t]; }

    // ---- 2. index the library -------------------------------------------------------------------------------------------
    std::unordered_map<uint32_t, size_t> lDef;                  // lib id -> instruction index (types / consts / functions)
    std::unordered_map<std::string, uint32_t> lFn;              // api name -> lib function id
    std::unordered_map<uint32_t, std::string> lName; uint32_t lRq = 0, lGlsl = 0;
    struct FnRange { size_t b, e; }; std::unordered_map<uint32_t, FnRange> lFnRange;
    for (size_t i = 0; i < L.size(); ++i) {
        const RqInst& I = L[i]; uint32_t idx;
        if (I.op == 5 && I.w.size() >= 3) {
            std::string nm = RqStr(I.w, 2); lName[I.w[1]] = nm;
            size_t par = nm.find('('); if (par != std::string::npos) nm = nm.substr(0, par);
            if (nm.compare(0, 3, "rq_") == 0) lFn[nm] = I.w[1];
            if (nm == "RQ") lRq = I.w[1];
        }
        if (I.op == 11) lGlsl = I.w[1];
        if (RqDefinesId(I.op, idx) && I.op != 59) lDef[I.w[idx]] = i;
        if (I.op == 54) { size_t e = i; while (L[e].op != 56) ++e; lFnRange[I.w[2]] = { i, e }; lDef[I.w[2]] = i; i = e; }
    }
    // the function-name table above only has pre-"(" part; some functions (swrt_*) are reached through calls only
    if (!lRq) { err = "lib: RQ struct not found"; return false; }

    // ---- 3. which API functions are needed ----------------------------------------------------------------------------
    struct ApiOp { uint32_t op; const char* fn; };
    static const ApiOp kApi[] = {
        {4474,"rq_terminate"},{4475,"rq_noop"},{4476,"rq_noop"},{4477,"rq_proceed"},{4479,"rq_type"},{6016,"rq_tmin"},{6017,"rq_flags"},
        {6018,"rq_t"},{6019,"rq_custom"},{6020,"rq_instid"},{6021,"rq_sbt"},{6022,"rq_geom"},{6023,"rq_prim"},{6024,"rq_bary"},{6025,"rq_front"},
        {6026,"rq_aabbopaque"},{6027,"rq_od"},{6028,"rq_oo"},{6029,"rq_wd"},{6030,"rq_wo"},{6031,"rq_o2w"},{6032,"rq_w2o"} };
    auto apiFor = [&](uint32_t op) -> const char* { for (auto& a : kApi) if (a.op == op) return a.fn; return nullptr; };
    std::set<std::string> needed;
    for (auto& I : T) {
        if (I.op == 4473) needed.insert("rq_begin");
        else if (I.op >= 4473 && I.op <= 4496 || I.op >= 6016 && I.op <= 6032) {
            const char* f = apiFor(I.op); if (!f && I.op != 4473) { err = "unsupported ray query op " + std::to_string(I.op); return false; }
            if (f) needed.insert(f);
        }
    }
    if (needed.empty()) { err = "no ray query ops"; return false; }
    // library function closure through OpFunctionCall
    std::set<uint32_t> fnClosure; std::vector<uint32_t> work;
    for (auto& s : needed) { auto it = lFn.find(s); if (it == lFn.end()) { err = "lib fn missing " + s; return false; } work.push_back(it->second); }
    while (!work.empty()) {
        uint32_t f = work.back(); work.pop_back();
        if (!fnClosure.insert(f).second) continue;
        FnRange r = lFnRange[f];
        for (size_t i = r.b; i <= r.e; ++i) if (L[i].op == 57) { uint32_t callee = L[i].w[3]; if (lFnRange.count(callee)) work.push_back(callee); }
    }
    // ids used by the needed function bodies -> needed types/constants (transitively)
    std::set<uint32_t> needTypes;
    std::function<void(uint32_t)> needDef = [&](uint32_t id) {
        auto it = lDef.find(id); if (it == lDef.end()) return;
        const RqInst& I = L[it->second]; if (I.op == 54) return;
        if (!needTypes.insert(id).second) return;
        ForEachId(I.w.data(), I.op, (uint32_t)I.w.size(), [&](size_t p, char r) { if (r != 'R') needDef(I.w[p]); });
    };
    for (uint32_t f : fnClosure) { FnRange r = lFnRange[f];
        for (size_t i = r.b; i <= r.e; ++i) ForEachId(L[i].w.data(), L[i].op, (uint32_t)L[i].w.size(), [&](size_t p, char role) { if (role != 'R') needDef(L[i].w[p]); }); }
    needDef(lRq);

    // ---- 4. id mapping + pending type/constant/variable instructions --------------------------------------------------------
    std::unordered_map<uint32_t, uint32_t> mp;                  // lib id -> target id
    auto fresh = [&]() { return bound++; };
    auto M = [&](uint32_t lid) { auto it = mp.find(lid); if (it != mp.end()) return it->second; uint32_t v = fresh(); mp[lid] = v; return v; };
    mp[lRq] = rqType;
    // GLSL.std.450 import
    uint32_t glslId = 0; for (auto& I : T) if (I.op == 11 && RqStr(I.w, 2) == "GLSL.std.450") glslId = I.w[1];
    bool addGlslImport = false; if (!glslId) { glslId = fresh(); addGlslImport = true; }
    mp[lGlsl] = glslId;

    std::map<std::vector<uint32_t>, uint32_t> table;           // structural key -> target id (dedupable types / constants)
    auto keyOf = [&](const RqInst& I, uint32_t idx) { std::vector<uint32_t> k; k.push_back(I.op); for (size_t i = 1; i < I.w.size(); ++i) if (i != idx) k.push_back(I.w[i]); return k; };
    std::vector<RqInst> typeSecOrder;                           // original type-section instructions (kept in place unless moved)
    for (auto& I : T) { uint32_t idx; if (RqDefinesId(I.op, idx) && RqIsDedupOp(I.op)) table.emplace(keyOf(I, idx), I.w[idx]); }

    struct Pend { RqInst I; uint32_t id; };
    std::vector<Pend> pend; std::unordered_map<uint32_t, size_t> pendIdx;
    std::unordered_set<uint32_t> fwd;                           // forward-declared pointer ids
    std::vector<RqInst> newAnnot;
    auto addPend = [&](RqInst I, uint32_t id) { pendIdx[id] = pend.size(); pend.push_back({ std::move(I), id }); };
    // moved original instructions (keep ids)
    for (auto& I : T) { uint32_t idx; if (RqDefinesId(I.op, idx) && movedIds.count(I.w[idx]) && I.op != 4472) addPend(I, I.w[idx]); }
    // re-register the rq struct itself later through the library import (id = rqType)

    std::unordered_set<uint32_t> freshImported;                 // target ids of freshly imported (non-dedup) lib ids
    auto remapWords = [&](const RqInst& I, RqInst& o, bool defineFresh) {
        o = I;
        ForEachId(I.w.data(), I.op, (uint32_t)I.w.size(), [&](size_t p, char r) { (void)r; o.w[p] = M(I.w[p]); });
        (void)defineFresh;
    };
    for (size_t i = 0; i < L.size(); ++i) {
        const RqInst& I = L[i];
        if (I.op == 39) {            // OpTypeForwardPointer
            if (!needTypes.count(I.w[1])) continue;
            RqInst o = I; o.w[1] = M(I.w[1]); fwd.insert(o.w[1]); addPend(o, 0xFFFFFFFFu - (uint32_t)pend.size());
            freshImported.insert(o.w[1]); continue;
        }
        uint32_t idx; if (!RqDefinesId(I.op, idx) || I.op == 59 || !needTypes.count(I.w[idx])) continue;
        const uint32_t lid = I.w[idx];
        if (mp.count(lid) && lid != lRq && !fwd.count(mp[lid])) continue;
        RqInst o; remapWords(I, o, true);
        if (lid == lRq) { o.w[idx] = rqType; addPend(o, rqType); freshImported.insert(rqType); continue; }
        if (RqIsDedupOp(I.op) && !fwd.count(o.w[idx])) {
            // result id was just allocated by M(); undo that and try to find an existing equivalent
            uint32_t tmp = o.w[idx]; auto k = keyOf(o, idx);
            auto it = table.find(k);
            if (it != table.end()) { mp[lid] = it->second; continue; }
            table.emplace(k, tmp); addPend(o, tmp); continue;
        }
        freshImported.insert(o.w[idx]); addPend(o, o.w[idx]);
    }
    // decorations of freshly imported ids (struct Block/Offset, ArrayStride, AliasedPointer ...) - function-local ones are added below
    // ---- 5. helpers creating extra types / constants in the target ---------------------------------------------------------
    auto getType = [&](uint32_t op, std::vector<uint32_t> ops) -> uint32_t {      // ops excludes result id
        RqInst I; I.op = op; I.w.push_back(((uint32_t)(ops.size() + 2) << 16) | op); I.w.push_back(0); I.w.insert(I.w.end(), ops.begin(), ops.end());
        auto k = keyOf(I, 1); auto it = table.find(k); if (it != table.end()) return it->second;
        uint32_t id = fresh(); I.w[1] = id; table.emplace(k, id); addPend(I, id); return id; };
    auto getConst = [&](uint32_t type, uint32_t val) -> uint32_t {
        RqInst I; I.op = 43; I.w = { (4u << 16) | 43u, type, 0, val };
        auto k = keyOf(I, 2); auto it = table.find(k); if (it != table.end()) return it->second;
        uint32_t id = fresh(); I.w[2] = id; table.emplace(k, id); addPend(I, id); return id; };

    uint32_t tUint = getType(21, { 32, 0 });
    uint32_t tPtrFnRq = getType(32, { 7, rqType });
    std::vector<uint32_t> rqMember;                             // member types of struct RQ (target ids)
    for (auto& I : L) if (I.op == 30 && I.w[1] == lRq) for (size_t k = 2; k < I.w.size(); ++k) rqMember.push_back(M(I.w[k]));
    if (rqMember.size() != 10) { err = "lib: unexpected RQ layout"; return false; }
    uint32_t tV2Uint = rqMember[6];
    uint32_t voidT = 0; for (auto& I : L) if (I.op == 19) voidT = M(I.w[1]);
    std::vector<uint32_t> rqMemberPtr, cIdx;                     // created up front: late additions would miss the type section
    for (size_t k = 0; k < rqMember.size(); ++k) { rqMemberPtr.push_back(getType(32, { 7, rqMember[k] })); cIdx.push_back(getConst(tUint, (uint32_t)k)); }             // Function pointer to RQ (type of the per-function scratch copy)
    uint32_t tPtrSBuint = 0, c0 = 0, cLo = 0, cHi = 0;
    if (asType) {
        tPtrSBuint = getType(32, { 12, tUint });                 // StorageBuffer uint
        c0 = getConst(tUint, 0); cLo = getConst(tUint, 10); cHi = getConst(tUint, 11);
        uint32_t rta = getType(29, { tUint });
        // decorate (only if we created it fresh: harmless duplicates are avoided by the set below)
        static const uint32_t kArrayStride = 6, kBlock = 2, kOffset = 35, kNonWritable = 24;
        std::set<uint32_t> decorated;
        if (decorated.insert(rta).second) newAnnot.push_back(RqMake(71, { rta, kArrayStride, 4 }));
        uint32_t blk = fresh(); addPend(RqMake(30, { blk, rta }), blk);
        newAnnot.push_back(RqMake(71, { blk, kBlock }));
        newAnnot.push_back(RqMake(72, { blk, 0, kOffset, 0 }));
        newAnnot.push_back(RqMake(72, { blk, 0, kNonWritable }));
        uint32_t ptrBlk = fresh(); addPend(RqMake(32, { ptrBlk, 12, blk }), ptrBlk);
        for (uint32_t v : asVars) addPend(RqMake(59, { ptrBlk, v, 12 }), v);   // OpVariable %ptr %id StorageBuffer
    }

    // ---- 6. import library functions -----------------------------------------------------------------------------------------
    std::vector<RqInst> impFns;
    for (auto& I : L) {    // decorations of fresh ids that are types (before function ids get allocated lazily)
        if (I.op == 71 && mp.count(I.w[1]) && freshImported.count(mp[I.w[1]])) { RqInst o = I; o.w[1] = mp[I.w[1]]; newAnnot.push_back(o); }
        if (I.op == 72 && mp.count(I.w[1]) && freshImported.count(mp[I.w[1]])) { RqInst o = I; o.w[1] = mp[I.w[1]]; newAnnot.push_back(o); }
    }
    for (size_t i = 0; i < L.size(); ++i) {
        if (L[i].op != 54 || !fnClosure.count(L[i].w[2])) continue;
        size_t e = lFnRange[L[i].w[2]].e;
        for (size_t k = i; k <= e; ++k) { RqInst o; remapWords(L[k], o, false); impFns.push_back(std::move(o)); }
    }
    // function-local variable decorations (AliasedPointer etc.) for variables inside imported functions
    {
        std::unordered_set<uint32_t> localIds;
        for (auto& I : impFns) if (I.op == 59) localIds.insert(I.w[2]);
        for (auto& I : L) if (I.op == 71 && mp.count(I.w[1]) && localIds.count(mp[I.w[1]])) { RqInst o = I; o.w[1] = mp[I.w[1]]; newAnnot.push_back(o); }
    }
    // library function signatures (for operand adaptation)
    auto fnParamTypes = [&](const char* nm, std::vector<uint32_t>& pt, uint32_t& ret) {
        uint32_t lf = lFn[nm]; const RqInst& F = L[lFnRange[lf].b]; ret = M(F.w[1]);
        for (auto& I : L) if (I.op == 33 && I.w[1] == F.w[4]) { for (size_t k = 3; k < I.w.size(); ++k) pt.push_back(M(I.w[k])); }
        };

    // ---- 7. rewrite the target ------------------------------------------------------------------------------------------------
    std::vector<RqInst> O;
    std::unordered_map<uint32_t, uint32_t> loadVar;              // AS load id -> AS variable id
    std::unordered_set<uint32_t> removedTypeIds = asSet;         // ids that disappear
    bool havePsbCap = false, haveInt64 = false, havePsbExt = false;
    for (auto& I : T) { if (I.op == 17 && I.w[1] == 5347) havePsbCap = true; if (I.op == 17 && I.w[1] == 11) haveInt64 = true;
                        if (I.op == 10 && RqStr(I.w, 1) == "SPV_KHR_physical_storage_buffer") havePsbExt = true; }
    size_t firstFn = 0; for (size_t i = 0; i < T.size(); ++i) if (T[i].op == 54) { firstFn = i; break; }
    size_t firstType = firstFn; for (size_t i = 0; i < firstFn; ++i) { uint32_t idx; if ((RqDefinesId(T[i].op, idx)) || T[i].op == 4472 || T[i].op == 5341 || T[i].op == 39) { firstType = i; break; } }
    size_t lastCap = 0, lastExt = 0; for (size_t i = 0; i < firstFn; ++i) { if (T[i].op == 17) lastCap = i; if (T[i].op == 10) lastExt = i; }

    // emit the pending list in dependency order
    std::vector<RqInst> pendOut; std::unordered_set<uint32_t> done;
    std::function<void(uint32_t)> ensure = [&](uint32_t id) {
        auto it = pendIdx.find(id); if (it == pendIdx.end() || done.count(id)) return;
        done.insert(id);
        const RqInst& I = pend[it->second].I;
        if (!fwd.count(id)) ForEachId(I.w.data(), I.op, (uint32_t)I.w.size(), [&](size_t p, char r) { if (r != 'R') { uint32_t d = I.w[p]; if (!fwd.count(d)) ensure(d); } });
        pendOut.push_back(I);
    };
    for (auto& P : pend) { if (P.I.op == 39) { pendOut.push_back(P.I); done.insert(P.id); } }
    for (auto& P : pend) ensure(P.id);

    bool inFn = false;
    size_t fnInsertPos = 0; bool fnHasLabel = false; uint32_t fnTmp = 0;
    // emits the copy-in, the call and (for mutating calls) the copy-out. Returns the pointer to hand to the library.
    auto rqPtrFor = [&](uint32_t rq, bool copyIn) -> uint32_t {
        auto pt = typeOf.find(rq); bool priv = false;
        if (pt != typeOf.end()) { auto pc = ptrClass.find(pt->second); if (pc != ptrClass.end() && pc->second == 6) priv = true; }
        if (!priv) return rq;
        if (!fnTmp) fnTmp = fresh();
        if (copyIn) { uint32_t v = fresh(); O.push_back(RqMake(61, { rqType, v, rq })); O.push_back(RqMake(62, { fnTmp, v })); }
        return fnTmp;
    };
    auto copyOut = [&](uint32_t rq) {
        auto pt = typeOf.find(rq); bool priv = false;
        if (pt != typeOf.end()) { auto pc = ptrClass.find(pt->second); if (pc != ptrClass.end() && pc->second == 6) priv = true; }
        if (!priv) return;
        uint32_t v = fresh(); O.push_back(RqMake(61, { rqType, v, fnTmp })); O.push_back(RqMake(62, { rq, v }));
    };
    for (size_t idx = 0; idx < T.size(); ++idx) {
        const RqInst& I = T[idx];
        if (idx == firstType) for (auto& a : newAnnot) O.push_back(a);
        if (I.op == 17 && (I.w[1] == 4472 || I.w[1] == 4478)) continue;
        if (I.op == 10 && RqStr(I.w, 1) == "SPV_KHR_ray_query") continue;
        if (I.op == 14 && I.w[1] == 0) { RqInst m = I; m.w[1] = 5348; O.push_back(m); continue; }
        // names / decorations of removed ids
        if ((I.op == 5 || I.op == 6 || I.op == 71 || I.op == 72) && removedTypeIds.count(I.w[1])) continue;
        if (!inFn) {
            if (I.op == 54) {
                inFn = true;
                for (auto& p : pendOut) O.push_back(p);
            } else {
                uint32_t di;
                if (I.op == 4472 || I.op == 5341) continue;                       // RQ / AS types are re-created elsewhere
                if (RqDefinesId(I.op, di)) {
                    uint32_t id = I.w[di];
                    if (movedIds.count(id) || asSet.count(id)) continue;
                    if (I.op == 59 && std::find(asVars.begin(), asVars.end(), I.w[2]) != asVars.end()) continue;
                }
                O.push_back(I);
                continue;
            }
        }
        // ---- function bodies ----
        if (I.op == 54) { fnHasLabel = false; fnTmp = 0; }
        if (I.op == 56 && fnTmp) { O.insert(O.begin() + fnInsertPos, RqMake(59, { tPtrFnRq, fnTmp, 7 })); }
        if (I.op == 61 && I.w.size() >= 4 && asSet.count(I.w[1])) {          // OpLoad of an acceleration structure
            auto v = std::find(asVars.begin(), asVars.end(), I.w[3]);
            if (v == asVars.end()) { err = "AS load from unknown variable"; return false; }
            loadVar[I.w[2]] = I.w[3]; continue;
        }
        if (I.op == 4473) {                                                   // OpRayQueryInitializeKHR
            auto lv = loadVar.find(I.w[2]); if (lv == loadVar.end()) { err = "initialize: AS operand is not a plain load"; return false; }
            // TLAS address = first two words at word offset 10/11 of the blob (header: self address)
            uint32_t a0 = fresh(), lo = fresh(), a1 = fresh(), hi = fresh(), tl = fresh();
            O.push_back(RqMake(65, { tPtrSBuint, a0, lv->second, c0, cLo })); O.push_back(RqMake(61, { tUint, lo, a0 }));
            O.push_back(RqMake(65, { tPtrSBuint, a1, lv->second, c0, cHi })); O.push_back(RqMake(61, { tUint, hi, a1 }));
            O.push_back(RqMake(80, { tV2Uint, tl, lo, hi }));                  // OpCompositeConstruct uvec2
            uint32_t rq = rqPtrFor(I.w[1], false);
            // member stores: 0 origin, 1 tmin, 2 dir, 3 tmax, 4 flags, 5 mask, 6 tlas
            struct { int m; uint32_t val; } st[7] = { {0, I.w[5]}, {1, I.w[6]}, {2, I.w[7]}, {3, I.w[8]}, {4, I.w[3]}, {5, I.w[4]}, {6, tl} };
            for (auto& e : st) {
                uint32_t mt = rqMember[e.m], val = e.val;
                uint32_t have = typeOf.count(val) ? typeOf[val] : (e.m == 6 ? tV2Uint : 0);
                if (have != mt) { uint32_t b = fresh(); O.push_back(RqMake(124, { mt, b, val })); val = b; }
                uint32_t p = fresh(); O.push_back(RqMake(65, { rqMemberPtr[e.m], p, rq, cIdx[e.m] })); O.push_back(RqMake(62, { p, val }));
            }
            O.push_back(RqMake(57, { voidT, fresh(), mp[lFn["rq_begin"]], rq }));
            copyOut(I.w[1]); continue;
        }
        if (I.op >= 4474 && I.op <= 4479 || I.op >= 6016 && I.op <= 6032) {
            const char* f = apiFor(I.op); if (!f) { err = "unsupported ray query op"; return false; }
            std::vector<uint32_t> pt; uint32_t ret; fnParamTypes(f, pt, ret);
            const bool hasResult = (I.op == 4477 || I.op == 4479 || I.op >= 6016);
            const uint32_t rqOrig = hasResult ? I.w[3] : I.w[1];
            uint32_t rq = rqPtrFor(rqOrig, true);
            const bool mutating = (I.op == 4474 || I.op == 4475 || I.op == 4476 || I.op == 4477);
            if (hasResult) {
                uint32_t want = I.w[1], res = I.w[2];
                if (want == ret) { O.push_back(RqMake(57, { ret, res, mp[lFn[f]], rq })); }
                else { uint32_t tmp = fresh(); O.push_back(RqMake(57, { ret, tmp, mp[lFn[f]], rq })); O.push_back(RqMake(124, { want, res, tmp })); }
            } else O.push_back(RqMake(57, { ret, fresh(), mp[lFn[f]], rq }));
            if (mutating) copyOut(rqOrig);
            continue;
        }
        O.push_back(I);
        if (I.op == 248 && !fnHasLabel) { fnHasLabel = true; fnInsertPos = O.size(); }
    }
    for (auto& I : impFns) O.push_back(I);
    {   // capabilities / extension needed by the library (64-bit integers, physical storage buffer addressing)
        size_t posCap = 0; for (size_t i = 0; i < O.size(); ++i) if (O[i].op == 17) posCap = i + 1;
        std::vector<RqInst> add;
        if (!haveInt64) add.push_back(RqMake(17, { 11 }));
        if (!havePsbCap) add.push_back(RqMake(17, { 5347 }));
        O.insert(O.begin() + posCap, add.begin(), add.end());
        if (!havePsbExt) {
            size_t posExt = 0; for (size_t i = 0; i < O.size(); ++i) if (O[i].op == 10 || O[i].op == 17) posExt = i + 1;
            RqInst e; e.op = 10; std::string s = "SPV_KHR_physical_storage_buffer"; std::vector<uint32_t> ws(s.size() / 4 + 1, 0); memcpy(ws.data(), s.data(), s.size());
            e.w.push_back(((uint32_t)(ws.size() + 1) << 16) | 10); e.w.insert(e.w.end(), ws.begin(), ws.end()); O.insert(O.begin() + posExt, e);
        }
    }
    if (addGlslImport) { // OpExtInstImport must come before OpMemoryModel; insert after the last OpExtension
        size_t pos = 0; for (size_t i = 0; i < O.size(); ++i) if (O[i].op == 10 || O[i].op == 17) pos = i + 1;
        RqInst e; e.op = 11; std::string s = "GLSL.std.450"; std::vector<uint32_t> ws(s.size() / 4 + 1, 0); memcpy(ws.data(), s.data(), s.size());
        e.w.push_back(((uint32_t)(ws.size() + 2) << 16) | 11); e.w.push_back(glslId); e.w.insert(e.w.end(), ws.begin(), ws.end()); O.insert(O.begin() + pos, e);
    }
    // safety: no ray-query instructions may remain
    for (auto& I : O) if (I.op == 4472 || I.op == 5341 || (I.op >= 4473 && I.op <= 4479) || (I.op >= 6016 && I.op <= 6032)) { err = "leftover ray query instruction"; return false; }
    out.assign(in, in + 5); out[3] = bound;
    for (auto& I : O) out.insert(out.end(), I.w.begin(), I.w.end());
    return true;
}
