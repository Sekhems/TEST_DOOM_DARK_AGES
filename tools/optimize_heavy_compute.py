#!/usr/bin/env python3
"""Optimize ONE proxy-transformed compute SPIR-V module using Vulkan SDK tools.

The output is experimental. Validation and a shorter module do not prove that a
legacy driver will compile or execute it safely, nor that its rendering is right.
"""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def entries(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError("SPIR-V length is invalid")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    if words[0] != 0x07230203 or words[4] != 0:
        raise ValueError("SPIR-V header is invalid")
    names = []
    i = 5
    while i < len(words):
        length, op = words[i] >> 16, words[i] & 65535
        if not length or i + length > len(words):
            raise ValueError("SPIR-V instruction length is invalid")
        if op == 15:
            if length < 4 or words[i + 1] != 5:
                raise ValueError("only compute entry points are allowed")
            raw = data[(i + 3) * 4:(i + length) * 4]
            if b"\0" not in raw or not raw.split(b"\0", 1)[0]:
                raise ValueError("entry point name is invalid")
            names.append(raw.split(b"\0", 1)[0])
        i += length
    if not names:
        raise ValueError("compute entry point not found")
    return words[1], sorted(names)


def find_tool(name):
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for filename in (name + ".exe", name):
            p = Path(sdk) / "Bin" / filename
            if p.is_file():
                return str(p)
    return shutil.which(name) or shutil.which(name + ".exe")


def run(argv):
    print("+ " + " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="spv_heavy_<hash>_0.spv from the proxy")
    parser.add_argument("--limit", type=int, required=True, help="current HeavyComputeWords (e.g. 61000)")
    parser.add_argument("--output-dir", type=Path, help="default: spv_opt directory next to input")
    parser.add_argument("--scalar-block-layout", action="store_true", help="validate with scalarBlockLayout; requires it to be enabled on the target Vulkan device")
    parser.add_argument("--spirv-opt", help="path to spirv-opt from Vulkan SDK")
    parser.add_argument("--spirv-val", help="path to spirv-val from Vulkan SDK")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    opt = args.spirv_opt or find_tool("spirv-opt")
    val = args.spirv_val or find_tool("spirv-val")
    if not opt or not val:
        raise RuntimeError("spirv-opt and spirv-val are required (Vulkan SDK Bin or PATH)")
    data = args.source.read_bytes()
    version, names = entries(data)
    key = fnv1a64(data)
    out_dir = args.output_dir or args.source.parent / "spv_opt"
    dest = out_dir / f"spv_{key:016x}.spv"
    print(f"Input hash={key:016x}; words={len(data)//4}; limit={args.limit}", flush=True)
    if args.scalar_block_layout:
        print("WARNING: scalar block layout validation requires scalarBlockLayout to be enabled on the target device.", flush=True)
    if dest.exists():
        raise RuntimeError(f"Refusing to overwrite {dest}; move or remove the old candidate first")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spv_opt_") as tmp:
        candidate = Path(tmp) / "candidate.spv"
        layout = ["--scalar-block-layout"] if args.scalar_block_layout else []
        run([val, "--target-env", "vulkan1.2", *layout, args.source])
        run([opt, "--target-env=vulkan1.2", *layout, "--preserve-bindings", "--preserve-interface", "--preserve-spec-constants",
             "-Os", args.source, "-o", candidate])
        run([val, "--target-env", "vulkan1.2", *layout, candidate])
        optimized = candidate.read_bytes()
        out_version, out_names = entries(optimized)
        if out_version != version or out_names != names:
            raise RuntimeError("SPIR-V version or compute entry points changed; candidate not installed")
        if len(optimized) >= len(data) or len(optimized) // 4 >= args.limit:
            raise RuntimeError(f"Candidate is {len(optimized)//4} words; it must be smaller than input and below {args.limit}. Not installed.")
        # Exclusive creation prevents accidentally replacing a previously tested candidate.
        with dest.open("xb") as f:
            f.write(optimized)
    print(f"Candidate: {dest} ({len(optimized)//4} words). Set OptimizedComputeHash={key:016x} only for an explicit test.")
    print("WARNING: validation/size do not guarantee semantics, driver stability or GPU completion.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
