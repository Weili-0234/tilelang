# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang is a Pythonic DSL for writing high-performance GPU/CPU kernels, built on TVM. Users write kernels using Python decorators (`@tilelang.jit`) and TileLang compiles them to optimized native code for NVIDIA CUDA, AMD ROCm, Apple Metal, CPU, and WebGPU backends.

## Build & Install

**Editable install (development):**
```bash
python3 -m pip install --no-build-isolation --verbose --editable .
```

**Prerequisites:** Clone with `--recurse-submodules` (TVM, CUTLASS, Composable Kernel are vendored in `3rdparty/`). The build uses CMake (>=3.26) + scikit-build-core with C++17. ccache/sccache are auto-detected.

**Alternative (PYTHONPATH-based):** Build C++ extensions manually then set PYTHONPATH. See https://tilelang.com/get_started/Installation.html#working-from-source-via-pythonpath

## Testing

```bash
# Run all tests
python3 -m pytest testing

# Run a specific test file
python3 -m pytest testing/python/kernel/test_gemm.py

# Run a specific test
python3 -m pytest testing/python/kernel/test_gemm.py::test_name -v
```

Tests are in `testing/python/` organized by subsystem (cuda, amd, metal, cpu, jit, language, kernel, transform, etc.). The conftest seeds random/torch/numpy with 0 for determinism and errors if all tests are skipped.

## Linting & Formatting

```bash
# Run all lint checks (ruff, clang-format, codespell, pymarkdown)
pre-commit run --all-files

# Format changed files vs origin/main (default)
bash format.sh

# Format all files
bash format.sh --all

# Format specific files
bash format.sh --files path/to/file.py
```

**Setup pre-commit hooks:** `pre-commit install --install-hooks`

**Python:** ruff (line-length=140, double quotes, rules: E/W/F/UP/FA/B/SIM). Config in `pyproject.toml [tool.ruff]`.
**C++:** clang-format. Config in `.clang-format`.
**Spelling:** codespell with wordlist at `docs/spelling_wordlist.txt`.

## Architecture

### Compilation Pipeline

```
@tilelang.jit decorated function
    → Language DSL (tilelang/language/) parses Python function body
    → AST/IR construction (tilelang/language/ast/, tilelang/language/parser/)
    → Engine lowering to TVM IR (tilelang/engine/lower.py, phase.py)
    → Transform passes (tilelang/transform/)
    → TVM codegen (CUDA/ROCm/Metal/CPU/WebGPU)
    → JIT compilation via Cython adapter (tilelang/jit/adapter/)
    → Kernel execution on target device
```

### Key Modules

- **`tilelang/jit/`** — JIT compilation entry point. `jit()` decorator, `compile()`, `par_compile()`. `kernel.py` manages compilation, `adapter/` contains backend-specific compilation (Cython wrappers).
- **`tilelang/language/`** — The DSL implementation. `builtin.py` (largest file) defines tile-level operations (T.gemm, T.copy, T.reduce, etc.). `allocate.py` for shared/local memory. `loop.py` for loop constructs. `parser/` converts Python AST to TileLang IR.
- **`tilelang/engine/`** — Lowers TileLang IR to TVM. `lower.py` is the main entry. `phase.py` defines compilation phases. `callback.py` for hooks.
- **`tilelang/transform/`** — IR optimization passes (buffer store wrapping, type cast decoupling, simplification, etc.).
- **`tilelang/layout/`** — Memory layout system. `Layout` and `Fragment` classes for tile memory mapping. `swizzle.py` for L2 cache optimization.
- **`tilelang/tileop/`** — High-level tile operation abstractions (GEMM, sparse GEMM).
- **`src/`** — C++ implementations registered via TVM FFI. Layout computation, copy/fill/atomic operations, IR transformations.
- **`tilelang/env.py`** — Environment detection (CUDA, ROCm, Metal). Controls light import mode and caching.

### Key Patterns

- **Lazy loading:** Heavy imports (torch, TVM) are deferred. `env.is_light_import()` gates heavy initialization in `__init__.py`.
- **TVM FFI:** C++ functions in `src/` are registered with TVM's FFI system and called from Python via `_ffi_api.py` files.
- **Kernel caching:** Compiled kernels are cached to avoid recompilation (`tilelang/cache/`).
- **DSL convention:** Kernel functions use `T.` prefix for tile operations (e.g., `T.gemm()`, `T.copy()`, `T.alloc_shared()`, `T.Pipelined()`). Buffer parameters are declared as type annotations on the kernel function.

### Public API

The main user-facing API is exported from `tilelang/__init__.py`:
- `tilelang.jit` — JIT decorator for kernel functions
- `tilelang.compile` — Compile with specific shapes
- `tilelang.par_compile` — Parallel compilation
- `tilelang.autotune` — Auto-tuning
- `tilelang.Profiler` — Performance profiling
- `tilelang.Layout`, `tilelang.Fragment` — Memory layout types
