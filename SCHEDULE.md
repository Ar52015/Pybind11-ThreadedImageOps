# Phase 2: GIL-Free Threaded Image Operations

**The Validation Task**:

> To validate Phase 2, the developer must construct a hybrid Python-C++ module. Using Modern CMake, they will compile a Pybind11 shared object library (.so). The C++ library must implement a memory-safe RAII class that accepts a multi-dimensional NumPy tensor from Python. The C++ code must leverage multi-threading (via std::thread) to concurrently manipulate different quadrants of the image array in-place, bypassing the Python Global Interpreter Lock (GIL) entirely. The Python script will import this compiled library, pass a massive image payload, and verify that the transformation occurred instantaneously without any memory reallocation or copying overhead.

This phase proves that the developer can bridge the Python-C++ boundary at zero cost — no copies, no GIL contention, no safety violations — and exploit hardware parallelism directly from a Python caller.
**Goal**: Demonstrate mastery of native extension architecture: CMake-driven builds, RAII ownership semantics over foreign memory, GIL-free concurrency on shared mutable buffers, and empirical proof of zero-copy data flow.

## Design Decisions

**Decomposition strategy**: Inside-out — the project has a linear data flow (buffer wrapper -> transform kernels -> threaded dispatch -> Python bindings). Core domain logic first, then boundary adapters.
**Walking skeleton**: Day 3 — Python passes a NumPy array -> `ImageBuffer` wraps it zero-copy -> Python verifies `ImageBuffer.data_ptr() == ndarray.ctypes.data`. Proves the architecture works end-to-end: CMake builds, pybind11 links, buffer protocol extracts a raw pointer, zero-copy invariant holds.
**First-risk day**: Day 3 — the pybind11 buffer protocol binding (`py::array_t` -> `py::buffer_info` -> raw pointer extraction) is the riskiest unknown. If zero-copy doesn't hold here, the project's core invariant fails and everything downstream is wrong.
**Deferred concerns**: No networking, no persistence.
**Technology choices**:
- pybind11 over Cython — *selected for learning*; Cython generates C but doesn't teach C++ interop, RAII semantics, or GIL-release mechanics.
- std::thread over OpenMP — *selected for learning*; OpenMP is higher-level but hides the threading model this project aims to teach.
- CMake over Meson — *selected for learning*; Meson has simpler syntax but less industry adoption for C++ extension projects.
- GoogleTest over Catch2 — *selected for learning*; both are solid, GoogleTest has broader ecosystem support and a FetchContent quickstart.
**Architectural patterns**:
- Non-owning buffer view over owning wrapper — NumPy owns the memory, `ImageBuffer` manages access lifetime only. Enables zero-copy but requires careful RAII scoping (all threads must join before the view goes out of scope).
- Row-range partitioning over task-based parallelism — quadrants are non-overlapping row ranges, avoiding synchronization entirely. Simpler and sufficient for a fixed-partition workload; a thread pool adds complexity without benefit here.

## Preflight Checklist

Before starting Day 0, verify:

**Hardware & environment**:
- [ ] C++17-capable compiler installed (`g++ --version` >= 7 or `clang++ --version` >= 5)
- [ ] CMake >= 3.16 installed (`cmake --version`)
- [ ] Python >= 3.14 installed (`python3 --version`)
- [ ] `uv` installed (`uv --version`)
- [ ] Docker installed (`docker --version`)

---

## Day 0: Project Skeleton and Build Toolchain

**Focus**: Modern CMake, FetchContent, Pybind11 discovery, uv project init, linting, pre-commit hooks
**Load**: Level 2 — many tools to configure but each is a short config file
**Depends on**: none

- **Tasks**:
    - [ ] Create the top-level directory layout: `src/` for C++ sources, `include/` for headers, `python/` for the driver script and tests, and `build/` (gitignored) for CMake artifacts.
    - [ ] Initialize the Python project with `uv init` (generates `pyproject.toml`, `.python-version`, `uv.lock`). Add dependencies via `uv add numpy` and `uv add --dev pytest pytest-benchmark ruff mypy pre-commit`. Run `uv sync` to create the `.venv`.
        - Note: Pin the NumPy version explicitly in `pyproject.toml` (e.g. `numpy>=2.4,<3`) so the ABI stays stable across rebuilds. Commit `uv.lock` for reproducible installs.
    - [ ] Configure pre-commit hooks (`.pre-commit-config.yaml`):
        - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`
        - Local hooks: `ruff check --fix`, `ruff format`, `mypy .`, `pytest --benchmark-disable` (all via `uv run`)
        - C++ hooks: `clang-format -i`, `clang-tidy --fix`, `ctest --test-dir build --output-on-failure`
        - Commit-msg hook: `commitlint` or equivalent to enforce [Conventional Commits](https://www.conventionalcommits.org/) (`<type>(<scope>): <description>`) from the first commit.
        - Note: Run `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg` to activate both hook types. Hooks must pass before any commit lands.
    - [ ] Configure ruff in `pyproject.toml`: enable rule sets `E`, `F`, `I` (isort), `UP` (pyupgrade), `NPY` (NumPy-specific).
    - [ ] Configure strict mypy in `pyproject.toml` (`[tool.mypy]` with `strict = true`).
    - [ ] Add a `.clang-format` file for C++ source formatting (`BasedOnStyle: Google`).
    - [ ] Add a `.clang-tidy` config with `concurrency-*`, `google-*`, `performance-*`, `readability-*`, `bugprone-*`, `modernize-*` checks.
    - [ ] Author the root `CMakeLists.txt` — `cmake_minimum_required(VERSION 3.16)`, `CMAKE_CXX_STANDARD 17`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`, `FetchContent` for pybind11 (`v3.0.2`) and GoogleTest (`v1.17.0`).
        - Note: pybind11 and GoogleTest are both pulled via `FetchContent_Declare` + `FetchContent_MakeAvailable` — no git submodules, no system installs. The GoogleTest quickstart (see Resources) shows this exact pattern. For pybind11, use `pybind11_add_module` after `FetchContent_MakeAvailable(pybind11)`.
    - [ ] Add a minimal `src/module.cpp` with `PYBIND11_MODULE` entry point exposing a `noop()` function returning a string literal.
    - [ ] Add stub `src/transforms.cpp` and test files in `tests/` (`test_transform_quadrant.cpp`, `test_quadrant_partitioning.cpp`, `test_thread_safety.cpp`).
    - [ ] Verify the full build-import cycle: `cmake -B build && cmake --build build`, then `PYTHONPATH=build uv run python -c "import threaded_image_ops; print(threaded_image_ops.noop())"` prints `hello from C++`.
    - [ ] Symlink `compile_commands.json` to project root (`ln -s build/compile_commands.json .`) for clang-tidy LSP support.
    - [ ] Add an `.editorconfig` for cross-editor consistency (indent style, line endings, trailing whitespace, final newline).
    - [ ] Add `build/`, `*.so`, `__pycache__/`, `.venv/`, and `compile_commands.json` to `.gitignore`.

- **Resources**:
    - [CMake Tutorial — Step 1: Getting Started with CMake](https://cmake.org/cmake/help/latest/guide/tutorial/Getting%20Started%20with%20CMake.html) — Step-by-step intro to CMake. Read **Exercise 1** through **Exercise 3** for the four core commands (`cmake_minimum_required`, `project`, `add_library`, `target_link_libraries`). This is a Tier 5 technology — read the full exercises, don't skim.
    - [FetchContent — Modern CMake](https://cliutils.gitlab.io/modern-cmake/chapters/projects/fetch.html) — Clear, concise explanation of `FetchContent_Declare` + `FetchContent_MakeAvailable`. Shows the complete pattern with a real library. Read this *before* the official CMake FetchContent reference — it's shorter and more practical.
        - *If the above is unclear:* [FetchContent — CMake Official Docs](https://cmake.org/cmake/help/latest/module/FetchContent.html) — the authoritative reference, but dense. Use it as a lookup after reading the Modern CMake page.
    - [Quickstart: Building with CMake — GoogleTest](https://google.github.io/googletest/quickstart-cmake.html) — **Complete working CMakeLists.txt** using FetchContent to pull GoogleTest, create a test executable, and discover tests via `gtest_discover_tests()`. This is both your GoogleTest setup guide AND a working example of FetchContent composition. Read the full page — it's short.
    - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — CMake integration for pybind11. Focus on the **pybind11_add_module** section for how to define a Python extension target, and **find_package vs. add_subdirectory** for how pybind11 integrates with CMake. Note: this page does not cover FetchContent directly — the FetchContent pattern is the same as GoogleTest (declare + make available), then use `pybind11_add_module` as documented here.
    - [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) — Covers `PYBIND11_MODULE`, `m.def()`, and basic type conversions. Read the **Creating bindings for a simple function** section — it's the minimum needed to write `module.cpp`.
    - [pre-commit documentation](https://pre-commit.com/) — Official docs for the pre-commit hook framework. Read **Adding pre-commit plugins** and **Creating new hooks** for `.pre-commit-config.yaml` syntax and local hook definitions.
    - [Configuring Ruff](https://docs.astral.sh/ruff/configuration/) — Ruff linter/formatter configuration in `pyproject.toml`. Read **Rule selection** for how `select` and `ignore` work with rule code prefixes.
    - [mypy configuration file](https://mypy.readthedocs.io/en/stable/config_file.html) — mypy config reference. Read **The mypy configuration file** for `[tool.mypy]` in `pyproject.toml`.
    - [Clang-Format Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html) — Full reference for `.clang-format`. Skim **BasedOnStyle** for how predefined styles (Google, LLVM) work.
    - [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/) — clang-tidy overview. Read **Configuring Checks** for `.clang-tidy` YAML syntax; browse the [checks list](https://clang.llvm.org/extra/clang-tidy/checks/list.html) for what `bugprone-*`, `modernize-*`, etc. cover.

---

## Day 1: C++ Foundations I — Pointers, Memory, and Classes (Foundation)

**Focus**: Header files, raw pointers, pointer arithmetic, classes, constructors, destructors, RAII, const correctness, exceptions
**Load**: Level 3 — dense conceptual ground to cover, but exercises are scoped
**Depends on**: none
**Prepares for**: Day 3 — RAII Buffer Wrapper and Zero-Copy Binding

- **Objectives**:
    1. You can explain the C++ compilation model — why code is split into `.hpp` headers and `.cpp` source files, and what the compiler and linker each do.
    2. You can declare, dereference, and do arithmetic on raw pointers to walk through a contiguous byte buffer.
    3. You can write a class with a validating constructor (member initializer list), `const` getters, private data members, and a defaulted destructor.
    4. You can explain RAII — tying resource lifetime to object lifetime — and why `ImageBuffer`'s destructor will be `= default` (it manages access, not allocation).

- **Tasks**:

    **Compilation Model & Headers**
    - [ ] Read learncpp.com lessons **0.5** (compiler, linker, libraries), **2.11** (header files), and **2.12** (header guards / `#pragma once`).
    - [ ] Read **15.2** (classes and header files) — this explains why the class definition goes in `.hpp` and method implementations can go in `.cpp`.

    **Raw Pointers & Pointer Arithmetic**
    - [ ] Read learncpp.com lessons **12.7** (introduction to pointers), **12.9** (pointers and `const`), and **17.9** (pointer arithmetic and subscripting).
    - [ ] **Exercise 1**: Write a standalone program (outside the project) that heap-allocates a `uint8_t` array of 256 elements, walks it with pointer arithmetic to set each byte to its index value, and prints the results.
        - Expected output: `0 1 2 3 4 ... 254 255` (256 space-separated values). This mirrors what `transform_quadrant` will do in Day 5 — walking a buffer with raw pointer math.

    **Classes, Constructors, Destructors, RAII & Exceptions**
    - [ ] Read learncpp.com lessons **14.3** (member functions), **14.4** (`const` member functions), **14.5** (public/private access specifiers), **14.9** (constructors), **14.10** (member initializer lists), and **15.4** (destructors).
    - [ ] Read learncpp.com lesson **22.1** (introduction to smart pointers and move semantics) — the first half covers RAII with concrete examples. Read the [RAII page on cppreference](https://en.cppreference.com/w/cpp/language/raii.html) for the canonical definition.
    - [ ] Read learncpp.com lesson **27.2** (basic exception handling: `throw`, `try`, `catch`). You only need this one lesson — pybind11 auto-translates C++ exceptions to Python exceptions.
    - [ ] Skim learncpp.com lesson **11.6** (function templates). The goal is to recognize angle-bracket syntax like `py::array_t<uint8_t>` — you will not write templates in this project.
    - [ ] **Exercise 2**: Write a class `BufferView` that: takes `uint8_t* data`, `int width`, `int height` in its constructor (via member initializer list), validates them (throws `std::invalid_argument` if any dimension is <= 0 or data is `nullptr`), exposes `const` getters for all three, and has an explicitly defaulted destructor (`= default`).
        - Expected output: the program compiles, constructs a `BufferView` over a stack-allocated array, prints dimensions via getters, and catches the exception when constructed with `nullptr`.
        - Note: This class is a simplified version of the `ImageBuffer` you'll implement in Day 3. Same pattern: non-owning view, validating constructor, `const` getters, defaulted destructor.

- **Resources**:
    - [Learn C++](https://www.learncpp.com/) — Comprehensive, free C++ tutorial. Lessons referenced above are in chapters 0, 2, 11, 12, 14, 15, 17, 22, and 27. Read in the order listed in the tasks — the chapters are designed to build on each other.
    - [RAII — cppreference](https://en.cppreference.com/w/cpp/language/raii.html) — Canonical definition of RAII. Short page — read it after learncpp.com 22.1 for the formal framing.

- **If Stuck**:
    - Pointer arithmetic off-by-one or segfault: print the pointer address at each step (`std::cout << (void*)ptr`). Verify your loop bounds match the array size. Remember `ptr + i` advances by `i * sizeof(*ptr)` bytes.
    - Class won't compile — "no matching constructor": check that your member initializer list order matches the declaration order of members in the class.
    - Exception not caught: make sure you `catch` by `const` reference (`catch (const std::invalid_argument& e)`), not by value.

---

## Day 2: Pybind11 & NumPy Buffer Protocol (Foundation)

**Focus**: Python-C++ binding model, `py::module_`, `py::class_`, `py::array_t`, buffer protocol, `py::buffer_info`
**Load**: Level 3 — new API surface but exercises build directly on Day 0's noop binding
**Depends on**: Day 0, Day 1
**Prepares for**: Day 3 — RAII Buffer Wrapper and Zero-Copy Binding

- **Objectives**:
    1. You can explain how pybind11 exposes C++ classes and functions to Python — the role of `PYBIND11_MODULE`, `py::class_`, `py::init`, and `m.def()`.
    2. You can write a pybind11 binding that accepts a `py::array_t<uint8_t>` and extracts `py::buffer_info` from it, reading shape, strides, and the raw data pointer.
    3. You can explain the NumPy buffer protocol — how `py::array_t` wraps a contiguous memory buffer with shape and stride metadata, and why `c_style | forcecast` guarantees row-major layout.

- **Tasks**:

    **Pybind11 Binding Model**
    - [ ] Read the [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) page in full. Focus on how `PYBIND11_MODULE` defines the entry point, how `m.def()` binds free functions, and how return value policies work.
    - [ ] Read the [Object-oriented code — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) page. Focus on **Creating bindings for a custom type** and **Instance and static fields** — these cover `py::class_`, `py::init`, and property definitions.
    - [ ] **Exercise 1**: Write a small C++ file with a `struct Point { int x; int y; }` and a pybind11 module that binds it with a constructor and read-only `.x`, `.y` properties. Build with CMake and verify from Python: `from mymod import Point; p = Point(3, 4); print(p.x, p.y)`.
        - Expected output: `3 4`. This proves you can set up the full pybind11 binding pipeline — CMake build, module import, class instantiation from Python.

    **NumPy Buffer Protocol & `py::array_t`**
    - [ ] Read the [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) page. Focus on the **Buffer protocol** and **Arrays** sections for how `py::buffer_info` maps to NumPy internals (shape, strides, ptr, ndim). Skip **Vectorizing functions** — irrelevant here.
    - [ ] Read [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html). Read **Internal memory layout of an ndarray** for the stride formula and contiguity guarantees; skim **Array attributes** for `.ctypes`, `.strides`, `.flags`.
    - [ ] **Exercise 2**: Write a pybind11 function `describe_array(py::array_t<uint8_t> arr)` that extracts `py::buffer_info`, prints `ndim`, `shape`, `strides`, and the raw pointer address as `uintptr_t`. Call it from Python with `np.zeros((4, 4, 3), dtype=np.uint8)` and verify the output.
        - Expected output: `ndim=3 shape=[4, 4, 3] strides=[12, 3, 1] ptr=<some address>`. The strides `[12, 3, 1]` confirm row-major C-contiguous layout (4 cols x 3 channels = 12 bytes per row). This previews exactly how `ImageBuffer`'s constructor will extract buffer metadata in Day 3.

- **Resources**:
    - [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) — `PYBIND11_MODULE`, `m.def()`, basic type conversions. Read the full page.
    - [Object-oriented code — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) — `py::class_`, `py::init`, properties. Focus on **Creating bindings for a custom type** and **Instance and static fields**.
    - [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) — `py::array_t`, buffer protocol, stride access. Focus on **Buffer protocol** and **Arrays**; skip **Vectorizing functions**.
    - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy's memory model. Read **Internal memory layout of an ndarray** and **Array attributes** for `.ctypes`, `.strides`, `.flags`.

- **If Stuck**:
    - Exercise 1 won't import: make sure the module name in `PYBIND11_MODULE(mymod, m)` matches what you `import` in Python, and that `PYTHONPATH` points to the build directory.
    - `py::buffer_info` has unexpected values: print all fields. Check that your Python array uses `dtype=np.uint8` — the default `float64` will have different strides and item size.
    - Strides don't match expected values: NumPy may return a non-contiguous array. Use `np.ascontiguousarray()` or pass `py::array::c_style` in the type signature.

---

## Day 3: RAII Buffer Wrapper and Zero-Copy Binding

**Focus**: ImageBuffer RAII class, pybind11 `py::array_t`, `py::buffer_info`, zero-copy pointer extraction, Python integration test
**Load**: Level 4 — three components to integrate across an FFI boundary (C++ class, pybind11 binding, Python test)
**Depends on**: Day 0, Day 1, Day 2

- **Objectives**:
    1. `ImageBuffer` class accepts a `py::array_t<uint8_t>` in its constructor, extracts the raw pointer and dimensions from `py::buffer_info`, validates shape (3D, 3-4 channels, writable), and stores only C++ primitives.
    2. The pybind11 module binds `ImageBuffer` with its constructor and `data_ptr()` getter.
    3. A Python test proves zero-copy: `ImageBuffer.data_ptr() == ndarray.ctypes.data`.

- **Tasks**:

    **ImageBuffer Class**
    - [ ] Implement `ImageBuffer` in `include/image_buffer.hpp`. Constructor accepts `py::array_t<uint8_t, py::array::c_style | py::array::forcecast>`, calls `.request()` to get `py::buffer_info`, validates `ndim == 3`, `shape[2]` is 3 or 4, and buffer is writable. Store `uint8_t* data_`, `int height_`, `int width_`, `int channels_` — no `py::` types as members.
        - Note: The constructor is the only place `py::` types appear. After construction, `ImageBuffer` holds only raw C++ primitives. This is what makes it safe to read from `std::thread` — no Python API calls from threads. This is the RAII pattern from Day 1's Exercise 2, now with a real buffer source. The `py::buffer_info` extraction is exactly what Day 2's Exercise 2 practiced.
    - [ ] Add `const` getters: `data_ptr()`, `height()`, `width()`, `channels()`. Destructor is `= default` (non-owning view; NumPy owns the memory).
    - [ ] Throw `std::invalid_argument` for shape violations — pybind11 auto-translates this to Python `ValueError`.

    **Pybind11 Binding**
    - [ ] In `src/module.cpp`, bind `ImageBuffer` with `py::class_<ImageBuffer>`. Expose the constructor taking `py::array_t<uint8_t>`, and bind `data_ptr()` to return the pointer as `uintptr_t` (so Python can compare it with `ndarray.ctypes.data`).
        - Note: Returning `uintptr_t` instead of `uint8_t*` avoids pybind11 trying to interpret the pointer as a Python object. This is the standard pattern for exposing raw addresses to Python for verification.

    **Zero-Copy Verification**
    - [ ] Write `python/test_zerocopy.py` with pytest. Create a NumPy array (`np.zeros((100, 100, 3), dtype=np.uint8)`), pass it to `ImageBuffer`, and assert `buf.data_ptr() == arr.ctypes.data`.
    - [ ] Add a test for shape validation: assert that passing a 2D array raises `ValueError`.
    - [ ] Run `cmake -B build && cmake --build build && PYTHONPATH=build uv run pytest python/test_zerocopy.py -v` — all tests pass.

- **Acceptance Criteria**:
    - `ImageBuffer.data_ptr()` returns the exact same address as `ndarray.ctypes.data` — verified by pytest assertion.
    - Passing a 2D NumPy array to `ImageBuffer` raises `ValueError` in Python.
    - `ImageBuffer` has zero `py::` member variables — grep `image_buffer.hpp` for `py::` and confirm only the constructor parameter uses it.

- **Resources**:
    - [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) — Review the **Buffer protocol** and **Arrays** sections from Day 2. This time you're applying them to a real class, not an exercise.
    - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — Reference the **pybind11_add_module** section if you need to add new source files to the build target.

- **If Stuck**:
    - `py::buffer_info` has wrong `ndim` or `shape`: print `info.ndim`, `info.shape[0]`, etc. inside the constructor. NumPy's default dtype for `np.zeros` is `float64`, not `uint8` — make sure your Python test uses `dtype=np.uint8`.
    - Pointer mismatch (zero-copy fails): check that you're using `py::array::c_style | py::array::forcecast` in the constructor signature. Without `c_style`, pybind11 may accept non-contiguous arrays and silently copy.
    - `ImportError` when importing the module: verify `cmake --build build` succeeded and you're passing `PYTHONPATH=build`.
    - `data_ptr()` returns a Python `int` that looks different from `ctypes.data`: ensure `data_ptr()` returns `uintptr_t`, not a raw pointer. Compare as integers in Python.

---

## Day 4: C++ Foundations II — Threads and Concurrency (Foundation)

**Focus**: `std::thread`, `join`, workload partitioning, data races, non-overlapping regions
**Load**: Level 3 — fewer concepts than Day 1 but concurrency requires careful reasoning
**Depends on**: Day 1
**Prepares for**: Day 6 — Threaded Dispatch and GIL Release
**Note**: Days 4 and 5 have no dependency between them and can be done in parallel.

- **Objectives**:
    1. You can spawn a `std::thread`, pass arguments to it, and join it before the thread object is destroyed.
    2. You can explain what a data race is and why non-overlapping memory regions eliminate the need for synchronization.
    3. You can partition an integer range `[0, N)` into K non-overlapping sub-ranges that cover the full range with no gaps.

- **Tasks**:

    **std::thread Basics**
    - [ ] Read [std::thread — cppreference](https://en.cppreference.com/w/cpp/thread/thread.html) — focus on the constructor (passing a callable + arguments), `join()`, and the note that a `std::thread` must be joined or detached before destruction.
    - [ ] Read learncpp.com lesson **28.1** (introduction to concurrency) if it exists, or the [C++ Concurrency in Action free sample chapter](https://livebook.manning.com/book/c-plus-plus-concurrency-in-action/chapter-2/) for a gentler intro to `std::thread`.
    - [ ] **Exercise 1**: Write a program that spawns 4 `std::thread` instances, each printing its thread index and a message, then joins all 4. Observe that output may interleave.
        - Expected output: 4 lines like `Thread 0: hello`, `Thread 1: hello`, etc. — order may vary due to scheduling.

    **Data Races and Partitioning**
    - [ ] Read [Data races — cppreference](https://en.cppreference.com/w/cpp/language/memory_model.html#Threads_and_data_races) — understand that two threads writing to the same memory location without synchronization is undefined behavior.
    - [ ] **Exercise 2**: Write a program that allocates a `uint8_t` array of 1000 elements, partitions it into 4 non-overlapping row-ranges (compute `start_row` and `end_row` for each thread using integer division with remainder distribution), spawns 4 threads where each thread sets its range to its thread index, then joins all and prints the full array to verify no gaps or overlaps.
        - Expected output: `0 0 0 ... 0 1 1 1 ... 1 2 2 2 ... 2 3 3 3 ... 3` with exactly 1000 values, 250 each (or 250/250/250/250 for even division).
        - Note: This is exactly the partitioning logic `process_quadrants` will use in Day 6 — dividing image rows among 4 threads with non-overlapping ranges so no synchronization is needed.

- **Resources**:
    - [std::thread — cppreference](https://en.cppreference.com/w/cpp/thread/thread.html) — Canonical reference. Focus on the constructor overloads and `join()`. Skim `detach()` and `joinable()` — you'll use `join()` exclusively in this project.
    - [Threads and data races — cppreference](https://en.cppreference.com/w/cpp/language/memory_model.html#Threads_and_data_races) — Formal definition of data races. Dense but short — read it once to understand why non-overlapping regions are the key invariant.
        - *If the above is unclear:* [Data Races — LearnCpp (if available)](https://www.learncpp.com/) or search "C++ data race simple explanation" — the concept is simpler than the formal spec.
    - [Concurrency support library — cppreference](https://en.cppreference.com/w/cpp/thread.html) — Overview page listing all threading primitives. You only need `std::thread` for this project, but good to see the landscape.

- **If Stuck**:
    - Program crashes on thread destruction: you forgot to `join()`. Every `std::thread` must be joined before it goes out of scope, or the program calls `std::terminate`.
    - Threads print garbled output: `std::cout` is not thread-safe by default. For Exercise 1 this is expected (demonstrates interleaving). For Exercise 2, don't print from threads — print after all joins.
    - Partitioning has gaps or overlaps: use `start = i * total / n_threads` and `end = (i+1) * total / n_threads` — integer arithmetic handles uneven division correctly.

---

## Day 5: Transform Kernels and C++ Unit Tests

**Focus**: Pixel transformation functions, raw pointer arithmetic on image buffers, GoogleTest
**Load**: Level 3 — domain logic is straightforward but testing across the FFI boundary adds complexity
**Depends on**: Day 3
**Note**: Days 4 and 5 have no dependency between them and can be done in parallel.

- **Objectives**:
    1. `transform_quadrant()` takes a raw `uint8_t*`, row range, width, and channels, and applies an in-place pixel transformation using pointer arithmetic.
    2. `process_quadrants()` (single-threaded for now) partitions an `ImageBuffer` into 4 row-ranges and calls `transform_quadrant()` on each.
    3. C++ unit tests verify transform correctness and partitioning logic.

- **Tasks**:

    **Transform Kernel**
    - [ ] Implement `transform_quadrant()` in `src/transforms.cpp` — takes `uint8_t* data`, `int start_row`, `int end_row`, `int width`, `int channels`, and an `int quadrant_index`. Apply a distinct per-quadrant transformation (e.g., invert, threshold, channel swap, brightness shift) by walking the row range with raw pointer arithmetic.
        - Note: `data + row * width * channels` gives you the start of any row. This is the pointer arithmetic from Day 1's Exercise 1, now applied to a real image buffer.
    - [ ] Declare `transform_quadrant()` and `process_quadrants()` in a header (`include/transforms.hpp` or directly in `image_buffer.hpp`).

    **Single-Threaded process_quadrants**
    - [ ] Implement `process_quadrants()` in `src/transforms.cpp` — takes a `const ImageBuffer&`, partitions `[0, height)` into 4 row-ranges (same partitioning logic as Day 4's Exercise 2), and calls `transform_quadrant()` on each range sequentially.
        - Note: Threading comes in Day 6. Building the single-threaded version first lets you verify transform correctness without concurrency complications.

    **C++ Unit Tests**
    - [ ] In `tests/test_transform_quadrant.cpp`: allocate a test buffer, run `transform_quadrant()` on it, assert pixel values match expected output.
        - Note: This is a **unit test** — pure C++ function, no Python, no FFI. GoogleTest's `EXPECT_EQ` and `ASSERT_EQ` are all you need.
    - [ ] In `tests/test_quadrant_partitioning.cpp`: test that 4 row-ranges from the partitioning logic cover `[0, H)` exactly with no gaps or overlaps for various image heights (1, 3, 4, 100, 1080).
    - [ ] Run `cmake --build build && ctest --test-dir build --output-on-failure` — all tests pass.

    **Python Integration**
    - [ ] Bind `process_quadrants()` in `src/module.cpp` — for now, without GIL release (that's Day 6).
    - [ ] Add a Python test in `python/test_transforms.py`: create a NumPy array with known values, call `process_quadrants()`, assert the array was modified in-place with expected values.
    - [ ] Run `PYTHONPATH=build uv run pytest python/ -v` — all Python tests pass (including Day 3's zero-copy tests).

- **Acceptance Criteria**:
    - `ctest --test-dir build --output-on-failure` passes all C++ tests.
    - `PYTHONPATH=build uv run pytest python/ -v` passes all Python tests.
    - `process_quadrants()` modifies the NumPy array in-place — no new array is returned, the original array's values have changed.

- **Resources**:
    - [GoogleTest Primer](https://google.github.io/googletest/primer.html) — Core testing concepts: `TEST()`, `EXPECT_EQ`, `ASSERT_EQ`, test fixtures. Read the full page — it's the minimum needed to write the C++ tests.
    - [GoogleTest Quickstart: Building with CMake](https://google.github.io/googletest/quickstart-cmake.html) — Reference for the CMake integration you set up in Day 0. Revisit if you need to add new test executables.

- **If Stuck**:
    - Transform produces wrong pixel values: print the raw buffer before and after. Double-check your pointer arithmetic — `data[row * width * channels + col * channels + c]` is the standard layout for a row-major HWC image.
    - ctest finds no tests: make sure `gtest_discover_tests(<target>)` is in your `CMakeLists.txt` for each test executable.
    - Python test fails but C++ tests pass: the binding may be passing the array by copy. Verify `process_quadrants` takes `ImageBuffer&` (not by value) and the Python side passes the array directly.

---

## Day 6: Threaded Dispatch and GIL Release

**Focus**: `std::thread` dispatch, `py::call_guard<py::gil_scoped_release>`, GIL-free execution, ThreadSanitizer
**Load**: Level 4 — threading + GIL release is the hardest integration in the project
**Depends on**: Day 4, Day 5

- **Objectives**:
    1. `process_quadrants()` spawns 4 `std::thread` instances, each running `transform_quadrant()` on a non-overlapping row-range, and joins all before returning.
    2. The pybind11 binding for `process_quadrants()` uses `py::call_guard<py::gil_scoped_release>()` to release the GIL for the entire C++ call.
    3. ThreadSanitizer reports no data races.

- **Tasks**:

    **Threaded Dispatch**
    - [ ] Modify `process_quadrants()` in `src/transforms.cpp` to spawn 4 `std::thread` instances instead of calling `transform_quadrant()` sequentially. Each thread gets its own non-overlapping row-range. Join all 4 threads before the function returns.
        - Note: This is Day 4's Exercise 2 applied to real image data. The `ImageBuffer` members (`data_ptr()`, `height()`, etc.) are effectively immutable after construction — safe to read from all threads without synchronization. The row-ranges are non-overlapping — no thread writes to another's memory. These two invariants together mean zero synchronization needed.
    - [ ] Ensure all threads are joined in a scope-exit loop — if any thread throws, the others must still be joined. Use a simple `for` loop over a `std::vector<std::thread>` or `std::array<std::thread, 4>`.

    **GIL Release**
    - [ ] In `src/module.cpp`, add `py::call_guard<py::gil_scoped_release>()` to the `process_quadrants` binding.
        - Note: This releases the GIL before entering C++ and reacquires it when the function returns. No C++ code inside `process_quadrants` or `transform_quadrant` touches any `py::` types — they only use raw `uint8_t*` and `int`. This is the GIL discipline: all thread-side code is pure C++.
    - [ ] Verify that `ImageBuffer`'s constructor (which DOES use `py::` types) runs BEFORE `process_quadrants` is called — the GIL is held during construction, released only for the transform.

    **ThreadSanitizer Validation**
    - [ ] Build with ThreadSanitizer: `cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=thread" && cmake --build build`.
    - [ ] Run the C++ tests under TSan: `ctest --test-dir build --output-on-failure`. Fix any reported races.
    - [ ] Run the Python tests under TSan: `PYTHONPATH=build uv run pytest python/ -v`. TSan should report zero warnings.
        - Note: ThreadSanitizer is a compiler flag + runtime. It instruments memory accesses at compile time and detects data races at runtime. If it reports nothing, your threading is correct. If it reports a race, it will show the exact two accesses that conflict — read the output carefully.

    **Thread Safety Test**
    - [ ] In `tests/test_thread_safety.cpp`: create a large buffer, run `process_quadrants()`, verify all quadrants were transformed correctly (same assertions as Day 5, but now running threaded).

- **Acceptance Criteria**:
    - `process_quadrants()` uses 4 threads — verify by temporarily adding a print inside `transform_quadrant` showing `std::this_thread::get_id()` and confirming 4 distinct IDs.
    - ThreadSanitizer reports zero data races in both C++ and Python test suites.
    - All Day 5 Python tests still pass — threading must not change observable behavior.
    - `grep -n "py::" src/transforms.cpp` returns zero matches — no Python API in the threaded code path.

- **Resources**:
    - [Miscellaneous — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/misc.html) — Read the **Global Interpreter Lock (GIL)** section for how `gil_scoped_release` and `gil_scoped_acquire` work, and the **Concurrency and Parallelism in Python with pybind11** section for the concurrency patterns. These two sections are the authoritative reference for GIL management in pybind11.
    - [Functions — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/functions.html) — Read the **Call guard** section for the `py::call_guard<>` syntax used to release the GIL declaratively in the binding.
    - [ThreadSanitizer — Clang documentation](https://clang.llvm.org/docs/ThreadSanitizer.html) — How to build with `-fsanitize=thread` and interpret the output. Short page — read it fully.

- **If Stuck**:
    - ThreadSanitizer reports a race on `ImageBuffer` members: the threads are reading `data_ptr()` etc. while something else writes them. Check that `ImageBuffer` is fully constructed BEFORE threads are spawned — the constructor must finish before any thread reads the members.
    - Program crashes with "terminate called without an active exception": a `std::thread` went out of scope without being joined. Make sure your join loop runs even if an earlier thread threw.
    - Python segfault after adding `gil_scoped_release`: something in the C++ call path is touching a `py::` type without the GIL. `grep -rn "py::" src/transforms.cpp` — if any match, that's the bug.
    - TSan false positives from Python internals: TSan may flag Python's own allocator. These are known false positives — look for your code paths in the report, ignore frames in `libpython`.

---

## Day 7: Benchmarks, Validation, and Final Proof

**Focus**: pytest-benchmark, zero-copy empirical proof, allocation tracking, acceptance validation
**Load**: Level 3 — instrumentation and measurement, no new architecture
**Depends on**: Day 6

- **Objectives**:
    1. Benchmarks prove that `process_quadrants()` on a large image (e.g., 4000x4000x3) completes with no measurable allocation overhead.
    2. A validation script runs the full pipeline end-to-end and asserts all invariants hold.
    3. All C++ tests, Python tests, and benchmarks pass.

- **Tasks**:

    **Benchmarks**
    - [ ] Write `python/test_bench.py` using pytest-benchmark. Benchmark `process_quadrants()` on a `4000x4000x3 uint8` array. Use `benchmark.pedantic(rounds=10, warmup_rounds=2)` for stable results.
        - Note: pytest-benchmark provides the `benchmark` fixture automatically. The key metric is wall-clock time — compare against a pure NumPy equivalent to see the speedup from C++ threading.
    - [ ] Add a benchmark that allocates the NumPy array inside the timed region vs. outside, to prove that `process_quadrants` itself does not allocate.

    **Manual Benchmark Script**
    - [ ] Write `python/bench.py` — a standalone script that creates a large array, calls `process_quadrants()`, and prints timing results using `time.perf_counter_ns()`. Include before/after memory tracking with `tracemalloc` to empirically prove zero allocation during the C++ call.

    **Acceptance Validation**
    - [ ] Write `python/run_validation.py` — runs the full acceptance check:
        1. Create a large NumPy array.
        2. Record `ndarray.ctypes.data`.
        3. Call `process_quadrants()`.
        4. Assert `ndarray.ctypes.data` hasn't changed (same address — no reallocation).
        5. Assert pixel values were transformed (not all zeros).
        6. Print "PASS" with timing.

    **Final Test Suite**
    - [ ] Run the full test suite: `ctest --test-dir build --output-on-failure && PYTHONPATH=build uv run pytest python/ -v`.
    - [ ] Run benchmarks: `PYTHONPATH=build uv run pytest python/test_bench.py -v --benchmark-enable`.
    - [ ] Run validation: `PYTHONPATH=build uv run python python/run_validation.py`.

- **Acceptance Criteria**:
    - `ctest --test-dir build --output-on-failure` — all C++ tests pass.
    - `uv run pytest python/ -v` — all Python tests pass (zero-copy, transforms, threading).
    - `uv run pytest python/test_bench.py -v --benchmark-enable` — benchmarks complete without errors.
    - `uv run python python/run_validation.py` — prints PASS, confirms pointer equality (zero-copy) and transformation correctness.
    - `tracemalloc` in `bench.py` shows zero Python-side allocations during the `process_quadrants()` call.

- **Resources**:
    - [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/en/latest/) — Read the quickstart for the `benchmark` fixture and `pedantic()` mode. The `--benchmark-enable` flag is needed because pytest-benchmark disables benchmarks by default in test runs.
    - [tracemalloc — Python documentation](https://docs.python.org/3/library/tracemalloc.html) — Built-in memory allocation tracker. Use `tracemalloc.start()` before and `tracemalloc.get_traced_memory()` after to measure Python-side allocations.

- **If Stuck**:
    - pytest-benchmark results show high variance: increase `rounds` and `warmup_rounds`. Ensure no other heavy processes are running. Use `--benchmark-disable-gc` to prevent garbage collection during timing.
    - `tracemalloc` shows allocations during `process_quadrants()`: these may be from Python-side argument marshalling (creating the `py::array_t` wrapper). The key assertion is that `ndarray.ctypes.data` doesn't change — the underlying buffer was not reallocated.
    - Validation script fails pointer equality: the array was copied somewhere. Check that you're not accidentally slicing or reshaping the array before passing it to C++.

---

## Day 8: DevOps — Makefile, CI Pipeline, and Containerization

**Focus**: Makefile, GitHub Actions, Docker, multi-stage builds, CI with lint + build + test, reproducible builds
**Load**: Level 3 — three infrastructure components but each is straightforward config
**Depends on**: Day 7

- **Objectives**:
    1. A Makefile wraps every project command — build, test, lint, format, bench, validate, docker, clean.
    2. A GitHub Actions workflow runs lint, build, and test on every push and PR.
    3. A multi-stage Dockerfile builds and tests the project from scratch on a clean machine.

- **Tasks**:

    **Makefile**
    - [ ] Author a root `Makefile` with targets: `build` (`cmake -B build && cmake --build build`), `test` (ctest + pytest), `bench` (pytest-benchmark), `validate` (`run_validation.py`), `lint` (ruff + mypy + clang-tidy), `format` (ruff format + clang-format), `docker` (build and run the Docker image), `clean` (removes `build/`, `.venv/`, `__pycache__/`, `*.so`, `uv` cache, pre-commit environments, Docker image — everything non-source so the project rebuilds from scratch).
        - Note: The Makefile is a convenience wrapper — the real build system is CMake. All Python commands use `PYTHONPATH=build uv run`. `make clean` is the nuclear option: after running it, `uv sync && make build` should be all that's needed to get back to a working state.

    **GitHub Actions Workflow**
    - [ ] Create `.github/workflows/ci.yml`. Trigger on `push` and `pull_request` to `main`.
    - [ ] Set up the job: Ubuntu latest, install CMake, install Python 3.14 via `actions/setup-python`, install `uv` via `astral-sh/setup-uv`.
    - [ ] Steps: `uv sync`, `make build`, `make lint`, `make test`, `make validate`.
        - Note: Don't run benchmarks in CI — wall-clock timing is unreliable on shared runners. `make validate` tests correctness, not performance.
    - [ ] Add a matrix strategy for GCC and Clang if you want to test both compilers.

    **Dockerfile**
    - [ ] Write a multi-stage `Dockerfile`:
        - **Build stage**: `python:3.14-slim` base, install system deps (`cmake`, `g++`, `curl`), install `uv`, copy the project, run `uv sync && make build && make test`.
        - **Runtime stage**: copy only the built `.so` and Python files into a slim image. This is the artifact you'd ship.
        - Note: The build stage proves reproducibility — a clean machine that has never seen your code. The multi-stage split keeps the final image small by excluding build tools. Copy `pyproject.toml` and `uv.lock` first (before source code) for better layer caching.
    - [ ] Add `make docker` target to the Makefile that builds and runs the Docker image.
    - [ ] Add the Docker image to `make clean`'s removal list.

- **Acceptance Criteria**:
    - `make test` runs the full C++ + Python test suite and exits 0.
    - `make lint` runs all linters and exits 0.
    - Push to a branch triggers the GitHub Actions workflow; the workflow passes with a green check.
    - `make docker` builds the multi-stage image and all tests pass inside the container.
    - `docker images` shows the runtime image is significantly smaller than the build stage.

- **Resources**:
    - [GNU Make Manual — Introduction](https://www.gnu.org/software/make/manual/html_node/Introduction.html) — Read **An Introduction to Makefiles** for targets, prerequisites, and recipes. Skip implicit rules and pattern matching — you won't need them for a wrapper Makefile.
    - [GitHub Actions Documentation — Quickstart](https://docs.github.com/en/actions/quickstart) — Read the quickstart for workflow YAML syntax, triggers, and job steps.
    - [astral-sh/setup-uv — GitHub Action](https://github.com/astral-sh/setup-uv) — Official action for installing `uv` in CI. Read the README for usage.
    - [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Reference for `FROM`, `RUN`, `COPY`, `WORKDIR`. Focus on the **Multi-stage builds** section — it's the key technique for this day.
    - [Multi-stage builds — Docker documentation](https://docs.docker.com/build/building/multi-stage/) — Dedicated guide to multi-stage builds. Shows the `FROM ... AS build` / `COPY --from=build` pattern.

- **If Stuck**:
    - CI can't find `cmake`: add `sudo apt-get install -y cmake` to the workflow, or use a `cmake` action.
    - CI can't import the `.so` module: make sure `PYTHONPATH=build` is set in the test step's environment, or use `env:` in the workflow YAML.
    - CI fails on clang-tidy but passes locally: clang-tidy versions differ across systems. Pin the version in CI or use `clang-tidy-18` explicitly.
    - Docker build fails on `uv sync`: make sure `pyproject.toml` and `uv.lock` are copied before the source code for layer caching.
    - Runtime stage can't find the `.so`: use `COPY --from=build /app/build/*.so /app/` (adjust paths to match your build stage layout).

---

## Day 9: Documentation and Polish (Optional)

**Focus**: README, project polish
**Load**: Level 1 — writing, no new code
**Depends on**: Day 8

- **Objectives**:
    1. A README provides everything needed to clone, build, test, and use the project.

- **Tasks**:

    **README**
    - [ ] Write `README.md` with:
        - Project description (one paragraph).
        - Prerequisites: Python 3.14, CMake >= 3.16, C++17 compiler, uv, Docker.
        - Quick start: `uv sync && make build && make test`.
        - Available `make` targets (build, test, bench, validate, lint, format, docker, clean).
        - Architecture overview (reference or summarize CLAUDE.md's Architecture section).
        - CI badge from GitHub Actions.

- **Acceptance Criteria**:
    - A new developer can clone the repo, read the README, and run `uv sync && make build && make test` successfully without any other documentation.

- **Resources**:
    - [Make a README](https://www.makeareadme.com/) — Quick guide to writing a good README. Short page — read it once.

- **If Stuck**:
    - Not sure what to include: look at the README of any well-maintained open-source project in your stack (e.g., pybind11, GoogleTest) for structure inspiration.
