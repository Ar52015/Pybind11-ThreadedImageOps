# Phase 2: GIL-Free Threaded Image Operations

**The Validation Task**:

> To validate Phase 2, the developer must construct a hybrid Python-C++ module. Using Modern CMake, they will compile a Pybind11 shared object library (.so). The C++ library must implement a memory-safe RAII class that accepts a multi-dimensional NumPy tensor from Python. The C++ code must leverage multi-threading (via std::thread) to concurrently manipulate different quadrants of the image array in-place, bypassing the Python Global Interpreter Lock (GIL) entirely. The Python script will import this compiled library, pass a massive image payload, and verify that the transformation occurred instantaneously without any memory reallocation or copying overhead.

This phase proves that the developer can bridge the Python-C++ boundary at zero cost — no copies, no GIL contention, no safety violations — and exploit hardware parallelism directly from a Python caller.
**Goal**: Demonstrate mastery of native extension architecture: CMake-driven builds, RAII ownership semantics over foreign memory, GIL-free concurrency on shared mutable buffers, and empirical proof of zero-copy data flow.

---

## Day 0: Project Skeleton and Build Toolchain

**Focus**: Modern CMake, FetchContent, Pybind11 discovery, uv project init, linting, pre-commit hooks
**Load**: Level 2

- **Tasks**:
  - [x] Create the top-level directory layout: `src/` for C++ sources, `include/` for headers, `python/` for the driver script and tests, and `build/` (gitignored) for CMake artifacts.
  - [x] Initialize the Python project with `uv init` (generates `pyproject.toml`, `.python-version`, `uv.lock`). Add dependencies via `uv add numpy` and `uv add --dev pytest pytest-benchmark ruff mypy pre-commit`. Run `uv sync` to create the `.venv`.
    - Note: Pin the NumPy version explicitly in `pyproject.toml` (e.g. `numpy>=2.4,<3`) so the ABI stays stable across rebuilds. Commit `uv.lock` for reproducible installs.
  - [x] Configure pre-commit hooks (`.pre-commit-config.yaml`):
    - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`
    - Local hooks: `ruff check --fix`, `ruff format`, `mypy .`, `pytest --benchmark-disable` (all via `uv run`)
    - C++ hooks: `clang-format -i`, `clang-tidy --fix`, `ctest --test-dir build --output-on-failure`
    - Note: Run `uv run pre-commit install` to activate. Hooks must pass before any commit lands.
  - [x] Configure ruff in `pyproject.toml`: enable rule sets `E`, `F`, `I` (isort), `UP` (pyupgrade), `NPY` (NumPy-specific).
  - [x] Configure strict mypy in `pyproject.toml` (`[tool.mypy]` with `strict = true`).
  - [x] Add a `.clang-format` file for C++ source formatting (`BasedOnStyle: Google`).
    - Note: clang-format enforces consistent style across all `.cpp` and `.hpp` files. The Google style is widely used and well-documented.
  - [x] Add a `.clang-tidy` config with `concurrency-*`, `google-*`, `performance-*`, `readability-*`, `bugprone-*`, `modernize-*` checks.
    - Note: clang-tidy is a static analyzer, not just a formatter. It catches real bugs (use-after-move, thread-safety violations) that the compiler misses.
  - [x] Author the root `CMakeLists.txt` — `cmake_minimum_required(VERSION 3.16)`, `CMAKE_CXX_STANDARD 17`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`, `FetchContent` for pybind11 (`v3.0.2`) and GoogleTest (`v1.17.0`).
    - Note: pybind11 v3.0.3 exists as of 2026-03-31, but v3.0.2 is stable and already pinned in the project. GoogleTest v1.17.0 requires C++17 which matches our standard.
  - [x] Add a minimal `src/module.cpp` with `PYBIND11_MODULE` entry point exposing a `noop()` function returning a string literal.
  - [x] Add stub `src/transforms.cpp` and test files in `tests/` (`test_transform_quadrant.cpp`, `test_quadrant_partitioning.cpp`, `test_thread_safety.cpp`).
  - [x] Verify the full build-import cycle: `cmake -B build && cmake --build build`, then `PYTHONPATH=build uv run python -c "import threaded_image_ops; print(threaded_image_ops.noop())"` prints `hello from C++`.
  - [x] Symlink `compile_commands.json` to project root (`ln -s build/compile_commands.json .`) for clang-tidy LSP support.
  - [x] Add `build/`, `*.so`, `__pycache__/`, `.venv/`, and `compile_commands.json` to `.gitignore`.

- **Resources**:
  - [CMake Tutorial — Official Docs](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) — Step-by-step intro to CMake. Read **Step 1: Getting Started** for the four core commands (`cmake_minimum_required`, `project`, `add_executable`, `target_sources`); read **Step 2: Adding a Library** for `add_library` and `target_link_libraries`. This is a Tier 5 technology — read the full first two steps, don't skim.
  - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — CMake integration for pybind11. Focus on the **FetchContent** and **pybind11_add_module** sections for how to pull pybind11 and build a `.so` from CMake.
  - [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) — Covers `PYBIND11_MODULE`, `m.def()`, and basic type conversions. Read the full page — it's the minimum needed to write `module.cpp`.
  - [pre-commit documentation](https://pre-commit.com/) — Official docs for the pre-commit hook framework. Read **Adding pre-commit plugins** and **Creating new hooks** for `.pre-commit-config.yaml` syntax and local hook definitions.
  - [Configuring Ruff](https://docs.astral.sh/ruff/configuration/) — Ruff linter/formatter configuration in `pyproject.toml`. Read **Rule selection** for how `select` and `ignore` work with rule code prefixes.
  - [mypy configuration file](https://mypy.readthedocs.io/en/stable/config_file.html) — mypy config reference. Read **The mypy configuration file** for `[tool.mypy]` in `pyproject.toml`; skim **Strict mode** for which flags `strict = true` enables.
  - [Clang-Format Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html) — Full reference for `.clang-format`. Read **Configuring Style with clang-format** intro and **BasedOnStyle** for how predefined styles (Google, LLVM) work.
  - [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/) — clang-tidy overview and check categories. Read **Configuring Checks** for `.clang-tidy` YAML syntax; browse the [checks list](https://clang.llvm.org/extra/clang-tidy/checks/list.html) for what `bugprone-*`, `modernize-*`, etc. cover.

---

## Day 1: C++ Foundations I — Pointers, Memory, and Classes (Foundation)

**Focus**: Header files, raw pointers, pointer arithmetic, classes, constructors, destructors, RAII, const correctness, exceptions
**Load**: Level 3
**Prepares for**: Day 3 — RAII Wrapper and Zero-Copy NumPy Buffer Binding

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
  - [ ] **Exercise 2**: Write a class `BufferView` that: takes `uint8_t* data`, `int width`, `int height` in its constructor (via member initializer list), validates them (throws `std::invalid_argument` if any dimension is ≤ 0 or data is `nullptr`), exposes `const` getters for all three, and has an explicitly defaulted destructor (`= default`).
    - Expected output: construct a `BufferView` with valid args and print `width=640 height=480`. Then construct one with `height=0` and catch the exception, printing `caught: invalid dimension`. This class is a simplified version of the `ImageBuffer` you'll implement in Day 3.

- **Resources**:
  - [Introduction to the compiler, linker, and libraries (0.5)](https://www.learncpp.com/cpp-tutorial/introduction-to-the-compiler-linker-and-libraries/) — How C++ goes from source files to an executable. Continue to [2.11 — Header files](https://www.learncpp.com/cpp-tutorial/header-files/) and [2.12 — Header guards](https://www.learncpp.com/cpp-tutorial/header-guards/) for `#include` mechanics and `#pragma once`. Revisit [15.2 — Classes and header files](https://www.learncpp.com/cpp-tutorial/classes-and-header-files/) after reading the classes lessons.
  - [Introduction to pointers (12.7)](https://www.learncpp.com/cpp-tutorial/introduction-to-pointers/) — Pointer declaration, dereferencing, and nullptr. Continue to [12.9 — Pointers and const](https://www.learncpp.com/cpp-tutorial/pointers-and-const/) for const pointer rules, then [17.9 — Pointer arithmetic and subscripting](https://www.learncpp.com/cpp-tutorial/pointer-arithmetic-and-subscripting/) for the offset math behind buffer traversal.
  - [Member functions (14.3)](https://www.learncpp.com/cpp-tutorial/member-functions/) — Start of the classes arc. Read sequentially through [14.4](https://www.learncpp.com/cpp-tutorial/const-class-objects-and-const-member-functions/) (const members), [14.5](https://www.learncpp.com/cpp-tutorial/public-and-private-members-and-access-specifiers/) (access specifiers), skip to [14.9](https://www.learncpp.com/cpp-tutorial/introduction-to-constructors/) (constructors), [14.10](https://www.learncpp.com/cpp-tutorial/constructor-member-initializer-lists/) (member initializer lists). Finish with [15.4 — Introduction to destructors](https://www.learncpp.com/cpp-tutorial/introduction-to-destructors/).
  - [Introduction to smart pointers and move semantics (22.1)](https://www.learncpp.com/cpp-tutorial/introduction-to-smart-pointers-move-semantics/) — The first half explains RAII: acquiring resources in constructors, releasing in destructors. Also read the [RAII page on cppreference](https://en.cppreference.com/w/cpp/language/raii.html) — it's short and canonical.
  - [Basic exception handling (27.2)](https://www.learncpp.com/cpp-tutorial/basic-exception-handling/) — `throw`, `try`, `catch` mechanics. One lesson is enough — skip the rest of chapter 27 for now.
  - [Function templates (11.6)](https://www.learncpp.com/cpp-tutorial/function-templates/) — Skim to recognize `template<typename T>` and angle-bracket type parameters. You don't need to write templates — just read them.

---

## Day 2: Pybind11 & NumPy Buffer Protocol (Foundation)

**Focus**: Python-C++ binding model, `py::module_`, `py::class_`, `py::array_t`, buffer protocol, `py::buffer_info`
**Load**: Level 3
**Prepares for**: Day 3 — RAII Wrapper and Zero-Copy NumPy Buffer Binding

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
    - Expected output: `ndim=3 shape=[4, 4, 3] strides=[12, 3, 1] ptr=<some address>`. The strides `[12, 3, 1]` confirm row-major C-contiguous layout (4 cols × 3 channels = 12 bytes per row). This previews exactly how `ImageBuffer`'s constructor will extract buffer metadata in Day 3.

- **Resources**:
  - [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) — `PYBIND11_MODULE`, `m.def()`, basic type conversions. Read the full page.
  - [Object-oriented code — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) — `py::class_`, `py::init`, properties. Focus on **Creating bindings for a custom type** and **Instance and static fields**.
  - [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) — `py::array_t`, buffer protocol, stride access. Focus on **Buffer protocol** and **Arrays**; skip **Vectorizing functions**.
  - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy's memory model. Read **Internal memory layout of an ndarray** and **Array attributes** for `.ctypes`, `.strides`, `.flags`.

---

## Day 3: RAII Wrapper and Zero-Copy NumPy Buffer Binding

**Focus**: `py::array_t` buffer protocol, RAII resource semantics, NumPy stride arithmetic, `py::buffer_info`
**Load**: Level 3

- **Objectives**:
  1. A C++ RAII class owns a non-copying, mutable view into a caller-supplied NumPy array and validates its shape on construction.
  2. The class exposes raw pointer access and dimension metadata through a clean public interface — no Python types leak past the constructor.
  3. Python can instantiate the class, pass an `(H, W, C)` uint8 array, and read back the dimensions without any data duplication.

- **Tasks**:

  **RAII Class (C++)**
  - [ ] Define `ImageBuffer` in `include/image_buffer.hpp` — constructor accepts `py::array_t<uint8_t, py::array::c_style | py::array::forcecast>` and extracts `py::buffer_info` from it.
    - Note: The `py::array::c_style` flag guarantees contiguous row-major layout. Without it, a Fortran-order array silently produces garbage strides.
  - [ ] Store the raw `uint8_t*` data pointer plus `height`, `width`, and `channels` as private members. Compute and store `row_stride` (bytes per row) from `buffer_info.strides[0]`.
    - Note: Do **not** store `py::array_t` or `py::buffer_info` as members — the GIL must not be required to read these fields later during threaded work.
  - [ ] Assert in the constructor that `buffer_info.ndim == 3`, that `buffer_info.shape[2]` is 3 or 4 (RGB/RGBA), and that the buffer is writable (`buffer_info.readonly == false`). Throw `std::invalid_argument` on violation.
  - [ ] Implement a destructor that is explicitly defaulted (`= default`) — the class does **not** own the buffer's memory (NumPy does), so no deallocation occurs.
    - Note: This is the RAII discipline under test: the class manages _access lifetime_, not _allocation lifetime_. The destructor's job is to guarantee that no dangling work (threads) outlives the buffer view.

  **Pybind11 Binding**
  - [ ] In `src/module.cpp`, bind `ImageBuffer` via `py::class_<ImageBuffer>` with an `__init__` that accepts `py::array_t<uint8_t>`.
  - [ ] Expose read-only properties: `.height`, `.width`, `.channels`.
  - [ ] Bind a `data_ptr()` method that returns the raw pointer as `uintptr_t` — this is the diagnostic hook Python will use to prove zero-copy.

  **Verification Script (Python)**
  - [ ] Write `python/test_zerocopy.py`: allocate a NumPy array with `np.zeros((4096, 4096, 3), dtype=np.uint8)`, pass it to `ImageBuffer`, and assert that `buf.data_ptr() == arr.ctypes.data` — same integer address means zero copies.
    - Note: This is an integration test — it crosses the FFI boundary (Python → C++ → Python). It verifies the zero-copy invariant that is the project's core constraint.
  - [ ] Assert `buf.height == 4096`, `buf.width == 4096`, `buf.channels == 3`.

- **Acceptance Criteria**:
  - `PYTHONPATH=build uv run pytest python/test_zerocopy.py -v` exits 0 and prints no assertion errors.
  - The `data_ptr()` returned by the C++ side is byte-identical to `ndarray.ctypes.data` on the Python side — proving no intermediate copy was allocated.
  - Passing a 2D array (`np.zeros((100, 100))`) raises a Python `ValueError` originating from the C++ `std::invalid_argument`.

- **Resources**:
  - [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) — Re-read **Buffer protocol** and **Arrays** from Day 2. This time focus on `py::buffer_info` field access patterns (`.ptr`, `.ndim`, `.shape`, `.strides`, `.readonly`).
  - [Object-oriented code — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) — Re-read **Creating bindings for a custom type**. Focus on `py::init` constructor binding and `def_property_readonly` for exposing getters.
  - [RAII — cppreference.com](https://en.cppreference.com/w/cpp/language/raii.html) — Canonical definition of RAII. Read the full page — it's short. Pay attention to the bad-vs-good mutex example.
  - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — Reference for `.ctypes.data`, `__array_interface__`, and `flags` attributes used in the zero-copy assertion.

---

## Day 4: C++ Foundations II — Threads, Lambdas, Concurrency, and GoogleTest (Foundation)

**Focus**: Lambda expressions, lambda captures, `std::thread`, move semantics, object lifetime, scope-exit patterns, GoogleTest basics
**Load**: Level 3
**Prepares for**: Day 5 — Multi-Threaded Quadrant Transforms with GIL Release

- **Objectives**:
  1. You can write lambda expressions with explicit captures and explain the difference between capture-by-value and capture-by-reference.
  2. You can construct `std::thread` objects, `join()` them, and explain why destroying a joinable thread calls `std::terminate`.
  3. You can write a GoogleTest test case using `TEST()`, `EXPECT_EQ`, and `ASSERT_*` macros, and run it with `ctest`.

- **Tasks**:

  **Lambda Expressions & Move Semantics**
  - [ ] Read learncpp.com lessons **20.6** (introduction to lambdas) and **20.7** (lambda captures).
  - [ ] Read learncpp.com lessons **16.5** (returning std::vector — introduction to move semantics) and **22.4** (`std::move`).
    - Note: You don't need to write move constructors for this project. The goal is understanding why `threads.emplace_back(std::thread(...))` works but `threads.push_back(some_thread)` doesn't (without `std::move`).
  - [ ] **Exercise 1**: Write a program that creates a `std::vector<int>` with values `{1, 2, 3, 4, 5}`, then uses `std::for_each` with a lambda that captures a `multiplier` by value and prints each element multiplied. Then change the capture to by-reference, modify `multiplier` inside the lambda, and observe the difference.
    - Expected output (capture by value): `2 4 6 8 10` (with `multiplier = 2`). Then (capture by reference, incrementing multiplier each call): `2 6 12 20 30`. This demonstrates how capture mode affects shared state — critical for understanding what `std::thread` captures from its caller.

  **`std::thread`, GoogleTest, and Scope-Exit Patterns**
  - [ ] Read the [std::thread cppreference page](https://en.cppreference.com/w/cpp/thread/thread.html) — focus on the constructor, `join()`, `detach()`, and the Examples section.
  - [ ] Read learncpp.com lesson **27.3** (exceptions, functions, and stack unwinding) — this explains how destructors run during exception propagation, which is why RAII-based thread joining matters.
  - [ ] Read the [GoogleTest Primer](https://google.github.io/googletest/primer.html) — focus on **Simple Tests**, **Assertions**, and **Test Fixtures**. Then read [Quickstart: Building with CMake](https://google.github.io/googletest/quickstart-cmake.html) for how `FetchContent_Declare`, `enable_testing()`, and `gtest_discover_tests()` work together.
  - [ ] **Exercise 2**: Write a program that spawns 4 `std::thread` instances, each printing its thread index and squaring a value in a shared `std::array<int, 4>` (each thread writes to its own index — no overlap). Join all 4 in a loop. Then write a GoogleTest test file that uses `TEST()` and `EXPECT_EQ` to verify each element is correctly squared. Build and run with `ctest`.
    - Expected output: `ctest` reports all tests passed. The console output from the threads may be interleaved (that's expected — threads run concurrently). This previews the exact pattern of Day 5: partitioned writes to non-overlapping regions, verified by unit tests.
    - Note: The threads in this exercise write to non-overlapping indices of a shared array — the same safety model as the quadrant transforms. Understanding why this is safe without mutexes is the key insight for Day 5.

- **Resources**:
  - [Introduction to lambdas (20.6)](https://www.learncpp.com/cpp-tutorial/introduction-to-lambdas-anonymous-functions/) — Lambda syntax, default captures, return types. Continue to [20.7 — Lambda captures](https://www.learncpp.com/cpp-tutorial/lambda-captures/) for `[&]`, `[=]`, and explicit captures — critical for understanding how `std::thread` receives its callable.
  - [Returning std::vector, and an introduction to move semantics (16.5)](https://www.learncpp.com/cpp-tutorial/returning-stdvector-and-an-introduction-to-move-semantics/) — Gentle first exposure to move semantics. Then read [22.4 — std::move](https://www.learncpp.com/cpp-tutorial/stdmove/) for the mechanics of ownership transfer.
  - [std::thread — cppreference.com](https://en.cppreference.com/w/cpp/thread/thread.html) — Full reference. Focus on **Member functions** (constructor, `join`, `detach`) and the note that destroying a joinable thread calls `std::terminate`. Read the **Example** section.
  - [Exceptions, functions, and stack unwinding (27.3)](https://www.learncpp.com/cpp-tutorial/exceptions-functions-and-stack-unwinding/) — How exceptions unwind the call stack and trigger destructors. Key for understanding why threads must be joined in a scope-exit RAII pattern.
  - [GoogleTest Primer](https://google.github.io/googletest/primer.html) — `TEST()`, `EXPECT_*`, `ASSERT_*`, test fixtures. Read **Simple Tests** and **Assertions** sections.
  - [Quickstart: Building with CMake — GoogleTest](https://google.github.io/googletest/quickstart-cmake.html) — `FetchContent_Declare`, `enable_testing()`, `gtest_discover_tests()`. Read the full page.

---

## Day 5: Multi-Threaded Quadrant Transforms with GIL Release

**Focus**: `std::thread`, `py::gil_scoped_release`, quadrant partitioning, thread join safety, data races
**Load**: Level 4

- **Objectives**:
  1. A free function partitions the image into four quadrants and dispatches four `std::thread` instances — one per quadrant — that each apply an in-place pixel transformation concurrently.
  2. The GIL is released **before** threads are spawned and reacquired **after** all threads are joined — C++ threads never touch the Python API.
  3. Thread lifetime is fully RAII-managed: if any thread throws, all threads are joined before the exception propagates.

- **Tasks**:

  **Quadrant Logic (C++)**
  - [ ] Implement a standalone function `transform_quadrant(uint8_t* base, int row_start, int row_end, int width, int channels, int row_stride)` in `src/transforms.cpp` that applies a per-pixel operation (e.g., bitwise invert `pixel = 255 - pixel`) to every byte in the given row range.
    - Note: Keep the transform trivial. The point is proving concurrent memory access, not image processing. Inversion is ideal because it's visually verifiable and commutative (applying it twice restores the original).
  - [ ] Implement `process_quadrants(ImageBuffer& buf)` that computes the four row-ranges `[0, H/4)`, `[H/4, H/2)`, `[H/2, 3H/4)`, `[3H/4, H)`, constructs four `std::thread` objects each calling `transform_quadrant` with the appropriate slice, and joins all four.
    - Note: Use integer division. If height is not divisible by 4, the last quadrant absorbs the remainder rows — off-by-one here is a classic bug.
  - [ ] Wrap all four `std::thread` objects in a local `std::vector<std::thread>` and join them in a scope-exit loop. This guarantees no thread is left detached if an earlier join throws.

  **GIL Release (Binding)**
  - [ ] Bind `process_quadrants` in `src/module.cpp` using `py::call_guard<py::gil_scoped_release>()` so the GIL is released for the entire duration of the C++ call.
    - Note: `call_guard` is cleaner than manually scoping `py::gil_scoped_release release;` inside the function body. It applies RAII at the binding layer rather than polluting C++ logic.

  **C++ Unit Tests (GoogleTest)**
  - [ ] Write `tests/test_transform_quadrant.cpp` — test `transform_quadrant` on a small heap-allocated buffer (e.g. 8x8x3). Verify every byte is inverted. Verify double-inversion restores the original.
    - Note: This is a unit test — it tests pure C++ logic in isolation, no Python involved. It catches transform correctness bugs before the FFI layer adds complexity.
  - [ ] Write `tests/test_quadrant_partitioning.cpp` — test that `process_quadrants` covers `[0, H)` exactly for heights divisible by 4, not divisible by 4, and edge cases (height < 4, height = 1).
  - [ ] Write `tests/test_thread_safety.cpp` — run `process_quadrants` on a known buffer and verify the result is deterministic across repeated runs (non-overlapping writes should produce identical output every time).
    - Note: This does not replace ThreadSanitizer — it catches logical races (wrong output), not memory races.

  **Thread Safety Audit**
  - [ ] Verify that no two threads write to overlapping row ranges — the quadrant boundaries must be non-overlapping and cover `[0, H)` exactly.
  - [ ] Confirm that `ImageBuffer` members read during threading (`data_ptr`, `width`, `channels`, `row_stride`) are `const` or effectively immutable after construction — no synchronization needed.
  - [ ] Verify that no thread calls any `py::` API — all parameters are raw C++ types (`uint8_t*`, `int`).

- **Acceptance Criteria**:
  - After calling `process_quadrants` from Python on a white `(4096, 4096, 3)` image (`np.full(..., 255)`), every pixel in the result array is `0` (bitwise inversion).
  - Calling `process_quadrants` twice restores the original array exactly (idempotency proof of inversion).
  - Running under `ThreadSanitizer` (`cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread" ...`) reports zero data races.
  - The `data_ptr()` value is unchanged before and after `process_quadrants` — the buffer was mutated in-place, never reallocated.
  - `ctest --test-dir build --output-on-failure` passes all C++ unit tests.

- **Resources**:
  - [Miscellaneous — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/misc.html) — GIL management in pybind11. Read the **Global Interpreter Lock (GIL)** section — it covers `gil_scoped_release`, `gil_scoped_acquire`, and `call_guard`. This is the single most critical section for this day's work.
  - [std::thread — cppreference.com](https://en.cppreference.com/w/cpp/thread/thread.html) — Re-read the constructor and `join()` sections from Day 4. Focus on the precondition: destroying a joinable thread calls `std::terminate`.
  - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — Re-read **FetchContent** and **pybind11_add_module** for linking additional source files into the module target.
  - [Quickstart: Building with CMake — GoogleTest](https://google.github.io/googletest/quickstart-cmake.html) — Reference for adding new test executables and discovering them with `gtest_discover_tests()`.
  - [ThreadSanitizer — Clang documentation](https://clang.llvm.org/docs/ThreadSanitizer.html) — What ThreadSanitizer is, how to enable it (`-fsanitize=thread`), and how to read its output. Read the full page — it's short. ThreadSanitizer is a compiler instrumentation tool that detects data races at runtime. You enable it by adding `-fsanitize=thread` to both compile and link flags, run your program normally, and it prints any detected races to stderr. No races = no output.
  - [Google Sanitizers Wiki — ThreadSanitizer](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual) — Practical usage guide. Read **Usage** and **Typical Workflow** for how to interpret race reports (what "read/write of size N" and "previous write" mean). Skip **Suppressions** unless you hit false positives.

---

## Day 6: Performance Benchmarking and Validation Proof

**Focus**: Wall-clock timing, memory profiling, pytest-benchmark, scaling analysis, acceptance harness
**Load**: Level 3

- **Objectives**:
  1. A Python test suite quantitatively proves that the multi-threaded C++ path is faster than a single-threaded Python baseline on the same operation.
  2. Memory allocation is provably zero: the array's `ctypes.data` pointer and `__array_interface__` base address are unchanged across the entire pipeline.
  3. Thread scaling is measurable: benchmarks compare 1-thread vs 4-thread execution to show actual parallel speedup.

- **Tasks**:

  **Benchmark Harness (Python)**
  - [ ] Write `python/bench.py` using `time.perf_counter_ns()` to measure wall-clock time of `process_quadrants` on `(8192, 8192, 3)` uint8 arrays (192 MB payload). Print results in microseconds.
  - [ ] Implement a pure-Python baseline (`arr[:] = 255 - arr`) performing the same inversion, and time it identically. Print both results and the speedup ratio.
  - [ ] Add a `pytest-benchmark` test in `python/test_bench.py` that benchmarks both the C++ and Python paths using the `benchmark` fixture for statistically rigorous comparison (automatic calibration, warmup, percentile reporting).
    - Note: This is an integration test with performance measurement — it crosses the FFI boundary and measures real-world throughput.

  **Zero-Copy Proof (Python)**
  - [ ] In `python/test_validation.py`, capture `arr.ctypes.data` and `arr.__array_interface__['data'][0]` before and after calling `process_quadrants`. Assert both are identical — the same heap address, not a copy.
  - [ ] Assert `arr.base is None` (the array owns its memory) before the call, and `arr.base is None` after — confirming no view indirection was introduced.
  - [ ] Assert `np.shares_memory(arr, arr)` remains true and `arr.flags['OWNDATA']` is still `True` after the C++ call mutates it.

  **Scaling Test**
  - [ ] Add an optional `process_single_thread(ImageBuffer& buf)` binding that runs all four quadrants sequentially on one thread, for controlled comparison.
  - [ ] Benchmark 1-thread vs 4-thread on `(8192, 8192, 3)` and assert that the 4-thread path is at least 1.5x faster (conservative bound accounting for thread overhead on small payloads).

  **Full Acceptance Script**
  - [ ] Write `python/run_validation.py` that runs the complete validation sequence: construct array, confirm pointer, run transform, confirm pointer unchanged, confirm pixel values, print `PASS` or `FAIL` for each check.
  - [ ] The script must exit with code 0 only if every check passes.

- **Acceptance Criteria**:
  - `uv run pytest python/ -v` passes all tests with zero failures.
  - `uv run python python/bench.py` prints a measurable speedup (>1x) for the C++ threaded path over the Python baseline on an 8192x8192 image.
  - `uv run python python/run_validation.py` prints `PASS` for all checks: pointer stability, pixel correctness, double-inversion idempotency, and dimension integrity.
  - No test allocates a second array of the same size — memory high-water mark stays at ~1x the input payload.

- **Resources**:
  - [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) — Benchmark fixture for pytest. This is a Tier 5 tool — read it thoroughly. Start with **Usage** for how the `benchmark` fixture works: you pass a callable to `benchmark()` and it handles warmup, calibration, and statistical analysis automatically. Then read **Comparing** for how to compare multiple benchmarks side-by-side, and **Command line** for flags like `--benchmark-enable`, `--benchmark-disable`, `--benchmark-only`. Finally, read the output table format: `min`, `max`, `mean`, `stddev`, `rounds` — these are what you'll use to report speedup ratios.
  - [time.perf_counter_ns — Python docs](https://docs.python.org/3/library/time.html#time.perf_counter_ns) — Reference for the high-resolution timer used in the manual benchmark script. One paragraph — just confirms it returns nanoseconds as an integer.
  - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — Re-read **Array attributes** — you will need `ctypes.data`, `__array_interface__`, `flags`, and `base` for the zero-copy assertions.

---

## Day 7: DevOps — CI/CD Pipeline

**Focus**: GitHub Actions, automated lint, build, and test enforcement
**Load**: Level 2

- **Objectives**:
  1. Every push and PR to `main` triggers automated lint, type-check, build, and test.
  2. A failing lint, build, or test blocks the PR from merging.

- **Tasks**:

  **CI Pipeline (GitHub Actions)**
  - [ ] Create `.github/workflows/ci.yml`:
    - Trigger on push and PR to `main`.
    - Job 1 — **Lint & Type Check**: `uv sync`, `uv run ruff check`, `uv run ruff format --check`, `uv run mypy python/`.
    - Job 2 — **Build & Test**: Install system deps (`cmake`, `g++`), `cmake -B build && cmake --build build`, `ctest --test-dir build --output-on-failure` for C++ tests, then `PYTHONPATH=build uv run pytest python/ -v` for Python tests.
    - Note: Install `uv` in the runner via `curl -LsSf https://astral.sh/uv/install.sh | sh` or use the official `astral-sh/setup-uv` action.
  - [ ] Add CI status badge to `README.md`.

- **Acceptance Criteria**:
  - Pushing to `main` triggers CI; all jobs pass green.
  - A deliberate lint violation (e.g., unused import) causes the lint job to fail and exit non-zero.
  - A deliberate test failure causes the test job to fail.

- **Resources**:
  - [Understanding GitHub Actions](https://docs.github.com/en/actions/about-github-actions/understanding-github-actions) — Start here. This is a Tier 4 technology — read this intro page fully. It explains the core concepts: workflows, events, jobs, steps, runners, and actions. Without this mental model, the YAML syntax won't make sense.
  - [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions) — Reference for workflow YAML. Read **Events that trigger workflows** for `push`/`pull_request` config, **Using jobs in a workflow** for multi-job setup, and **steps** for how `run:` and `uses:` work. Use this as a reference while writing `ci.yml`, not as a read-through.
  - [GitHub Actions Quickstart](https://docs.github.com/en/actions/writing-workflows/quickstart) — A minimal working example. Read the full page — it walks through creating a `.github/workflows/` file, triggering it, and reading the logs. Do this before writing your own workflow.
  - [astral-sh/setup-uv — GitHub](https://github.com/astral-sh/setup-uv) — Official GitHub Action for installing uv in CI runners. Read the README for usage examples.

---

## Day 8: Delivery — Build Automation and Packaging

**Focus**: Makefile, Docker, README, project polish
**Load**: Level 2

- **Objectives**:
  1. The entire build-test cycle is reproducible via a single `make` target.
  2. A Dockerfile produces a self-contained image that builds and tests the project from scratch.
  3. The README provides a clear quick-start for anyone cloning the repo.

- **Tasks**:

  **Makefile**
  - [ ] Write a `Makefile` with common targets:
    - `make build` — `cmake -B build && cmake --build build`.
    - `make test` — `ctest --test-dir build --output-on-failure && PYTHONPATH=build uv run pytest python/ -v`.
    - `make bench` — `PYTHONPATH=build uv run pytest python/test_bench.py -v --benchmark-enable`.
    - `make lint` — `uv run ruff check && uv run ruff format --check && uv run mypy python/`.
    - `make clean` — `rm -rf build/`.
    - `make validate` — `PYTHONPATH=build uv run python python/run_validation.py`.

  **Docker**
  - [ ] Write a `Dockerfile` that:
    - Uses a minimal base image (e.g., `python:3.14-slim` or `ubuntu:24.04`).
    - Installs system deps (`cmake`, `g++`, `curl`).
    - Installs `uv`, runs `uv sync`.
    - Runs `make build && make test` as the build verification step.
    - Note: Use multi-stage builds to keep the final image small if desired, but the primary goal is reproducibility, not image size.
  - [ ] Add a `make docker` target that builds and runs the Docker image.

  **README**
  - [ ] Update `README.md` with:
    - Project description (one paragraph).
    - Prerequisites: Python 3.14, CMake ≥3.16, C++17 compiler, uv.
    - Quick start: `uv sync && make build && make test`.
    - Available `make` targets.
    - Architecture overview (reference CLAUDE.md or summarize).

- **Acceptance Criteria**:
  - `make build && make test` succeeds on a clean clone after only `uv sync`.
  - `make lint` catches a deliberate formatting violation and exits non-zero.
  - `docker build .` completes successfully and the container's test step passes.

- **Resources**:
  - [GNU Make Manual — Introduction](https://www.gnu.org/software/make/manual/html_node/Introduction.html) — Start here for Makefile basics. Read **An Introduction to Makefiles** and **How to Read This Manual** — this covers targets, prerequisites, and recipes. Then jump to [Quick Reference](https://www.gnu.org/software/make/manual/html_node/Quick-Reference.html) as a cheat sheet while writing. Skip everything about implicit rules and pattern matching — you won't need them.
  - [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Full Dockerfile syntax reference. Read **FROM**, **RUN**, **COPY**, and **WORKDIR** for the basics; read **Multi-stage builds** if you want to optimize image size. Skip **ARG** and **ONBUILD** unless you need them.
  - [Docker Get Started — Build and push your first image](https://docs.docker.com/get-started/introduction/build-and-push-first-image/) — Hands-on tutorial. Read this before the reference — it gives you a working mental model of the build → run cycle.
