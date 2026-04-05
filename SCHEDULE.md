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
    - [x] Initialize the Python project with `uv init` (generates `pyproject.toml`, `.python-version`, `uv.lock`). Add dependencies via `uv add numpy` and `uv add --dev pytest pytest-benchmark ruff mypy`. Run `uv sync` to create the `.venv`.
        - Note: Pin the NumPy version explicitly in `pyproject.toml` (e.g. `numpy>=2.4,<3`) so the ABI stays stable across rebuilds. Commit `uv.lock` for reproducible installs.
    - [x] Configure pre-commit hooks (`.pre-commit-config.yaml`):
        - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`
        - Local hooks: `ruff check --fix`, `ruff format`, `mypy .`, `pytest --cov` (all via `uv run`)
        - C++ hooks: `clang-format -i`, `clang-tidy`, `ctest --test-dir build --output-on-failure`
        - Note: Run `uv run pre-commit install` to activate. Hooks must pass before any commit lands.
    - [x] Configure ruff in `pyproject.toml`: enable rule sets `E`, `F`, `I` (isort), `UP` (pyupgrade), `NPY` (NumPy-specific).
    - [x] Configure strict mypy in `pyproject.toml` (`[tool.mypy]` with `strict = true`).
    - [x] Add a `.clang-format` file for C++ source formatting (`BasedOnStyle: Google`).
    - [x] Add a `.clang-tidy` config with `concurrency-*`, `google-*`, `performance-*`, `readability-*`, `bugprone-*`, `modernize-*` checks.
    - [x] Author the root `CMakeLists.txt` — `cmake_minimum_required(VERSION 3.16)`, `CMAKE_CXX_STANDARD 17`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`, `FetchContent` for pybind11 (`v3.0.2`) and GoogleTest (`v1.17.0`).
    - [x] Add a minimal `src/module.cpp` with `PYBIND11_MODULE` entry point exposing a `noop()` function returning a string literal.
    - [x] Add stub `src/transforms.cpp` and test files in `tests/` (`test_transform_quadrant.cpp`, `test_quadrant_partitioning.cpp`, `test_thread_safety.cpp`).
    - [x] Verify the full build-import cycle: `cmake -B build && cmake --build build`, then `PYTHONPATH=build uv run python -c "import threaded_image_ops; print(threaded_image_ops.noop())"` prints `hello from C++`.
    - [x] Symlink `compile_commands.json` to project root (`ln -s build/compile_commands.json .`) for clang-tidy LSP support.
    - [x] Add `build/`, `*.so`, `__pycache__/`, `.venv/`, and `compile_commands.json` to `.gitignore`.

- **Resources**:
    - [CMake Tutorial — Official Docs](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) — Step-by-step intro to CMake. Read **Step 1: Getting Started** for the four core commands (`cmake_minimum_required`, `project`, `add_executable`, `target_sources`); read **Step 2: Adding a Library** for `add_library` and `target_link_libraries`.
    - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — CMake integration for pybind11. Focus on the **FetchContent** and **pybind11_add_module** sections for how to pull pybind11 and build a `.so` from CMake.
    - [First steps — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/basics.html) — Covers `PYBIND11_MODULE`, `m.def()`, and basic type conversions. Read the full page — it's the minimum needed to write `module.cpp`.
    - [pre-commit documentation](https://pre-commit.com/) — Official docs for the pre-commit hook framework. Read the **Adding pre-commit plugins** and **Creating new hooks** sections for `.pre-commit-config.yaml` syntax and local hook definitions.
    - [Configuring Ruff](https://docs.astral.sh/ruff/configuration/) — Ruff linter/formatter configuration in `pyproject.toml`. Read the **Rule selection** section for how `select` and `ignore` work with rule code prefixes.
    - [mypy configuration file](https://mypy.readthedocs.io/en/stable/config_file.html) — mypy config reference. Read the **The mypy configuration file** section for `[tool.mypy]` in `pyproject.toml`; skim **Strict mode** for which flags `strict = true` enables.
    - [Clang-Format Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html) — Full reference for `.clang-format`. Read the **Configuring Style with clang-format** intro and **BasedOnStyle** for how predefined styles (Google, LLVM) work.
    - [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/) — clang-tidy overview and check categories. Read the **Configuring Checks** section for `.clang-tidy` YAML syntax; browse the [checks list](https://clang.llvm.org/extra/clang-tidy/checks/list.html) for what `bugprone-*`, `modernize-*`, etc. cover.

---

## Day 1: C++ Foundations I — Pointers, Memory, and Classes
**Focus**: Header files, raw pointers, pointer arithmetic, classes, constructors, destructors, RAII, const correctness, exceptions
**Load**: Level 3

- **Objectives**:
    1. You understand the C++ compilation model — why code is split into `.hpp` headers and `.cpp` source files, and what the compiler and linker each do.
    2. You can declare, dereference, and do arithmetic on raw pointers to walk through a contiguous byte buffer.
    3. You can define a class with a validating constructor (member initializer list), `const` getters, private data members, and a defaulted destructor.
    4. You understand RAII — tying resource lifetime to object lifetime — and why `ImageBuffer`'s destructor is `= default` (it manages access, not allocation).

- **Tasks**:

    **Compilation Model & Headers**
    - [ ] Read learncpp.com lessons **0.5** (compiler, linker, libraries), **2.11** (header files), and **2.12** (header guards / `#pragma once`).
    - [ ] Read **15.2** (classes and header files) — this explains why the class definition goes in `.hpp` and method implementations can go in `.cpp`.

    **Raw Pointers & Pointer Arithmetic**
    - [ ] Read learncpp.com lessons **12.7** (introduction to pointers), **12.9** (pointers and `const`), and **17.9** (pointer arithmetic and subscripting).
    - [ ] Write a small standalone program (outside the project) that heap-allocates a `uint8_t` array, walks it with pointer arithmetic to set each byte to a value, and prints the results. This mirrors what `transform_quadrant` will do in Day 4.

    **Classes, Constructors, and Destructors**
    - [ ] Read learncpp.com lessons **14.3** (member functions), **14.4** (`const` member functions), **14.5** (public/private access specifiers), **14.9** (constructors), **14.10** (member initializer lists), and **15.4** (destructors).
    - [ ] Write a small class that: takes dimensions in its constructor (via member initializer list), validates them (throws `std::invalid_argument` if invalid), exposes `const` getters, and has an explicitly defaulted destructor (`= default`). This directly previews the `ImageBuffer` class.

    **RAII & Exceptions**
    - [ ] Read learncpp.com lesson **22.1** (introduction to smart pointers and move semantics) — the first half covers RAII with concrete examples. Read the [RAII page on cppreference](https://en.cppreference.com/w/cpp/language/raii.html) for the canonical definition.
    - [ ] Read learncpp.com lesson **27.2** (basic exception handling: `throw`, `try`, `catch`). You only need this one lesson — pybind11 auto-translates C++ exceptions to Python exceptions.

    **Templates (Reading Only)**
    - [ ] Skim learncpp.com lesson **11.6** (function templates). The goal is to recognize angle-bracket syntax like `py::array_t<uint8_t>` — you will not write templates in this project.

- **Resources**:
    - [Introduction to the compiler, linker, and libraries (0.5)](https://www.learncpp.com/cpp-tutorial/introduction-to-the-compiler-linker-and-libraries/) — How C++ goes from source files to an executable. Continue to [2.11 — Header files](https://www.learncpp.com/cpp-tutorial/header-files/) and [2.12 — Header guards](https://www.learncpp.com/cpp-tutorial/header-guards/) for `#include` mechanics and `#pragma once`. Revisit [15.2 — Classes and header files](https://www.learncpp.com/cpp-tutorial/classes-and-header-files/) after reading the classes lessons.
    - [Introduction to pointers (12.7)](https://www.learncpp.com/cpp-tutorial/introduction-to-pointers/) — Pointer declaration, dereferencing, and nullptr. Continue to [12.9 — Pointers and const](https://www.learncpp.com/cpp-tutorial/pointers-and-const/) for const pointer rules, then [17.9 — Pointer arithmetic and subscripting](https://www.learncpp.com/cpp-tutorial/pointer-arithmetic-and-subscripting/) for the offset math behind buffer traversal.
    - [Member functions (14.3)](https://www.learncpp.com/cpp-tutorial/member-functions/) — Start of the classes arc. Read sequentially through [14.4](https://www.learncpp.com/cpp-tutorial/const-class-objects-and-const-member-functions/) (const members), [14.5](https://www.learncpp.com/cpp-tutorial/public-and-private-members-and-access-specifiers/) (access specifiers), skip to [14.9](https://www.learncpp.com/cpp-tutorial/introduction-to-constructors/) (constructors), [14.10](https://www.learncpp.com/cpp-tutorial/constructor-member-initializer-lists/) (member initializer lists). Finish with [15.4 — Introduction to destructors](https://www.learncpp.com/cpp-tutorial/introduction-to-destructors/).
    - [Introduction to smart pointers and move semantics (22.1)](https://www.learncpp.com/cpp-tutorial/introduction-to-smart-pointers-move-semantics/) — The first half explains RAII: acquiring resources in constructors, releasing in destructors. Also read the [RAII page on cppreference](https://en.cppreference.com/w/cpp/language/raii.html) — it's short and canonical.
    - [Basic exception handling (27.2)](https://www.learncpp.com/cpp-tutorial/basic-exception-handling/) — `throw`, `try`, `catch` mechanics. One lesson is enough — skip the rest of chapter 27 for now.
    - [Function templates (11.6)](https://www.learncpp.com/cpp-tutorial/function-templates/) — Skim to recognize `template<typename T>` and angle-bracket type parameters. You don't need to write templates — just read them.

---

## Day 2: RAII Wrapper and Zero-Copy NumPy Buffer Binding
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
        - Note: This is the RAII discipline under test: the class manages *access lifetime*, not *allocation lifetime*. The destructor's job is to guarantee that no dangling work (threads) outlives the buffer view.

    **Pybind11 Binding**
    - [ ] In `src/module.cpp`, bind `ImageBuffer` via `py::class_<ImageBuffer>` with an `__init__` that accepts `py::array_t<uint8_t>`.
    - [ ] Expose read-only properties: `.height`, `.width`, `.channels`.
    - [ ] Bind a `data_ptr()` method that returns the raw pointer as `uintptr_t` — this is the diagnostic hook Python will use to prove zero-copy.

    **Verification Script**
    - [ ] Write `python/test_zerocopy.py`: allocate a NumPy array with `np.zeros((4096, 4096, 3), dtype=np.uint8)`, pass it to `ImageBuffer`, and assert that `buf.data_ptr() == arr.ctypes.data` — same integer address means zero copies.
    - [ ] Assert `buf.height == 4096`, `buf.width == 4096`, `buf.channels == 3`.

- **Acceptance Criteria**:
    - `uv run python python/test_zerocopy.py` exits 0 and prints no assertion errors.
    - The `data_ptr()` returned by the C++ side is byte-identical to `ndarray.ctypes.data` on the Python side — proving no intermediate copy was allocated.
    - Passing a 2D array (`np.zeros((100, 100))`) raises a Python `ValueError` originating from the C++ `std::invalid_argument`.

- **Resources**:
    - [NumPy — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) — Official guide to `py::array_t`, buffer protocol, and stride access. Read the **Buffer protocol** and **Arrays** sections for how `py::buffer_info` maps to NumPy internals; the **Vectorizing functions** section is irrelevant here.
    - [Object-oriented code — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) — Binding C++ classes to Python. Focus on **Creating bindings for a custom type** and **Instance and static fields** — these cover `py::class_`, `py::init`, and property definitions.
    - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy's memory model. Read **Internal memory layout of an ndarray** for the stride formula and contiguity guarantees; skim **Array attributes** for `.ctypes`, `.strides`, `.flags`.
    - [RAII — cppreference.com](https://en.cppreference.com/w/cpp/language/raii.html) — Canonical definition of RAII. Read the full page — it's short. Pay attention to the **Standard library** section listing which STL types follow RAII, and the bad-vs-good mutex example.

---

## Day 3: C++ Foundations II — Threads, Lambdas, and Concurrency
**Focus**: Lambda expressions, lambda captures, `std::thread`, move semantics, object lifetime, scope-exit patterns
**Load**: Level 3

- **Objectives**:
    1. You can write lambda expressions with explicit captures and understand the difference between capture-by-value and capture-by-reference.
    2. You understand `std::thread` construction, `join()`, and why destroying a joinable thread calls `std::terminate`.
    3. You understand move semantics enough to know why `std::thread` is move-only and how `std::vector<std::thread>` works with `emplace_back`.
    4. You can reason about object lifetime and scope — why threads must be joined before the data they reference goes out of scope.

- **Tasks**:

    **Lambda Expressions**
    - [ ] Read learncpp.com lessons **20.6** (introduction to lambdas) and **20.7** (lambda captures).
    - [ ] Write a small program that creates a vector of integers, then uses a lambda with explicit captures to transform each element. Experiment with `[&]`, `[=]`, and `[x, &y]` to see the difference.

    **Move Semantics**
    - [ ] Read learncpp.com lessons **16.5** (returning std::vector — introduction to move semantics), **22.1** (revisit the second half on move semantics), **22.3** (move constructors and move assignment), and **22.4** (`std::move`).
    - [ ] Understand the key insight: a move *transfers ownership* of resources instead of copying them. `std::thread` is move-only because two threads can't both own the same execution.
        - Note: You don't need to write move constructors for this project. The goal is understanding why `threads.emplace_back(std::thread(...))` works but `threads.push_back(some_thread)` doesn't (without `std::move`).

    **`std::thread` and Concurrency**
    - [ ] Read the [std::thread cppreference page](https://en.cppreference.com/w/cpp/thread/thread.html) — focus on the constructor, `join()`, `detach()`, and the Examples section.
    - [ ] Read the [Educative tutorial on modern C++ multithreading](https://www.educative.io/blog/modern-multithreading-and-concurrency-in-cpp) — focus on thread creation and joining sections.
    - [ ] Write a small program that spawns 4 threads, each printing its thread ID and a message. Join all 4 in a loop. Then intentionally comment out the joins and observe the crash (`std::terminate`).

    **Scope, Lifetime, and RAII for Threads**
    - [ ] Read learncpp.com lesson **27.3** (exceptions, functions, and stack unwinding) — this explains how destructors run during exception propagation, which is why RAII-based thread joining matters.
    - [ ] Understand the pattern: store threads in a `std::vector<std::thread>`, join them all in a scope-exit loop. If any thread's work throws before the loop, the remaining unjoined threads would crash the program — hence the RAII wrapper approach.

- **Resources**:
    - [Introduction to lambdas (20.6)](https://www.learncpp.com/cpp-tutorial/introduction-to-lambdas-anonymous-functions/) — Lambda syntax, default captures, and return types. Continue to [20.7 — Lambda captures](https://www.learncpp.com/cpp-tutorial/lambda-captures/) for `[&]`, `[=]`, and explicit captures — critical for understanding how `std::thread` receives its callable.
    - [Returning std::vector, and an introduction to move semantics (16.5)](https://www.learncpp.com/cpp-tutorial/returning-stdvector-and-an-introduction-to-move-semantics/) — Gentle first exposure to move semantics. Then read [22.3 — Move constructors and move assignment](https://www.learncpp.com/cpp-tutorial/move-constructors-and-move-assignment/) and [22.4 — std::move](https://www.learncpp.com/cpp-tutorial/stdmove/) for the mechanics of ownership transfer.
    - [std::thread — cppreference.com](https://en.cppreference.com/w/cpp/thread/thread.html) — Full reference. Focus on **Member functions** (constructor, `join`, `detach`) and the note that destroying a joinable thread calls `std::terminate`. Read the **Example** section for working code.
    - [A tutorial on modern multithreading and concurrency in C++](https://www.educative.io/blog/modern-multithreading-and-concurrency-in-cpp) — Beginner-friendly walkthrough with examples. Read the thread creation and joining sections; skip the mutex/condition variable parts until later.
    - [Exceptions, functions, and stack unwinding (27.3)](https://www.learncpp.com/cpp-tutorial/exceptions-functions-and-stack-unwinding/) — How exceptions unwind the call stack and trigger destructors. Key for understanding why threads must be joined in a scope-exit RAII pattern.

---

## Day 4: Multi-Threaded Quadrant Transforms with GIL Release
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
    - [ ] Wrap all four `std::thread` objects in a local `std::vector<std::thread>` and join them in a scope-exit loop (or use a small RAII thread-joiner wrapper). This guarantees no thread is left detached if an earlier join throws.

    **GIL Release (Binding)**
    - [ ] Bind `process_quadrants` in `src/module.cpp` using `py::call_guard<py::gil_scoped_release>()` so the GIL is released for the entire duration of the C++ call.
        - Note: `call_guard` is cleaner than manually scoping `py::gil_scoped_release release;` inside the function body. It applies RAII at the binding layer rather than polluting C++ logic.
    - [ ] Alternatively, if the function needs pre- or post-GIL work, use an explicit `py::gil_scoped_release` block within the bound lambda. Document which approach was chosen and why.

    **C++ Unit Tests (Google Test)**
    - [ ] Add a `tests/` directory for C++ tests. Add a CMake target (`add_executable` + `target_link_libraries` with `GTest::gtest_main`) and enable `gtest_discover_tests()`.
    - [ ] Write `tests/test_transform_quadrant.cpp` — test `transform_quadrant` on a small heap-allocated buffer (e.g. 8x8x3). Verify every byte is inverted. Verify double-inversion restores the original.
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
    - [std::thread — cppreference.com](https://en.cppreference.com/w/cpp/thread/thread.html) — Full reference for `std::thread`. Read **Member functions** (constructor, `join`, `detach`) and note the precondition: destroying a joinable thread calls `std::terminate`.
    - [Build systems — pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html) — CMake integration details. Read the **FetchContent** and **pybind11_add_module** sections for linking additional source files into the module target.
    - [pybind/cmake_example — GitHub](https://github.com/pybind/cmake_example) — Reference implementation of a CMake-based pybind11 project. Study the `CMakeLists.txt` for how `pybind11_add_module` is invoked and how source files are listed.
    - [Quickstart: Building with CMake — GoogleTest](https://google.github.io/googletest/quickstart-cmake.html) — Official guide to FetchContent + GoogleTest. Read the full page — it walks through `FetchContent_Declare`, `enable_testing()`, and `gtest_discover_tests()` end to end.

---

## Day 5: Performance Benchmarking and Validation Proof
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

    **Zero-Copy Proof**
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
    - No test allocates a second array of the same size — memory high-water mark stays at ~1x the input payload (verify via `tracemalloc` or `/proc/self/status` VmRSS).

- **Resources**:
    - [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/) — Benchmark fixture for pytest. Read the **Usage** section for how to pass callables to `benchmark()` and how to interpret the output table (min, max, mean, stddev, rounds).
    - [Google Benchmark User Guide](https://google.github.io/benchmark/user_guide.html) — C++-side microbenchmark reference. Read **Passing Arguments** and **Multithreaded Benchmarks** if you want to add C++ benchmarks alongside the Python ones; otherwise use this as a mental model for rigorous benchmarking methodology.
    - [The N-dimensional array (ndarray) — NumPy Manual](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — Memory layout reference. Re-read **Internal memory layout of an ndarray** and **Array attributes** — you will need `ctypes.data`, `__array_interface__`, `flags`, and `base` to write the zero-copy assertions.

---

## Day 6: DevOps & Delivery
**Focus**: GitHub Actions CI, Makefile, build automation, lint enforcement
**Load**: Level 3

- **Objectives**:
    1. Every push and PR to `main` triggers automated lint, type-check, build, and test.
    2. The entire build-test cycle is reproducible via a single `make` target.

- **Tasks**:

    **CI Pipeline (GitHub Actions)**
    - [ ] Create `.github/workflows/ci.yml`:
        - Trigger on push and PR to `main`.
        - Job 1 — **Lint & Type Check**: `uv sync`, `uv run ruff check`, `uv run ruff format --check`, `uv run mypy python/`.
        - Job 2 — **Build & Test**: `cmake -B build && cmake --build build`, `ctest --test-dir build --output-on-failure` for C++ tests, then `uv run pytest python/ -v` for Python tests.
        - Note: Install system dependencies (`cmake`, `g++`) in the runner before the build step.
    - [ ] Add CI status badge to `README.md`.

    **Makefile**
    - [ ] Write a `Makefile` with common targets:
        - `make build` — `cmake -B build && cmake --build build`.
        - `make test` — `ctest --test-dir build --output-on-failure && uv run pytest python/ -v`.
        - `make bench` — `uv run pytest python/test_bench.py -v --benchmark-enable`.
        - `make lint` — `uv run ruff check && uv run ruff format --check && uv run mypy python/`.
        - `make clean` — `rm -rf build/`.
        - `make validate` — `uv run python python/run_validation.py`.
    - [ ] Update `README.md` with quick start: prerequisites, `uv sync`, `make build`, `make test`.

- **Acceptance Criteria**:
    - Pushing to `main` triggers CI; all jobs pass green.
    - `make build && make test` succeeds on a clean clone after only `uv sync`.
    - `make lint` catches a deliberate formatting violation and exits non-zero.

- **Resources**:
    - [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions) — Reference for workflow triggers, jobs, and steps. Read **Events that trigger workflows** for `push`/`pull_request` config and **Using jobs in a workflow** for multi-job setup.
    - [CMake GitHub Actions Setup](https://github.com/lukka/get-cmake) — Action for installing CMake in CI runners. Use this if the runner's default CMake is too old for your `cmake_minimum_required`.
