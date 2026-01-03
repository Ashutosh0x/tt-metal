# Prompt for Implementing Simplified Compute Kernel Syntax in TT-Metal

> **IMPORTANT**: This prompt is best-effort guidance based on prior exploration. Verify any claims that seem unclear or incorrect by checking the actual codebase. The architecture may have changed since this was written.

---

## Goal

Enable compute kernels to use the simplified syntax `void kernel_main() { ... }` (like dataflow kernels) instead of the legacy boilerplate `namespace NAMESPACE { void MAIN { ... } }`.

## Background You Need to Know

**Why the boilerplate exists:**
- Compute kernels run on 3 TRISC processors (Unpack, Math, Pack) on each Tensix core
- One kernel source is compiled 3 times with different defines (`TRISC_UNPACK`, `TRISC_MATH`, `TRISC_PACK`)
- Each compilation produces a different entry point: `unpack_main()`, `math_main()`, `pack_main()`
- These must be in namespaces `chlkc_unpack`, `chlkc_math`, `chlkc_pack` respectively
- The framework calls these via `chlkc_list.h`: `chlkc_unpack::unpack_main()`, etc.

**How code generation currently works** (verify in `tt_metal/jit_build/genfiles.cpp`):
- `jit_build_genfiles_triscs_src()` generates 3 files: `chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`
- Each file has a prolog (`#define TRISC_*`) followed by kernel source
- Legacy kernels use macros `NAMESPACE` and `MAIN` which expand based on `TRISC_*` defines

**Key insight - what DOESN'T work:**
- ❌ Wrapping the `#include` in a namespace - this puts the kernel's `#include` directives inside the namespace, breaking compilation
- ❌ Using `#define kernel_main chlkc_unpack::unpack_main` - C++ doesn't allow defining a function with qualified name from outside the namespace

**What DOES work:**
- ✅ Source transformation: Read kernel source, find `void kernel_main()`, split into preamble (includes) and function body, wrap only the function in namespace

---

## Implementation

### Step 1: Add `get_content()` method to KernelSource

In `tt_metal/impl/kernels/kernel.hpp`, add a method to `KernelSource` struct:

```cpp
struct KernelSource {
    // ... existing members ...

    // Returns the actual source code content.
    // For FILE_PATH: reads and returns file contents.
    // For SOURCE_CODE: returns the source string directly.
    std::string get_content() const {
        if (source_type_ == SourceType::FILE_PATH) {
            std::ifstream file(path_);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open kernel source file: " + path_.string());
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }
        return source_;
    }
};
```

You'll need to add includes at the top of the header:
```cpp
#include <fstream>
#include <sstream>
```

### Step 2: Modify `genfiles.cpp`

Add these includes if not present:
```cpp
#include <cstring>
#include <regex>
```

Add these helper functions in the anonymous namespace:

```cpp
namespace {

// Detects simplified syntax by checking for "kernel_main" identifier.
// Legacy kernels use the MAIN macro instead, so this is unambiguous.
bool uses_simplified_kernel_syntax(const std::string& source) {
    return source.find("kernel_main") != std::string::npos;
}

// Finds "void kernel_main() {" pattern with flexible whitespace.
// Returns position of "void", or string::npos if not found.
size_t find_kernel_main_definition(const std::string& source) {
    std::regex pattern(R"(\bvoid\s+kernel_main\s*\(\s*\)\s*\{)");
    std::smatch match;
    if (std::regex_search(source, match, pattern)) {
        return static_cast<size_t>(match.position());
    }
    return std::string::npos;
}

// Transforms simplified kernel to legacy format:
//   - Splits at "void kernel_main()"
//   - Preamble (#includes) stays outside namespace
//   - Function body wrapped in namespace, renamed to func_name
std::string transform_to_legacy_syntax(
    const std::string& source,
    const char* ns_name,
    const char* func_name) {

    size_t func_pos = find_kernel_main_definition(source);
    if (func_pos == std::string::npos) {
        throw std::runtime_error("Could not find 'void kernel_main() {' in source");
    }

    std::string preamble = source.substr(0, func_pos);
    std::string function_part = source.substr(func_pos);

    // Rename kernel_main -> func_name
    size_t name_pos = function_part.find("kernel_main");
    if (name_pos != std::string::npos) {
        function_part.replace(name_pos, strlen("kernel_main"), func_name);
    }

    std::ostringstream result;
    result << preamble;
    result << "namespace " << ns_name << " {\n";
    result << function_part;
    result << "\n}  // namespace " << ns_name << "\n";
    return result.str();
}

// Generates TRISC prolog: #define + #include for defines_generated.h
std::string build_trisc_prolog(const char* trisc_define) {
    std::ostringstream prolog;
    prolog << "#define " << trisc_define << "\n";
    prolog << "#include \"defines_generated.h\"\n";
    return prolog.str();
}

}  // namespace
```

### Step 3: Modify `jit_build_genfiles_triscs_src()`

The changes are minimal - only add syntax detection and conditional transformation:

```cpp
void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    const std::string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    const std::string unpack_cpp = out_dir + "chlkc_unpack.cpp";
    const std::string math_cpp = out_dir + "chlkc_math.cpp";
    const std::string pack_cpp = out_dir + "chlkc_pack.cpp";

    // Read content for syntax detection (needed for both paths)
    const std::string kernel_content = kernel_src.get_content();
    const bool simplified = uses_simplified_kernel_syntax(kernel_content);

    if (simplified) {
        log_trace(tt::LogBuildKernels, "Detected simplified compute kernel syntax (kernel_main)");
    }

    // Build prologs (same for both syntaxes)
    const std::string unpack_prolog = build_trisc_prolog("TRISC_UNPACK");
    const std::string math_prolog = build_trisc_prolog("TRISC_MATH");
    const std::string pack_prolog = build_trisc_prolog("TRISC_PACK");

    // Determine kernel source for each TRISC.
    //
    // Why the if-else structure is necessary:
    // - Simplified syntax: MUST transform source, so we inline the transformed content
    // - Legacy syntax: use existing get_kernel_source_to_include() which returns:
    //   - FILE_PATH: #include directive (preserves file refs in compiler errors)
    //   - SOURCE_CODE: the source directly
    std::string unpack_src, math_src, pack_src;
    if (simplified) {
        unpack_src = transform_to_legacy_syntax(kernel_content, "chlkc_unpack", "unpack_main");
        math_src = transform_to_legacy_syntax(kernel_content, "chlkc_math", "math_main");
        pack_src = transform_to_legacy_syntax(kernel_content, "chlkc_pack", "pack_main");
    } else {
        // Legacy: use existing helper that handles FILE_PATH vs SOURCE_CODE appropriately
        const std::string src = get_kernel_source_to_include(kernel_src);
        unpack_src = math_src = pack_src = src;
    }

    // Generate the three TRISC source files
    { std::ofstream f(unpack_cpp); f << unpack_prolog << unpack_src; }
    { std::ofstream f(math_cpp);   f << math_prolog << math_src; }
    { std::ofstream f(pack_cpp);   f << pack_prolog << pack_src; }

    // Generate defines_generated.h with user-defined macros
    const std::string defines_path = out_dir + "defines_generated.h";
    std::ofstream defines_file(defines_path);
    settings.process_defines([&defines_file](const std::string& name, const std::string& value) {
        defines_file << "#define " << name << " " << value << "\n";
    });
}
```

**Note**: The if-else structure is intentional: legacy FILE_PATH kernels use `#include` which preserves source file references in compiler error messages. Simplified syntax requires source transformation, so we must inline the transformed content.

---

## Testing

1. **Test legacy syntax still works**: Ensure an existing kernel with `namespace NAMESPACE { void MAIN { } }` compiles and runs correctly.

2. **Test simplified syntax**: Modify a compute kernel to use the new syntax:

```cpp
// Change FROM:
namespace NAMESPACE {
void MAIN {
    // kernel body
}
}

// Change TO:
void kernel_main() {
    // same kernel body
}
```

3. **Clear kernel cache and run**:
```bash
rm -rf ~/.cache/tt-metal-cache/*/*/kernels/<kernel_name>
./build_Release/programming_examples/<example_name>
```

4. **Verify generated output**: Check that generated `chlkc_*.cpp` files have correct structure:
```cpp
#define TRISC_UNPACK
#include "defines_generated.h"
// ... kernel's #includes here (outside namespace) ...

namespace chlkc_unpack {
void unpack_main() {
    // ... kernel body ...
}
}  // namespace chlkc_unpack
```

---
