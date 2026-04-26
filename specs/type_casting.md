# nanobind Type Casting and `std::bad_cast` Troubleshooting

A practical reference for nanobind's three type conversion mechanisms, how
`std::bad_cast` arises in each, and a systematic method for isolating and
fixing it.

---

## 1. The Three Type Conversion Mechanisms

nanobind moves data between C++ and Python through three distinct mechanisms.
Understanding which one is in use for a given parameter is the first step in
diagnosing any cast failure.

### 1.1 Bindings

A C++ type is registered with `nb::class_<T>()`. nanobind manages a Python
wrapper object whose lifetime mirrors the C++ object. No conversion of the
data takes place — Python holds a reference to the same memory.

```cpp
nb::class_<MyType>(m, "MyType")
    .def("value", &MyType::value);
```

**Cast behaviour:** Fails with `TypeError` at call time if the Python object
is not an instance of the registered class. Does **not** produce
`std::bad_cast` at import or stub-generation time.

### 1.2 Wrappers

nanobind's own wrapper types: `nb::object`, `nb::list`, `nb::dict`,
`nb::tuple`, `nb::callable`, and similar. These wrap a `PyObject*` and
carry no type information beyond "is a Python object".

```cpp
.def("process",
    [](nb::list items) { /* ... */ })
```

**Cast behaviour:** `nb::cast<T>()` on a wrapper throws `nb::cast_error` (a
subclass of `std::runtime_error`) if the runtime type doesn't match, not
`std::bad_cast`.

### 1.3 Type Casters

The mechanism for STL types and other types that need value-semantic
conversion between Python and C++. Activated by including the appropriate
header. nanobind provides built-in casters for:

| Header | Types covered |
|---|---|
| `nanobind/stl/string.h` | `std::string` |
| `nanobind/stl/vector.h` | `std::vector<T>` |
| `nanobind/stl/map.h` | `std::map<K,V>`, `std::unordered_map<K,V>` |
| `nanobind/stl/optional.h` | `std::optional<T>` |
| `nanobind/stl/pair.h` | `std::pair<T1,T2>` |
| `nanobind/stl/tuple.h` | `std::tuple<...>` |
| `nanobind/stl/complex.h` | `std::complex<T>` |
| `nanobind/stl/shared_ptr.h` | `std::shared_ptr<T>` |
| `nanobind/stl/unique_ptr.h` | `std::unique_ptr<T>` |
| `nanobind/ndarray.h` | NumPy / DLPack arrays |

**This is where `std::bad_cast` originates.** Type casters work by casting
internal nanobind metadata structs at registration and introspection time. If
a type has no registered caster, or if caster metadata is malformed, the
internal `dynamic_cast` or `static_cast` fails and surfaces as
`std::bad_cast`.

---

## 2. The Type Caster API

### 2.1 Required structure (nanobind ≥ 2.0)

```cpp
namespace nanobind::detail {

template <> struct type_caster<MyType> {
    NB_TYPE_CASTER(MyType, const_name("MyType"))

    // Python → C++
    bool from_python(handle src,
                     uint8_t flags,
                     cleanup_list* cleanup) noexcept {
        // On failure: PyErr_Clear() and return false
        // On severe error: PyErr_WarnFormat() and return false
        // Never throw
    }

    // C++ → Python
    static handle from_cpp(const MyType& src,
                            rv_policy policy,
                            cleanup_list* cleanup) noexcept {
        // On failure: PyErr_SetString(...) and return handle()
        // Never throw
    }
};

} // namespace nanobind::detail
```

### 2.2 Key rules enforced by the porting guide

**`noexcept` is mandatory.** Both `from_python()` and `from_cpp()` must be
`noexcept`. A missing `noexcept` means any `std::bad_cast` thrown inside the
body propagates outward through nanobind's call dispatch, which does not
expect it and cannot recover from it.

**`from_python()` failure protocol:** Clear the Python error state with
`PyErr_Clear()` and return `false`. Do not throw. nanobind uses the `false`
return to move to the next overload candidate.

**`from_cpp()` failure protocol:** Set a Python error with
`PyErr_SetString()` or `PyErr_Format()` and return an invalid handle
(`return nb::handle()`). Do not throw. This path is more serious than a
`from_python()` failure because a successfully computed return value cannot
be converted.

**`None` is not supported through type casters.** `None`-valued arguments
are only supported by bindings and wrappers. Passing `None` to a parameter
backed by a type caster (such as `std::optional<T>` without
`nanobind/stl/optional.h`) triggers a cast failure that can manifest as
`bad_cast` depending on the error path.

**`cleanup_list*` may be `nullptr`.** The cleanup list is only populated
during function dispatch. When a caster is invoked via `nb::cast<T>()` with
implicit conversions disabled, `cleanup` will be `nullptr`. Check for this
case before using it.

---

## 3. How `std::bad_cast` Surfaces

### 3.1 At stub generation (`stubgen`)

This is the most common occurrence. `stubgen` imports the compiled extension
and introspects every registered type and function to emit `.pyi`
annotations. It does this by `dynamic_cast`-ing internal nanobind metadata
structs. If any of the following conditions hold, the cast fails:

- A type appears in a function signature that has no registered caster and
  is not a registered `nb::class_<>` type
- A platform-specific integer type (`__int128`, `unsigned __int128`) appears
  as a return or parameter type and cannot be mapped to a Python annotation
- `nb::sig()` and `nb::arg().doc()` are both present on the same `.def()`
  (they conflict — use one or the other)
- nanobind < 2.0 is installed but `nb::arg().doc()` is used (`.doc()` on
  `nb::arg()` requires nanobind ≥ 2.0)
- A custom type caster's `NB_TYPE_CASTER` macro specifies a type name that
  conflicts with an already-registered binding

### 3.2 At import time

Less common, but possible if a caster is invoked during module
initialisation (e.g. populating a module-level constant) and the caster
itself performs an unsafe cast internally.

### 3.3 At call time

Rare. Occurs if a custom `from_python()` or `from_cpp()` contains an
uncaught `dynamic_cast` on a pointer that does not actually point to the
expected type.

---

## 4. Isolation Methods

Apply these in order. Each step halves the problem space.

### Step 1 — Get the full traceback

The default `stubgen` invocation suppresses the traceback. Run it directly
with `-W error` to stop at the first failure:

```bash
cd build/cp314-cp314-linux_x86_64

# Option A: check if import itself throws
python -W error -c "
import traceback
try:
    import _core
    print('import OK')
except Exception:
    traceback.print_exc()
"

# Option B: run stubgen with full output
python -W error -m nanobind.stubgen \
    -i . -M py.typed -m _core -o _core.pyi 2>&1
```

The frame immediately **above** `bad_cast` in the traceback identifies the
binding or type that triggered it.

### Step 2 — Check the nanobind version

`nb::arg().doc()` requires nanobind ≥ 2.0. Confirm:

```bash
python -c "import nanobind; print(nanobind.__version__)"
```

If below 2.0, either upgrade:

```bash
uv add nanobind --upgrade
```

Or remove `.doc()` from all `nb::arg()` calls and replace with a single
`nb::sig()` on the `.def()`.

### Step 3 — Audit every non-primitive type in all signatures

Walk every `.def()`, `.def_prop_ro()`, and `.def_prop_rw()`. For each
parameter and return type, confirm it is one of:

- A C++ primitive (`bool`, `int`, `float`, `double`)
- A type for which the matching STL header is included
- A type registered with `nb::class_<>`
- `nb::object` or a nanobind wrapper type

Any other type — especially Qrack-specific typedefs — needs investigation:

```bash
# For a Qrack project: check what these resolve to on your platform
grep -E "typedef|using" /usr/include/qrack/qrack_types.hpp | \
    grep -E "bitCapInt|bitLenInt|real1_f|real1\b"
```

If `bitCapInt` resolves to `unsigned __int128` (enabled by `ENABLE_UINT128`
on 64-bit Linux), stubgen cannot emit an annotation for it.
Add a `nb::sig()` override for every affected method:

```cpp
.def("m_reg",
    [](QrackSim& s, bitLenInt start, bitLenInt length) -> bitCapInt {
        return s.sim->MReg(start, length);
    },
    nb::arg("start"), nb::arg("length"),
    nb::sig("def m_reg(self, start: int, length: int) -> int"),
    "Measure a register, returning the result as a classical integer.")
```

### Step 4 — Binary search by commenting out `.def()` blocks

If steps 1–3 don't pinpoint it, comment out half the `.def()` calls in the
offending file, rebuild, and run stubgen. Repeat on whichever half still
throws. This converges on the problematic definition in 4–5 iterations
regardless of how many bindings there are.

```bash
# Quick rebuild without full reinstall
cmake --build build/cp314-cp314-linux_x86_64 -- -j$(nproc)
python -m nanobind.stubgen -i build/cp314-cp314-linux_x86_64 -m _core -o /tmp/test.pyi
```

### Step 5 — Check for `nb::sig()` / `nb::arg().doc()` conflicts

Having both on the same `.def()` produces conflicting internal metadata.
Use **one or the other** on any given definition:

```cpp
// ✗ conflict — don't combine these
.def("my_method", ...,
    nb::arg("x").doc("the x value"),
    nb::sig("def my_method(self, x: int) -> bool"))

// ✓ use nb::sig() alone when you need full signature control
.def("my_method", ...,
    nb::sig("def my_method(self, x: int) -> bool"),
    "Method docstring.")

// ✓ use nb::arg().doc() alone when you want per-arg docs without sig override
.def("my_method", ...,
    nb::arg("x") = 0
        .doc("the x value"),
    "Method docstring.")
```

### Step 6 — Verify custom type caster `noexcept` annotations

Any `type_caster` specialisation you have written must have `noexcept` on
both `from_python()` and `from_cpp()`. Without it, any exception thrown
inside propagates past nanobind's dispatch machinery, which has no handler
for it:

```cpp
// ✗ missing noexcept — bad_cast escapes
bool from_python(handle src, uint8_t flags, cleanup_list* cl) {
    // ...
}

// ✓ correct
bool from_python(handle src, uint8_t flags, cleanup_list* cl) noexcept {
    // ...
}
```

### Step 7 — Check for `None` passed through a type caster

If a Python caller passes `None` to a parameter that uses a type caster
(rather than a binding), the caster receives a `None` handle it was not
designed for. The most common form: `std::optional<T>` without the optional
header.

```cpp
// ✗ std::optional used without the caster header
// --> #include <nanobind/stl/optional.h> is missing

// ✓ include the header so None maps to std::nullopt
#include <nanobind/stl/optional.h>

.def("set_noise",
    [](QrackSim& s, std::optional<real1_f> noise) { ... },
    nb::arg("noise").none() = nb::none())
```

---

## 5. Qrack-Specific Type Table

The following types appear throughout `qinterface.hpp` and need explicit
handling in a nanobind binding layer.

| C++ type | Resolves to (typical 64-bit Linux) | nanobind strategy |
|---|---|---|
| `bitLenInt` | `uint16_t` | Passes as `int` via arithmetic caster — no action needed |
| `real1_f` | `float` | Passes as `float` — no action needed |
| `real1` | `float` or `double` | Passes as `float` — no action needed |
| `bitCapInt` | `uint64_t` OR `unsigned __int128` | **If `__int128`: add `nb::sig()` to all affected methods** |
| `Qrack::complex` | `std::complex<float>` | Include `<nanobind/stl/complex.h>` |
| `QInterfacePtr` | `std::shared_ptr<QInterface>` | Include `<nanobind/stl/shared_ptr.h>`; do not expose directly |
| `std::vector<bitLenInt>` | Vector of `uint16_t` | Include `<nanobind/stl/vector.h>` |
| `std::vector<complex>` | Vector of `std::complex<float>` | Include both `stl/vector.h` and `stl/complex.h` |
| `Qrack::Pauli` | `enum` from `common/pauli.hpp` | Bind with `nb::enum_<Qrack::Pauli>` in `pauli.cpp` |

### Checking `bitCapInt` at build time

Add this to `CMakeLists.txt` to detect and log the type before it causes a
silent stubgen failure:

```cmake
include(CheckTypeSize)
check_type_size("unsigned __int128" UINT128_SIZE LANGUAGE CXX)
if(HAVE_UINT128_SIZE)
    message(STATUS "bitCapInt is __int128 — nb::sig() overrides required on "
                   "all methods returning or accepting bitCapInt")
endif()
```

---

## 6. Quick Reference

### When `bad_cast` throws during stubgen

```
ImportError / std::bad_cast during nanobind.stubgen
    │
    ├─ Is nanobind < 2.0?
    │   └─ Yes → remove nb::arg().doc() or upgrade nanobind
    │
    ├─ Does the traceback frame name a specific .def()?
    │   └─ Yes → check return type and all parameter types of that definition
    │
    ├─ Is bitCapInt = unsigned __int128 on this platform?
    │   └─ Yes → add nb::sig() overrides to all affected methods
    │
    ├─ Is nb::sig() combined with nb::arg().doc() on the same .def()?
    │   └─ Yes → remove one; they conflict
    │
    └─ None of the above → binary search: comment out half the .def() blocks,
       rebuild, rerun stubgen; repeat until the offending definition is found
```

### Caster checklist for a new type

```
□ Header included in binding_core.h (not in individual .cpp files)
□ from_python() is noexcept
□ from_cpp() is noexcept
□ from_python() returns false on failure (does not throw)
□ from_cpp() calls PyErr_SetString() and returns handle() on failure
□ cleanup_list* nullptr case handled if cleanup is used
□ None-argument path uses nb::arg().none() if the type is std::optional<T>
□ NB_TYPE_CASTER name does not conflict with an nb::class_<> registration
□ nb::sig() added to any method whose return or param type cannot be
  annotated by stubgen (e.g. __int128-based typedefs)
```

---

## 7. Further Reading

- [nanobind porting guide — Type Casters](https://nanobind.readthedocs.io/en/latest/porting.html#type-casters)
- [nanobind information exchange — Type casters vs bindings vs wrappers](https://nanobind.readthedocs.io/en/latest/exchanging.html)
- [nanobind typing guide — nb::sig and stub generation](https://nanobind.readthedocs.io/en/latest/typing.html)
- [nanobind STL pair caster — useful reference implementation](https://github.com/wjakob/nanobind/blob/master/include/nanobind/stl/pair.h)
