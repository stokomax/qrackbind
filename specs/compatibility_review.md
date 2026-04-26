---
tags:
  - qrack
  - compatibility
  - pennylane
  - qiskit
  - bloqade
  - qrackbind
  - review
---
## qrackbind Compatibility Review — April 2026

Review of remaining and newly discovered compatibility issues against PennyLane, Bloqade, Qiskit, and the broader pyqrack ecosystem. Based on the current plan notes cross-referenced against live upstream state as of April 2026.

---

## Summary of Status

| Consumer | Plan status | New findings | Risk |
|---|---|---|---|
| Bloqade (`bloqade-pyqrack`) | Mostly correct | Dynamic qubit allocation gap; `cloneSid` deferred but listed as preserved | Medium |
| `qiskit-qrack-provider` | **Not in plan at all** | Existing PyPI package already wraps pyqrack for Qiskit | High |
| PennyLane | Architecturally correct | PennyLane API evolving; TOML capabilities file requirement needs verification | Low–Medium |
| Qook (Rust bindings) | Not in plan | New Rust binding mirrors pyqrack API exactly — a parallel ecosystem concern | Low |
| pyqrack itself | Plan written against v1.x | Now at v1.87.5 (April 2026), actively releasing — `multiplex1_mtrx` and other methods not in plan | Medium |

---

## Issue 1 — `cloneSid` Is Listed as Preserved But Is Deferred

**Where:** [[QuEra Bloqade — pyqrack Dependency Analysis]] table marks `cloneSid` as `✅ Preserved`. [[qrackbind Phase 1 Revised]] says it is "deferred — requires a lookup table of live simulator IDs."

**The conflict:** These two notes contradict each other. Bloqade's `StackMemory` and `DynamicMemorySimulator` use `cloneSid` indirectly when creating per-qubit sub-simulators. If `cloneSid` is not implemented, Bloqade's memory management model will fail even after the import-line migration.

**Fix needed:** The Bloqade dependency analysis table needs to be corrected — `cloneSid` should be marked `⚠️ Deferred`. The plan needs to add it to Phase 1 or explicitly defer it to a named later phase with a note that Bloqade's `StackMemory` mode will not work until it is implemented.

---

## Issue 2 — Dynamic Qubit Allocation Not in Plan

**What Bloqade now does:** Live Bloqade docs show `DynamicMemorySimulator` — a mode where qubits are allocated and released at runtime rather than pre-declared. The documentation explicitly notes this is incompatible with `isTensorNetwork=True`. This maps to `QInterface::Allocate()` and `QInterface::Dispose()`.

**What the plan covers:** The plan exposes a fixed `qubitCount` constructor and no `allocate`/`release` methods. The `QInterface` header shows `Allocate(bitLenInt start, bitLenInt length)` and `Dispose(bitLenInt start, bitLenInt length)` as pure virtual — they exist and are bindable.

**Risk:** Bloqade users relying on `DynamicMemorySimulator` will get a hard failure even with the deprecated alias layer in place. This is a new gap not present in the original analysis.

**Fix needed:** Add `allocate(start, length)` and `dispose(start, length)` bindings to the Phase 1 checklist, or explicitly document that `DynamicMemorySimulator` is not supported until a named later phase.

---

## Issue 3 — `qiskit-qrack-provider` Is an Existing Package Not In the Plan

**What exists:** A PyPI package `qiskit-qrack-provider` already wraps pyqrack as a Qiskit backend. The Qrack documentation explicitly references it: "Install the `qiskit-qrack-provider` Python package, which will also install PyQrack as a dependency." This is a production integration used by Qiskit users today.

**What the plan says:** The Framework Plugin Architecture note treats the Qiskit backend as a future Phase 8 deliverable to be built from scratch. It does not account for the existing package.

**The gap:** If `qrackbind` ships without the existing `qiskit-qrack-provider` being updated to use it, Qiskit users will have two divergent paths — the old `qiskit-qrack-provider` (pyqrack) and whatever new plugin qrackbind eventually ships. This fragments the ecosystem rather than unifying it.

**Fix needed:** Contact or open an issue with the `qiskit-qrack-provider` maintainers early, similar to the planned Bloqade PR. The PR scope is the same — swap `from pyqrack import` → `from qrackbind import`, update the `m()` call. The Framework Plugin Architecture note should be updated to acknowledge this package exists.

---

## Issue 4 — `multiplex1_mtrx` and Other pyqrack Methods Not in Plan

**What pyqrack has:** The live pyqrack source (v1.87.5) exposes `multiplex1_mtrx()` — a uniformly-controlled single-qubit gate taking a flat list of 4×2^n complex values. The `QInterface` header confirms the underlying method is `UniformlyControlledSingleBit`. The pyqrack source also surfaces `mcmul`, `mcdiv`, `pown`, `mul`, `div` — arithmetic gates that are runtime-guarded (raise `RuntimeError` if `isTensorNetwork=True`).

**What the plan covers:** The Phase 1 checklist covers `mtrx` and `mcmtrx` but not `multiplex1_mtrx`. The arithmetic gate set (`mul`, `div`, `pown`, `mcmul`, `mcdiv`) is absent from all plan notes.

**Risk for Bloqade:** Bloqade's QASM2 interpreter may invoke `multiplex1_mtrx` for uniformly-controlled gates that appear in compiled circuits. If it does, those circuits will fail at runtime.

**Fix needed:** Add `multiplex1_mtrx` to the Phase 1 gate checklist. Move arithmetic gates (`mul`, `div`, `pown`) to an explicit later phase with a note about the `isTensorNetwork` runtime guard.

---

## Issue 5 — PennyLane TOML Capabilities File Not Documented

**What PennyLane requires:** The Framework Plugin Architecture note correctly identifies that `preprocess_transforms()` uses a TOML capabilities file to tell PennyLane which gates are natively supported. PennyLane's preprocessor decomposes unsupported gates before `execute()` is called.

**What the plan is missing:** No note documents what this TOML file looks like, what gate names PennyLane uses, or which qrackbind gates map to which PennyLane operation names. PennyLane uses its own gate name conventions (`RZ`, `CNOT`, `Hadamard`, `PauliX`, etc.) which differ from both pyqrack's names and Qiskit's names.

**Risk:** The dispatch table in `_dispatch.py` (shown in the framework note) has a name-mapping skeleton but no verified complete mapping. A missing or wrong gate name causes PennyLane's preprocessor to decompose that gate into primitives — which may or may not be efficient.

**Fix needed:** Create a dedicated note (or section in the Framework Architecture note) that maps PennyLane operation names → qrackbind method names and documents the TOML capabilities file format. This can be done before Phase 8 without writing any code.

---

## Issue 6 — Qook (Rust Bindings) Is a New Parallel Ecosystem

**What exists:** `Qook` is a new Rust binding for Qrack (announced by the Qrack team) that mirrors the PyQrack API exactly. Its documentation explicitly states "The API is meant to exactly mirror (Python-based) PyQrack." This is a signal that the pyqrack API surface is now considered stable enough to mirror in other languages.

**Relevance to qrackbind:** Qook means the pyqrack API is becoming a *de facto* standard interface for Qrack. Any deviation qrackbind makes from the pyqrack surface (renaming `m()` to `measure()`, changing `get_state_vector()` to a property) will create divergence from both pyqrack and Qook simultaneously. The deprecated alias layer mitigates this for Python but it is worth noting.

**No immediate action required** — but the existence of Qook strengthens the argument for keeping the compat alias layer in place indefinitely rather than removing it in a future major version.

---

## Issue 7 — pyqrack at v1.87.5: Version Drift Risk

**Current state:** pyqrack is at version 1.87.5 as of March 2026, with releases several times per month through late 2025. The plan notes were written against an earlier API snapshot.

**What this means:** The plan correctly captures the core stable gate API (`h`, `x`, `mcx`, `Pauli`, etc.) which hasn't changed. But higher-level features (`pyzxCircuit` constructor arg, `qiskitCircuit` constructor arg, newer arithmetic methods) may have evolved in ways the notes don't reflect.

**Fix needed:** Before writing the constructor binding, fetch the current pyqrack constructor signature from the live source to verify all kwargs are still accurate. The `pyzxCircuit` and `qiskitCircuit` constructor args in particular are not currently in the Phase 1 binding plan and are worth either including or explicitly excluding with a note.

---

## Confirmed — Still Correct

These plan elements have been verified against the live upstream state and require no changes:

- **Gate method names** (`h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `mcx`, `macx`, etc.) — unchanged in pyqrack, still correct
- **Constructor kwargs** (`isTensorNetwork`, `isSchmidtDecompose`, `isStabilizerHybrid`, etc.) — unchanged
- **`Pauli` enum** (`PauliX`, `PauliY`, `PauliZ`, `PauliI`) — unchanged
- **Deprecated alias approach** (`m()`, `m_all()`, `get_state_vector()`, `get_num_qubits()`) — still the right strategy
- **Bloqade PR scope** (~5–10 lines across 3–4 files) — still accurate for the non-dynamic-allocation use case
- **PennyLane `execute()` + TOML architecture** — still valid for PL 0.36+
- **`MCInvert`/`MACInvert`/`MCPhase`/`MACPhase`/`MCMtrx` as the multi-control primitives** — confirmed by qinterface.hpp

---

## Priority Action Items

1. **Fix the `cloneSid` contradiction** — correct the Bloqade dependency table to `⚠️ Deferred`
2. **Add `allocate`/`dispose` to plan** — either Phase 1 or a named deferred phase; document the `DynamicMemorySimulator` gap explicitly
3. **Add `multiplex1_mtrx` to Phase 1 checklist** — it's bindable via `UniformlyControlledSingleBit`
4. **Open issue with `qiskit-qrack-provider`** — don't build a parallel Qiskit plugin without coordinating with the existing one
5. **Document PennyLane gate name mapping** — TOML file content and gate name table before starting Phase 8
6. **Verify constructor kwargs against live pyqrack source** — specifically `pyzxCircuit` and `qiskitCircuit` args

---

## Related

- [[pyqrack Compatibility Strategy]]
- [[QuEra Bloqade — pyqrack Dependency Analysis]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrackbind Phase 1 Revised]]
- [[qrackbind Project Phase Breakdown]]
- [[QrackSimulator API Method Categories]]
