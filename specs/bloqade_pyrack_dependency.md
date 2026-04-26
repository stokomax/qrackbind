---
tags:
  - qrack
  - quera
  - bloqade
  - pyqrack
  - compatibility
  - qrackbind
---
## QuEra Bloqade — pyqrack Dependency Analysis

QuEra's Bloqade SDK is a concrete, production-grade consumer of the pyqrack API. Understanding exactly how it uses `QrackSimulator` defines what must be preserved and what Bloqade will need to change to adopt `qrackbind`.

---

## What Bloqade Is

Bloqade is QuEra's open-source Python SDK for neutral-atom quantum computers — a namespace package of eDSLs built on the Kirin compiler infrastructure. Its digital circuit stack uses pyqrack as a simulation backend.

Key packages:
- `bloqade` — top-level namespace
- `bloqade-circuit` — circuit eDSLs (SQUIN, QASM2, etc.)
- `bloqade-pyqrack` — dedicated pyqrack backend (separate PyPI package)

---

## How Bloqade Depends on pyqrack

### Direct Object References
Bloqade stores live `QrackSimulator` objects inside its own qubit type:
```python
PyQrackQubit(
    addr=0,
    sim_reg=<pyqrack.qrack_simulator.QrackSimulator object at 0x7fd2c827a510>,
    state=<QubitState.Active: 1>
)
```
The concrete class — not an abstract interface — is stored and referenced directly.

### Constructor Option Forwarding
Bloqade's `PyQrackOptions` dataclass forwards kwargs directly to `QrackSimulator.__init__`:
```python
pyqrack_options: PyQrackOptions = field(default_factory=_default_pyqrack_args)
# qubitCount is overwritten; all other options are forwarded as-is
```

### Gate Dispatch
Bloqade's `PyQrackInterpreter` walks compiled Kirin IR and calls gate methods by name: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `mcx`, `mcy`, `mcz`, and `m()` for measurement.

### State Extraction
Bloqade calls `get_state_vector()` (or similar) via qubit references for analysis purposes (reduced density matrix, eigensystem output).

### Roadmap Commitment
Bloqade's stated 2025 roadmap explicitly names "QASM2 dialect with pyqrack backend" — this is a first-class named integration, not incidental.

---

## Migration Analysis — What Changes for Bloqade

Under the **light-effort migration** model (see [[pyqrack Compatibility Strategy]]), `qrackbind` is its own package and Bloqade needs a small PR to `bloqade-pyqrack`.

### What changes (small)

| File | Change |
|---|---|
| `pyproject.toml` | `"pyqrack"` → `"qrackbind"` in dependencies |
| `base.py` / `target.py` / `reg.py` | `from pyqrack import ...` → `from qrackbind import ...` |
| `interpreter.py` (measurement) | `sim.m(q)` → `sim.measure(q)` (if not keeping alias) |

### What does NOT change for Bloqade

- All gate method calls: `h`, `x`, `mcx`, `rx`, etc. — **preserved**
- All constructor kwargs: `isTensorNetwork`, `isStabilizerHybrid`, etc. — **preserved**  
- `Pauli` enum values — **preserved**
- The `QrackSimulator` class name — **preserved**
- All qubit indexing and register logic — **preserved**

**Estimated PR size: 5–10 lines across 3–4 files.**

---

## The Transition Path

1. `qrackbind` reaches stability (Phase 1 complete)
2. Open a PR to `bloqade-pyqrack` — the diff is tiny, easy to review
3. `bloqade-pyqrack` releases a new version depending on `qrackbind`
4. Bloqade users `pip install bloqade[pyqrack]` and transparently get the nanobind backend

The PR is worth opening proactively — Bloqade is actively developed and the team is receptive to contributions.

---

## API Surface `qrackbind` Must Provide for Bloqade

| Category | Requirement | Status in qrackbind |
|---|---|---|
| Class name | `QrackSimulator` | ✅ Preserved |
| Constructor kwargs | All boolean flags, camelCase | ✅ Preserved |
| Gate methods | `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`, `rx`, `ry`, `rz`, `r1`, `u`, `u2`, `u3`, `mcx`, `mcy`, `mcz`, MC/MAC variants | ✅ Preserved |
| Measurement | `m(qubit)` | ⚠️ Renamed to `measure()` — deprecated alias available |
| State access | `get_state_vector()` | ⚠️ Renamed to `state_vector` property — deprecated alias available |
| `Pauli` enum | `from qrackbind import Pauli` | ✅ Preserved (new import path) |
| `cloneSid` constructor param | Works | ⚠️ Deferred — requires a live simulator ID lookup table; not in Phase 1. Bloqade's `StackMemory` and `DynamicMemorySimulator` modes depend on this indirectly. |

The ⚠️ items are the only ones requiring a code change in Bloqade's source.

---

## Related

- [[pyqrack Compatibility Strategy]]
- [[qrackbind Python API Design]]
- [[Framework Plugin Architecture (PennyLane + Qiskit)]]
- [[qrackbind Project Phase Breakdown]]
