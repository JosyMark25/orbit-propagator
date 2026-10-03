# orbit-propagator

A two-body orbital mechanics propagator, built from scratch.

Given six classical orbital elements and a time array, it answers
**where the satellite is and how fast it moves** — with every number
verifiable against hand calculation and textbook references.

> Started as a self-directed learning project (V1, analytic Kepler
> propagation). Roadmap: numerical integration (V2) → symplectic
> integrators (V3).

## Physics & Method (V1)

Classical orbital elements → perifocal PQW frame → ECI, the full
analytic chain:

```
(a, e, i, Ω, ω, ν₀) → M₀ → M(t) = M₀ + n·t
→ Kepler's equation solved by Newton iteration (E from M)
→ true anomaly ν → r, v in PQW → rotation to ECI
```

~137 lines of hand-written Python, no orbital-mechanics libraries.

## Validation Anchors

Four anchor numbers, each independently verified:

| Anchor | Value | Measured |
|---|---|---|
| Perigee direction R·(1,0,0) for i=30°, Ω=40°, ω=60° | (-0.099, 0.896, 0.433) | exact to 1e-5 |
| Kepler solution for M=0.5, e=0.3 | E = 0.6913 | back-substitution residual = 0 |
| Period, a = 8000 km | T = 7121.08 s | formula-verified |
| Energy conservation over one full orbit | drift < 1e-12 | **8.6e-16 (machine precision)** |

Angular momentum: conserved to 3.9e-16. Periodic closure: 8e-12 km.

## Quick Start

```bash
pip install numpy matplotlib pytest
python3 -m pytest tests/ -v        # 193 passed, 0 xfail
python3 main.py                    # interactive six-elements input
```

## Test Suite (193 tests)

Designed to survive the V2/V3 rewrites — tests assert physical
properties (conservation laws, orthogonality, vis-viva, round-trip
identities), not implementation details:

- `test_anchors.py` — acceptance anchors, frozen numbers
- `test_kepler.py` — 54-case residual grid, multi-rev M, anomaly round-trips
- `test_elements.py` — rotation matrix (orthogonality, det, Ω/ω swap guard), PQW physics
- `test_propagate.py` — energy/momentum conservation (black-box), geometry, semantics
- `test_end_to_end.py` — clean-subprocess full pipeline
- `test_known_bugs.py` — **regression tests for V1.1 fixes** (formerly the
  strict-xfail known-bug registry; all three bugs fixed in V1.1 and promoted)

## Design Notes (V1.1)

- `propagate_from_coe` takes `mu` as a **required argument** — the physics
  core has no Earth default baked in. "This is an Earth simulator" lives in
  the caller (`main.py`), so non-Earth central bodies (V2: the Moon) plug in
  without touching the core.
- `solve_kepler` **raises** on non-convergence (loud failure, no silent
  garbage) and rejects `maxsteps < 1` with a clear `ValueError`.

## Project Structure

```
constants.py    # μ_earth, R_earth
kepler.py       # solve_kepler (Newton), E↔ν conversions
elements.py     # rotation_pqw_to_eci, coe_to_pqw
propagate.py    # propagate_from_coe: elements + time → (r, v) arrays
visualize.py    # 3D orbit plot
main.py         # CLI entry
tests/          # pytest suite (192)
docs/           # Chinese docs: spec, acceptance checklist, V1 audit report
```

## Roadmap

- **V1.1** — clear P0/P1: save_path, mu penetration, convergence guard
- **V2** — numerical integration (RK4) behind `method=` parameter;
  conservation tests re-used as-is at numerical tolerances
- **V3** — symplectic integrators for long-term stability

## License

MIT

## Notes

Developed with AI-assisted tutoring; every line hand-written and
independently validated (audit report in `docs/`). V1 completed
2026-10-01 as a personal milestone.
