# Model A — φ⁴ Scalar Field Theory with Higher-Order Kinetic Term

```
ooo        ooooo   .oooooo.   oooooooooo.   oooooooooooo ooooo                   .o.
`88.       .888'  d8P'  `Y8b  `888'   `Y8b  `888'     `8 `888'                  .888.
 888b     d'888  888      888  888      888  888          888                  .8"888.
 8 Y88. .P  888  888      888  888      888  888oooo8     888                 .8' `888.
 8  `888'   888  888      888  888      888  888    "     888                .88ooo8888.
 8    Y     888  `88b    d88'  888     d88'  888       o  888       o       .8'     `888.
o8o        o888o  `Y8bood8P'  o888bood8P'   o888ooooood8 o888ooooood8      o88o     o8888o
```

## Overview

This code implements **Model A dynamics** (overdamped relaxational dynamics) for a
three-dimensional φ⁴ scalar field theory with an additional higher-order kinetic term.
The free-energy functional is

```
H[φ] = ∫ d³x [ Z/2 (∇φ)²  +  1/2 (∇²φ)²  +  m²/2 φ²  +  λ/4 φ⁴ ]
```

with **λ = 4**, **Γ = 1** (diffusion rate), **T = 1** (temperature).

Time evolution is given by the Langevin equation

```
∂φ/∂t = −Γ δH/δφ + η(x,t),   ⟨η(x,t) η(x′,t′)⟩ = 2Γ T δ³(x−x′) δ(t−t′)
```

which is integrated with a Metropolis-style accept/reject step using a Gaussian trial
move of width `Rate = √(2 Δt Γ)`.

---

## Repository Structure

```
modelA_LP/
├── src/
│   ├── modelA.jl        # Module entry point (imports, ASCII header)
│   ├── initialize.jl    # Command-line argument parsing and global constants
│   └── simulation.jl    # Core update engine (sublattice decomposition, ΔH, sweeps)
├── scripts/
│   ├── thermalize.jl    # Thermalises a field configuration and saves it to disk
│   ├── measure.jl       # Measures observables over a range of mass values
│   ├── measure_single.jl# Measures observables at a single mass value
│   ├── snap.jl          # Generates an ensemble of trajectory snapshots
│   ├── bootstrap.jl     # Bootstrap statistical analysis utilities
│   ├── measure.sh       # Bash wrapper for measure.jl
│   ├── therm.sh         # Bash wrapper for thermalize.jl
│   ├── submit_therm.sh  # LSF batch submission for thermalization (GPU)
│   ├── submit_snap.sh   # LSF batch submission for snapshot generation (GPU)
│   ├── submit_measure.sh# LSF batch submission for measurements (GPU)
│   ├── submit_reweight.sh# LSF batch submission for reweighting (GPU)
│   ├── run_cpu.sh       # LSF job template — CPU (16 threads)
│   ├── run_h100.sh      # LSF job template — H100 GPU
│   ├── run_l40s.sh      # LSF job template — L40S GPU
│   └── watch.sh         # Progress monitor for measurement jobs
├── data/                # Output directory for all simulation data
├── Project.toml         # Julia project dependencies
└── Manifest.toml        # Exact dependency versions
```

---

## Implementation Details

### Lattice and Boundary Conditions

The simulation uses a three-dimensional cubic lattice of side length **L** with
**periodic (toroidal) boundary conditions**.  The helper functions

```julia
NNp(n) = n % L + 1          # forward neighbour (1-indexed, wraps at L)
NNm(n) = (n + L - 2) % L + 1  # backward neighbour (1-indexed, wraps at L)
```

implement the toroidal arithmetic throughout.

### Energy Change ΔH

For a trial move `φ(x) → φ(x) + q` the energy difference reads

```
ΔH = ΔH_kin + ΔH_lap + ΔH_mass + ΔH_int
```

where

| Term | Expression | Source |
|------|------------|--------|
| `ΔH_kin`  | `Z (3Δϕ² − Δϕ ∑nn)`  | Conventional kinetic `Z/2 (∇φ)²` |
| `ΔH_lap`  | `Δϕ (21Δϕ + 42φ_old − 12∑nn + ∑nnn_axial + 2∑nnn_diag)` | Higher-order `1/2 (∇²φ)²` |
| `ΔH_mass` | `m²/2 Δϕ²` | Mass term |
| `ΔH_int`  | `λ/4 (φ_new⁴ − φ_old⁴)` | φ⁴ coupling |

with `Δϕ = q` (trial displacement) and `Δϕ²` denoting `φ_new² − φ_old²`
(difference of squared values, **not** `(Δϕ)²`).  The sums run over:

- **∑nn** — 6 nearest neighbours (±1 along each axis)
- **∑nnn_axial** — 6 axial next-nearest neighbours (±2 along each axis)
- **∑nnn_diag** — 12 diagonal next-nearest neighbours (±1 along two different axes)

Because `ΔH` reads sites within L1-distance ≤ 2, same-sublattice sites must be
separated by at least 3 lattice steps to allow race-free parallel updates.

### Sublattice (Checkerboard) Decomposition

To enable safe parallel execution the lattice is coloured with a **product-p coloring**:

```
colour(x₁, x₂, x₃) = (x₁−1)%p + p·((x₂−1)%p) + p²·((x₃−1)%p)
```

The coloring period `p_sub` is defined as the **smallest divisor of L that is ≥ 3 and
satisfies L ≥ 2·p** (ensuring that the minimum toroidal L1-distance between same-colour
sites is p ≥ 3).  There are `N_sub = p_sub³` sublattices per dissipative step.

```julia
const p_sub = let
    p = 3
    while p <= L
        if L % p == 0 && L >= 2*p
            break
        end
        p += 1
    end
    p > L && error("L=$L has no valid sublattice period ...")
    p
end
const N_sub = p_sub^3
```

One full **dissipative step** sweeps all `N_sub` sublattices in sequence; all sites
within a single sublattice are updated in parallel (CPU: `Threads.@threads`, GPU: CUDA
kernel).

### Monte Carlo Update Step

```julia
function step(ϕ, m², x1, x2, x3)
    q    = Rate * randn()                # Gaussian trial displacement
    δH   = ΔH(ϕ, m², (x1,x2,x3), q)
    ϕ[x1,x2,x3] += q * (rand() < exp(-δH/T))  # Metropolis accept/reject
end
```

### Parallelisation

| Backend | Mechanism | Selection |
|---------|-----------|-----------|
| CPU | `Threads.@threads` over sublattice sites | `--cpu` flag |
| GPU | CUDA kernel with auto-tuned blocks/threads | default |

The GPU kernel configuration is determined at startup via `launch_configuration`.

---

## Parameters

All parameters are set via command-line arguments:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `size` (positional) | `Int` | **required** | Lattice side length L |
| `--mass` | `Float64` | `-2.28587` | Actual mass parameter m² (used directly) |
| `--Z` | `Float64` | `1.0` | Coefficient Z of the conventional kinetic term |
| `--dt` | `Float64` | `0.04` | Langevin time step Δt |
| `--rng` | `Int` | `0` | Random seed (0 = unseeded) |
| `--fp64` | flag | off | Use Float64 instead of Float32 |
| `--cpu` | flag | off | Use CPU threading instead of GPU |
| `--init` | `String` | — | Path to `.jld2` file for initial field configuration (default: Gaussian hotstart) |

**Fixed physical constants:**

| Symbol | Value | Description |
|--------|-------|-------------|
| λ | 4.0 | φ⁴ self-coupling |
| Γ | 1.0 | Diffusion rate |
| T | 1.0 | Temperature |

---

## Workflow

### 1. Thermalization

```bash
julia --threads auto scripts/thermalize.jl <L> [options]
```

Runs `L` outer iterations, each performing `L²` dissipative steps, and saves the
field to disk after each outer iteration in the `data/` directory as
`thermalized_L_<L>_id_<seed>.jld2`.

### 2. Measurement

```bash
julia --threads auto scripts/measure_single.jl <L> --init <state.jld2> [options]
```

Evolves the field for `100·L²` dissipative steps, sampling every `L²/8` steps.
Writes output files to the `data/` directory with names
`magnetization_L_<L>_mass_<m²>_id_<seed>.dat`, each containing three columns:

```
<step>   <M>   <Σφ²>
```

where `M = Σφᵢ / L²` is the observable computed by the code (note: this is the
sum over all L³ sites divided by L², yielding units of field × L) and `Σφ²` is
the sum of squared field values.

`measure.jl` additionally scans mass values from m² = −3.5 down to m² = −4.0 in
steps of 0.01 (i.e. `for m²0 in -3.5:-0.01:-4.0`).

### 3. Snapshot Generation

```bash
julia --threads auto scripts/snap.jl <L> --init <state.jld2> [options]
```

Saves 2500 independent configurations (separated by `L²` dissipative steps each)
to the `data/` directory as `snapshot_L_<L>_seed_<seed>_id_<idx>.jld2`.

### 4. Statistical Analysis

`scripts/bootstrap.jl` provides `average`, `variance`, and `bootstrap` functions for
computing mean and uncertainty estimates from the measurement files.

---

## Valid Lattice Sizes

The sublattice decomposition requires L to have a divisor p ≥ 3 with L ≥ 2p.
The **smallest valid L is 6**.  All valid sizes up to 70 are listed below:

| L  | p_sub | N_sub  | | L  | p_sub | N_sub  | | L  | p_sub | N_sub  |
|----|-------|--------|-|----|-------|--------|-|----|-------|--------|
|  6 |   3   |     27 | | 28 |   4   |     64 | | 52 |   4   |     64 |
|  8 |   4   |     64 | | 30 |   3   |     27 | | 54 |   3   |     27 |
|  9 |   3   |     27 | | 32 |   4   |     64 | | 55 |   5   |    125 |
| 10 |   5   |    125 | | 33 |   3   |     27 | | 56 |   4   |     64 |
| 12 |   3   |     27 | | 34 |  17   |   4913 | | 57 |   3   |     27 |
| 14 |   7   |    343 | | 35 |   5   |    125 | | 58 |  29   |  24389 |
| 15 |   3   |     27 | | 36 |   3   |     27 | | 60 |   3   |     27 |
| 16 |   4   |     64 | | 38 |  19   |   6859 | | 62 |  31   |  29791 |
| 18 |   3   |     27 | | 39 |   3   |     27 | | 63 |   3   |     27 |
| 20 |   4   |     64 | | 40 |   4   |     64 | | 64 |   4   |     64 |
| 21 |   3   |     27 | | 42 |   3   |     27 | | 65 |   5   |    125 |
| 22 |  11   |   1331 | | 44 |   4   |     64 | | 66 |   3   |     27 |
| 24 |   3   |     27 | | 45 |   3   |     27 | | 68 |   4   |     64 |
| 25 |   5   |    125 | | 46 |  23   |  12167 | | 69 |   3   |     27 |
| 26 |  13   |   2197 | | 48 |   3   |     27 | | 70 |   5   |    125 |
| 27 |   3   |     27 | | 49 |   7   |    343 | |    |       |        |
|    |       |        | | 50 |   5   |    125 | |    |       |        |
|    |       |        | | 51 |   3   |     27 | |    |       |        |

**Invalid sizes** (no valid sublattice period exists):
2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67

The most efficient choices are **multiples of 3** (p_sub = 3, N_sub = 27) or
**multiples of 4 not divisible by 3** (p_sub = 4, N_sub = 64), as these minimise
the number of serial sublattice sweeps.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) | Command-line argument parsing |
| [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | GPU acceleration |
| [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) | Gaussian hotstart |
| [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) | Binary field snapshots |
| [CodecZlib.jl](https://github.com/JuliaIO/CodecZlib.jl) | Compressed JLD2 output |
| Printf | Formatted ASCII measurement output |

Install with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Example Usage

```bash
# Thermalise on GPU (default), L=24, default mass (m² = -2.28587)
julia --project=. scripts/thermalize.jl 24

# Thermalise on CPU with 8 threads, Float64, custom seed
julia --project=. --threads 8 scripts/thermalize.jl 24 --cpu --fp64 --rng 42

# Measure from a thermalised starting configuration
julia --project=. --threads auto scripts/measure_single.jl 24 \
    --init data/thermalized_L_24_id_42.jld2 --rng 42

# Measure at a specific mass value (m² = -2.38587)
julia --project=. scripts/measure_single.jl 24 --mass -2.38587 \
    --init data/thermalized_L_24_id_42.jld2
```
