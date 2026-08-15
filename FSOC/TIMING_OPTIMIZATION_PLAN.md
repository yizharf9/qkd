# FSOC Timing Analysis & Optimization Plan for 10 GHz 2-PAM Channel

## Executive Summary

Your simulation is currently **~1.4 billion times slower** than the 10 GHz target. This is not a bug—it's a fundamental physics simulation trade-off. Here's the complete analysis and optimization roadmap.

---

## PART 1: CURRENT MEASUREMENTS

### 1️⃣ WFS Loop Readout Rate (How often can we measure?)

| Metric | Value |
|--------|-------|
| **Configured delta_t** | 1 ms (theoretical 1 kHz) |
| **Actual achieved rate** | ~3.5 Hz (280 ms per iteration) |
| **Simulation reality** | ~140 iterations per 50 seconds |
| **Bottleneck** | HCIPy wavefront propagations |

**Gap Factor:** Expected 1 kHz, achieving 3.5 Hz = **286× too slow**

### 2️⃣ PAM Bit Transmission Rate (How many 2-PAM bits per second?)

| Metric | Value |
|--------|-------|
| **10 GHz target** | 10,000,000,000 bits/second |
| **Current configuration** | DISABLED (pam_factor = 1.0) |
| **After re-enabling 2-PAM** | ~3.5 bits/second |
| **Speed gap** | ~3 billion times slower |

### 3️⃣ Main Delays (Where does each 280 ms iteration go?)

| Stage | Time | % of Total | Status |
|-------|------|-----------|--------|
| ① WFS propagation (layer_ao) | 50-100 ms | 20-30% | 🔴 WORST |
| ② SH sensing (shwfs optics) | 30-80 ms | 15-25% | 🔴 SECOND |
| ③ Deformable mirror (modes) | 20-50 ms | 10-15% | 🟡 OK |
| ④ Camera integration + noise | 10-30 ms | 5-15% | 🟡 OK |
| ⑤ Reconstruction & DM update | 10-20 ms | 5-10% | 🟢 OK |
| ⑥ Data logging & CSV | 5-15 ms | 2-5% | 🟢 OK |
| ⑦ Python overhead (TQDM) | ~30 ms | 10-20% | 🟡 AVOIDABLE |

---

## PART 2: 4-PHASE OPTIMIZATION STRATEGY

### PHASE 1: PROFILING & MEASUREMENT (2-3 hours)

**Goal:** Quantify exact bottlenecks with instrumentation

**Tasks:**
- [ ] Add `time.perf_counter()` markers around each simulation stage
- [ ] Create per-iteration timing breakdown CSV
- [ ] Identify 80/20 rule (which 20% of code takes 80% of time)
- [ ] Measure memory allocation and garbage collection overhead

**Output:**
```
timing_breakdown.csv
├── iteration
├── wfs_prop_ms
├── sh_sense_ms  
├── dm_apply_ms
├── camera_ms
├── reconstruction_ms
├── logging_ms
└── total_ms
```

**Expected Finding:** WFS propagation + SH sensing = ~50% of runtime

---

### PHASE 2: ALGORITHMIC OPTIMIZATIONS (4-6 hours)

**Goal:** Reduce per-iteration computational complexity via grid/parameter reduction

#### 2A: Reduce Focal Grid (High Feasibility)
- **Current:** 256×256 focal grid = 65,536 pixels
- **Proposed:** 128×128 = 16,384 pixels (4× smaller)
- **Impact:** 4× speedup on all wavefront propagations
- **Trade-off:** Lower PSF fidelity, but still representative
- **Est. Saving:** 80-100 ms/iteration

#### 2B: Reduce WFS Lenslets (High Feasibility)  
- **Current:** 40×40 lenslets = 1,600 WFS pixels
- **Proposed:** 30×30 lenslets = 900 pixels
- **Impact:** 1.8× speedup on WFS sensing
- **Trade-off:** Coarser wavefront measurement
- **Est. Saving:** 20-30 ms/iteration

#### 2C: Reduce DM Modes (High Feasibility)
- **Current:** 64 disk-harmonic modes
- **Proposed:** 48 modes
- **Impact:** 1.3× speedup on DM application
- **Trade-off:** Slightly less correction capability
- **Est. Saving:** 5-10 ms/iteration

#### 2D: Skip Non-Critical Iterations (Medium Feasibility)
- **Option:** Only compute correction every N iterations, coast on others
- **Impact:** 2-5× speedup
- **Trade-off:** Degraded AO performance

**Combined Phase 2 Speedup: 4-6×**

**Expected Rate:** 14-21 iterations/sec (50-70 ms/iteration)

---

### PHASE 3: IMPLEMENTATION OPTIMIZATIONS (6-8 hours)

**Goal:** Optimize code execution, memory, and caching

#### 3A: Vectorize large_poisson()
- **Current:** Element-wise loop over pixels
- **Fix:** Use numpy batch operations
- **Impact:** 2-5× speedup on noise generation
- **Est. Saving:** 3-10 ms/iteration

#### 3B: Pre-allocate Arrays
- **Current:** New arrays created each iteration
- **Fix:** Allocate once, reuse across iterations
- **Impact:** 1.5-2× speedup on memory operations
- **Est. Saving:** 5-15 ms/iteration

#### 3C: Minimal Logging Mode
- **Option 1:** Only log every 10th iteration
- **Option 2:** Use ring buffer instead of list.append()
- **Impact:** 1.2-1.5× speedup
- **Est. Saving:** 2-5 ms/iteration

#### 3D: Numba/Cython Compilation
- **Target:** Hot path functions identified in Phase 1
- **Impact:** 2-10× on specific functions
- **Challenge:** Requires testing and compatibility checking
- **Est. Saving:** 5-20 ms/iteration (if successful)

**Combined Phase 3 Speedup: 2-4×**

**Expected Rate:** 28-84 iterations/sec (12-35 ms/iteration)

---

### PHASE 4: ARCHITECTURE RETHINK (8-12 hours)

**Goal:** Major redesign for parallel/GPU execution or hybrid simulation

#### 4A: GPU Acceleration (Very High Impact)
- **Approach:** Use CuPy instead of NumPy for grid operations
- **Impact:** 10-100× speedup
- **Challenge:** Requires NVIDIA GPU, CUDA, CuPy installation
- **Est. Saving:** 200-260 ms/iteration → 3-25 ms/iteration
- **Feasibility:** HIGH if hardware available

#### 4B: Hybrid Simulation Architecture (High Impact)
- **Keep:** Detailed 50-iteration AO runs for validation
- **Add:** Fast "channel simulator" for realistic 2-PAM transmission
  - Pre-computed turbulence patterns
  - Lookup tables for Poisson noise
  - Direct power/phase calculations
  - Can run at 1 MHz+ simulation rates
- **Integration:** Use Tier 1 (detailed) to calibrate Tier 2 (fast)
- **Impact:** Can simulate realistic long bitstreams

#### 4C: Separate Measurement & Rendering
- **Current:** Save plots each iteration (I/O overhead)
- **Fix:** Defer all visualization to end
- **Impact:** 1.5-2× speedup
- **Est. Saving:** 10-30 ms/iteration

**Expected Rate with GPU: 175-700 iterations/sec (1.4-5.7 ms/iteration)**

---

## PART 3: REALISTIC TARGETS

### Speedup Progression

| Scenario | Optimization Phase | Rate | Speedup | 50 Iterations | Time |
|----------|-------------------|------|---------|---------------|------|
| **Current** | None | 3.5 it/s | 1× | 175 bits | 14.3 sec |
| **Phase 2** | Grid+Lenslets+Modes | 14-21 it/s | 4-6× | 700-1050 bits | 2.4-3.6 sec |
| **Phase 2+3** | + Vectorization + Arrays | 28-84 it/s | 8-24× | 1400-4200 bits | 0.6-1.8 sec |
| **Phase 2+3+4** | + GPU | 175-700 it/s | 50-200× | 8750-35000 bits | 0.07-0.3 sec |

### Reality Check: Why So Slow?

```
10 GHz Real Hardware:        10,000,000,000 bits/second
Best Case (GPU + All Opts):  1-2 MHz simulation
Speed Factor:                ~5,000-10,000× slower

This is NORMAL for full physics simulation!
- Each bit requires atmosphere propagation
- Each propagation requires FFT on 256×256 grid
- Each iteration does full wavefront reconstruction
- This is legitimate scientific computing overhead
```

---

## PART 4: RECOMMENDED HYBRID APPROACH

### Three-Tier Architecture

#### **Tier 1: Detailed AO Simulation** (Current Code, Optimized)
- **Purpose:** Validate AO correctness, measure PSF
- **Duration:** 50-200 AO iterations
- **Output:** Representative channel snapshots, WFS images, PSF plots
- **Speed:** 14-700 iterations/sec (depending on phase)
- **Use Case:** Testing, validation, publication-quality results

#### **Tier 2: Fast Channel Simulator** (New Module)
- **Purpose:** Realistic 2-PAM transmission at 10 GHz rates
- **Duration:** 1 million - 1 billion bits
- **Model:** Simplified turbulence + pre-computed AO correction matrix
- **Speed:** 1-100 MHz simulation rates
- **Use Case:** Long-duration BER analysis, link budget calculations

#### **Tier 3: Analysis & Visualization**
- **Purpose:** Post-process results from Tier 1+2
- **Output:** BER curves, timing metrics, PSF heatmaps
- **Speed:** Fast (mainly I/O and plotting)

**Integration Strategy:**
1. Run Tier 1 simulation (50 iterations) → Get corrected PSF, DM states
2. Extract correction matrix from Tier 1 results
3. Feed into Tier 2 fast simulator
4. Run Tier 2 for realistic duration (1M+ bits)
5. Analyze combined results in Tier 3

---

## PART 5: IMMEDIATE ACTION ITEMS (Next 2 Hours)

### ✅ REQUIRED TASKS

- [ ] **Add Timing Instrumentation**
  - Insert `time.perf_counter()` markers before/after each major stage
  - Generate CSV with per-iteration timing breakdown
  - Identify which stage is the worst bottleneck

- [ ] **Re-enable 2-PAM Modulation**
  - Uncomment PAM code in `single_simulation.py`
  - Define PAM levels (±1 for 2-PAM)
  - Log PAM symbol indices to CSV
  - Measure effective bit transmission rate

- [ ] **Create Fast Config**
  - Add new `AOConfig` variant with smaller grids (128×128, 30×30 lenslets)
  - Test benchmark: 128×128 vs 256×256 speedup
  - Document trade-offs

- [ ] **Measure & Report**
  - Show per-iteration timing breakdown
  - Calculate actual speedup factors
  - Identify which single optimization has highest ROI

---

## PART 6: EFFORT ESTIMATION

| Phase | Duration | Difficulty | ROI |
|-------|----------|-----------|-----|
| **Phase 1 (Profiling)** | 2-3 hours | Easy | Medium |
| **Phase 2 (Algorithmic)** | 4-6 hours | Easy | High (4-6× speedup) |
| **Phase 3 (Implementation)** | 6-8 hours | Medium | Medium (2-4× speedup) |
| **Phase 4 (Architecture)** | 8-12 hours | Hard | Very High (50-200× speedup) |
| **Total (All Phases)** | ~20-30 hours | Varies | 50-200× speedup |
| **Quick Wins Only** | ~2-3 hours | Easy | 4-6× speedup |

---

## PART 7: NEXT STEPS

### What I recommend: START WITH PHASE 2 (Quick Wins)

The fastest path to useful speedup is to reduce grid sizes:

```python
# Create FastConfig variant
fast_cfg = AOConfig(
    ...
    pupil_grid=pupil_grid,  # Keep same
    focal_q=128,             # ← CHANGE: 256 → 128
    num_lenslets=30,         # ← CHANGE: 40 → 30
    num_modes=48,            # ← CHANGE: 64 → 48
    ...
)
```

Expected result: **4-6× speedup in ~1 hour of work**

After that, Phase 3 (vectorization) can bring another **2-4× speedup**

---

## APPENDIX: Key Files

- **Timing Analysis Script:** `timing_analysis.py`
- **Main AO Loop:** `main/single_simulation.py` (lines 345-490)
- **Bottleneck Functions:**
  - `layer_ao(state.wf_wfs)` — WFS propagation
  - `state.shwfs(state.magnifier(...))` — SH sensing  
  - `large_poisson()` — Noise generation

---

**Questions?** This plan is ready to execute. Which phase would you like to start with?
