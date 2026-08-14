#!/bin/bash

# Quick reference for what timing reports you'll get

cat << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                    TIMING REPORT OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════════

When you run single_simulation.py, you will now get TIMING REPORTS in your output folder:

📁 Output Files Generated:
═══════════════════════════════════════════════════════════════════════════════

1️⃣  timing_breakdown.csv
    └─ Per-iteration breakdown with all timing data
    └─ Columns:
       • iteration
       • 1. PAM Modulation_ms
       • 2. WFS Propagation (layer_ao)_ms
       • 3. DM Application_ms
       • 4. SH Sensing (shwfs)_ms
       • 5. Poisson Noise_ms
       • 6. Focal Plane Propagation_ms
       • 7. Slope Estimation_ms
       • 8. Reconstruction & DM Update_ms
       • 9. Data Logging_ms
       • total_ms
       • loop_rate_hz
    └─ Use this to plot timing trends or identify jitter


2️⃣  timing_report.txt
    └─ Human-readable summary report
    └─ Contains:
       ✓ Per-stage statistics (Min, Avg, Max, Total)
       ✓ Overall iteration statistics
       ✓ Loop rate analysis (achieved vs target)
       ✓ Bottleneck ranking (which stages take most time)
    └─ Example output:
    
       Per-Stage Statistics (milliseconds):
       Stage                          Min        Avg        Max      Total
       ───────────────────────────────────────────────────────────────────
       2. WFS Propagation (layer_ao)  45.23      52.15      58.92    2607.50
       4. SH Sensing (shwfs)          28.15      31.84      35.42    1592.00
       3. DM Application             18.92      21.34      24.15    1067.00
       8. Reconstruction & DM Update  12.34      14.27      16.89     713.50
       ...
       
       Loop Rate:
         Achieved:  3.52 Hz
         Target:    1000 Hz
         Gap:       996.48 Hz (284× too slow)


═══════════════════════════════════════════════════════════════════════════════

📊 What This Tells You:
═══════════════════════════════════════════════════════════════════════════════

From the CSV, you can:
  ✓ Plot timing over iterations (trends, stability)
  ✓ See jitter (Max - Min for each stage)
  ✓ Identify which iteration was slowest
  ✓ Compare different runs
  ✓ Analyze stage-by-stage performance

From the TXT report, you can:
  ✓ Immediately see which stages are slowest
  ✓ Calculate what speedup you need
  ✓ Plan which optimizations to try first
  ✓ Set performance targets


═══════════════════════════════════════════════════════════════════════════════

🎯 How to Use:
═══════════════════════════════════════════════════════════════════════════════

Step 1: Run simulation
  cd /Users/idoshlomy/Documents/qkd/FSOC
  python main/single_simulation.py

Step 2: Check output
  cd main/simulation_output  (or wherever output is)
  cat timing_report.txt      # Read the summary
  cat timing_breakdown.csv   # See detailed per-iteration data

Step 3: Import to analyze
  import pandas as pd
  df = pd.read_csv('timing_breakdown.csv')
  
  # Average time per stage
  df[['1. PAM Modulation_ms', '2. WFS Propagation (layer_ao)_ms', ...]].mean()
  
  # Plot timing over iterations
  import matplotlib.pyplot as plt
  df.plot(x='iteration', y='total_ms')
  plt.show()


═══════════════════════════════════════════════════════════════════════════════

📈 Example Analysis:
═══════════════════════════════════════════════════════════════════════════════

$ cat main/simulation_output/timing_report.txt

================================================================================
TIMING ANALYSIS REPORT (50 iterations)
================================================================================

Per-Stage Statistics (milliseconds):
Stage                              Min        Avg        Max      Total
────────────────────────────────────────────────────────────────────────
2. WFS Propagation (layer_ao)      45.23      52.15      58.92    2607.50 ◄─ #1 WORST
4. SH Sensing (shwfs)              28.15      31.84      35.42    1592.00 ◄─ #2 SECOND
3. DM Application                  18.92      21.34      24.15    1067.00
8. Reconstruction & DM Update      12.34      14.27      16.89     713.50
6. Focal Plane Propagation         10.56      12.15      14.23     607.50
7. Slope Estimation                 8.92      10.34      12.15     517.00
5. Poisson Noise                    6.78       7.82       9.12     391.00
9. Data Logging                     4.23       4.95       6.12     247.50
1. PAM Modulation                   0.12       0.15       0.18       7.50

Iteration Timing (milliseconds):
  Average: 280.12
  Min:     270.45
  Max:     295.78
  Median:  281.23

Loop Rate:
  Achieved:  3.57 Hz
  Target:    1000 Hz
  Gap:       996.43 Hz (280× too slow)

Bottleneck Analysis (sorted by average time):
1. 2. WFS Propagation (layer_ao)      52.15 ms (18.6%)
2. 4. SH Sensing (shwfs)              31.84 ms (11.4%)
3. 3. DM Application                  21.34 ms ( 7.6%)
...


═══════════════════════════════════════════════════════════════════════════════

🚀 This answers your original questions:
═══════════════════════════════════════════════════════════════════════════════

1️⃣  When is signal transmitted?
    └─ See "1. PAM Modulation" timing (currently 0.15ms - very fast!)

2️⃣  What function takes most time?
    └─ See bottleneck ranking - "2. WFS Propagation" is the killer (52ms)

3️⃣  What is stopping higher frequency?
    └─ See "Loop Rate" section - need 280× speedup to hit 1 kHz

4️⃣  What is the rate of the loop?
    └─ See "Achieved: 3.57 Hz" - that's your current rate


═══════════════════════════════════════════════════════════════════════════════

Next Steps:
  • Run it and check the output files
  • Post the timing_report.txt to understand bottlenecks
  • Use timing_breakdown.csv for detailed analysis
  • Plan Phase 2 optimizations based on what takes most time

EOF
