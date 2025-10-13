import os
from pathlib import Path
import time
import datetime as dt
print("\n")
print(391) #first raw is 391 13/10/2025 16:49 
print("Starting massive simulation runs...")
print("Current working directory:", os.getcwd())
print("Script start time:", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )

path_file="/Users/idoshlomy/PycharmProjects/qkd/oct_2025/turblance.py"
  # edit as needed
code = open(path_file,"r", encoding="utf-8").read()
code_obj = compile(code, "turblance.py", "exec")
r0_list=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] # example values for r0 in meters
for wavelength in [1.55e-6, 0.500e-6] :
    for r0_ref in r0_list :
        for run_number in range(1,10):  # example run number
            ns = {
                    "__name__": "__main__",
                    "__file__": str("turblance.py"),   # so Path(__file__) works inside turblance.py
                    "run_number": run_number,
                    "wavelength": wavelength,
                    "r0_ref": r0_ref,  # example value for r0_ref
                }

            print(f"\n=== run_number={run_number} ===")
            exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
