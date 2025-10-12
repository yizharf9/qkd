import os
from pathlib import Path

#print("\n")
#print("path")
file_name="turblance.py"
folder_name="oct_2025"
# If you already know the full path as a string:
#print("/Users/idoshlomy/PycharmProjects/qkd/oct_2025/turblance.py")
path_for_run=os.path.abspath(file_name)
path_for_run=(str(path_for_run)[0:len(str(path_for_run))-len(file_name)]+folder_name+"/"+file_name)
#print(len(path_for_run))
#print(path_for_run=="/Users/idoshlomy/PycharmProjects/qkd/oct_2025/turblance.py")
# runner.py (place next to turblance.py)

  # edit as needed
code = open(path_for_run,"r", encoding="utf-8").read()
code_obj = compile(code, "turblance.py", "exec")
r0_list=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] # example values for r0 in meters
for wavelength in [1.55e-6, 0.500e-6] :
    for r0 in r0_list :
        for run_number in range(1, 5):
            ns = {
                    "__name__": "__main__",
                    "__file__": str(path_for_run),   # so Path(__file__) works inside turblance.py
                    "run_number": run_number,
                    "wavelength": wavelength,
                    "r0": r0,  # example value for r0
                }

            print(f"\n=== run_number={run_number} ===")
            exec(code_obj, ns)  # fresh globals per run (no locals dict)
