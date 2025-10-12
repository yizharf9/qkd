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

RUNS = [0, 1, 2, 3, 4]  # edit as needed

code = open(path_for_run,"r", encoding="utf-8").read()
code_obj = compile(code, "turblance.py", "exec")

for run_number in RUNS:
    ns = {"run_number": run_number, "__name__": "__main__"}
    print(f"\n=== run_number={run_number} ===")
    exec(code_obj, ns, ns)

