import os
print(os.path.abspath("turblance.py"))
"""
# runner.py (place next to turblance.py)
RUNS = [0, 1, 2, 3, 4]  # edit as needed

code = open(file_path, "r", encoding="utf-8").read()
code_obj = compile(code, "turblance.py", "exec")

for run_number in RUNS:
    ns = {"run_number": run_number, "__name__": "__main__"}
    print(f"\n=== run_number={run_number} ===")
    exec(code_obj, ns, ns)
/Users/idoshlomy/PycharmProjects/qkd/oct_2025/turblance.py
"""