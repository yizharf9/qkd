import os
from pathlib import Path
import numpy as np
import time
import datetime as dt
print("\n")
print("Starting massive simulation runs...")
print("Current working directory:", os.getcwd())
print("Script start time:", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )

path_file="./single_simulation.py" #! <=== needs to be relative path to work
# edit as needed

save_images_prompt = input("save images (y/n) ?  ") #! <=== change to save photos in single turblance.py run 
if save_images_prompt == "y" :
    save_images = True
elif save_images_prompt == "n" :
    save_images = False
else : 
    exit("not a valid input!")

code = open(path_file,"r", encoding="utf-8").read()
code_obj = compile(code, "single_simulation.py", "exec")

# r0_list=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] # original values of simulation
num_of_r0_samples = 10
start = 0.01
end = 0.15
r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples,)

N = 20

for wavelength in [1.55e-6, 0.500e-6] :
    for r0_ref in r0_list :
        for run_number in range(1,N+1):  # example run number
            ns = {
                    "__name__": "__main__",
                    "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                    "run_number": run_number,
                    "wavelength": wavelength,
                    "r0_ref": r0_ref,  # example value for r0_ref
                    "save_images": save_images,  # example value for r0_ref
                }

            print(f"\n=== run_number={run_number} ===")
            exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
