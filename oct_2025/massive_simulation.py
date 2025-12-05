import os
import utils
from pathlib import Path
import numpy as np
import datetime as dt
import time
import math

utils.check_dir()
path_file="./single_simulation.py" #! <=== needs to be relative path to work
# edit as needed
save_images_prompt = input("save images (y/n) ?  ") #! <=== change to save photos in single turblance.py run 
if save_images_prompt == "y" :
    save_images = True
elif save_images_prompt == "n" :
    save_images = False
else : 
    exit("not a valid input!")

TurbulencLayer_prompt = input("add layer with no turbulance (y/n) ?  ") #! <=== change to add layer without turb photos in single turblance.py run 
if TurbulencLayer_prompt == "y" :
    TurbulencLayer = True
elif TurbulencLayer_prompt == "n" :
    TurbulencLayer = False
else : 
    exit("not a valid input!")

Noise_prompt = input("Add noise (y/n)? ")
if Noise_prompt.lower() == "y":
    Noise = True
elif Noise_prompt.lower() == "n":
    Noise = False
else:
    exit("Not a valid input! Please enter 'y' or 'n'.")


code = open(path_file,"r", encoding="utf-8").read()
code_obj = compile(code, "single_simulation.py", "exec")

# r0 values
num_of_r0_samples = 4
start = 0.01
end = 0.15
r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples)
# r0_list = [0.075]
focal_dim=[500e-6]
# wavelength values
# wavelengths = [1.55e-6, 0.500e-6]
wavelengths = [1.55e-6]
variances = np.logspace(4,7,7)
# [1e4,1e5,1e6,1e7,1e8]

# num of runs for values specified values
N = 50
realizations_per_run_no_turb=4

count = 1
time_for_start=time.time()
print("run with TurbulencLayer")
for focal in focal_dim:
    for wavelength in wavelengths :
        for r0_ref in r0_list :
            for noise_var in variances :
                for sent_signal in [True,False] :
                    for run_number in range(1,N+1):  # example run number
                        ns = {
                                "__name__": "__main__",
                                "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                                "run_number": run_number,
                                "wavelength": wavelength,
                                "r0_ref": r0_ref,  
                                "noise_var": noise_var,  
                                "sent_signal": sent_signal,  
                                "save_images": save_images,  
                                "Noise": Noise,  
                                "TurbulencLayer": TurbulencLayer,
                                "focal_dim": focal,
                            }
                        number_of_runs = len(wavelengths) * len(r0_list) * N * len(focal_dim) * len(variances) * 2
                        print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                        exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                        current_time_for_run=time.time()
                        count += 1
                        T2FInish=utils.time_asstimate(current_time_for_run,time_for_start,count,number_of_runs)
                        print("Time asstimate to finish: "+str(int(T2FInish/60))+" minutes")

print("run without TurbulencLayer")
if TurbulencLayer :
    count = 1
    for focal in focal_dim:
        for wavelength in wavelengths :
            for r0_ref in r0_list :
                for run_number in range(1,realizations_per_run_no_turb+1):  # example run number
                    ns = {
                            "__name__": "__main__",
                            "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                            "run_number": run_number,
                            "wavelength": wavelength,
                            "r0_ref": r0_ref,  
                            "save_images": save_images,  
                            "Noise": Noise,  
                            "TurbulencLayer": False,
                            "focal_dim":focal,
                        }
                    number_of_runs=len(r0_list) * len(wavelengths) * realizations_per_run_no_turb *len(focal_dim)
                    print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                    exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                    count += 1
