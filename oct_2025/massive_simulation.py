import os
import utils
from pathlib import Path
import numpy as np
import datetime as dt
import time
import math
from OA import set_OA_params
print(math.e)
from params import massive_simulation_begin_massage 
utils.check_dir()
try:
    print(massive_simulation_begin_massage)
except NameError:
    print(2)
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

code = open(path_file,"r", encoding="utf-8").read()
code_obj = compile(code, "single_simulation.py", "exec")

# r0 values
num_of_r0_samples = 5
start = 0.05
end = 0.05
r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples,)
#focal_dim=[8,1,1e-3,500e-6,9e-6]
# wavelength values
wavelengths = [1.55e-6, 0.500e-6]
r0_ref_list=[0.01,0.05,0.03,0.07,0.2]
# num of runs for values specified values
N = 1
N2 = 1
count = 1
wavelengths=1.55e-6
time_for_start=time.time()
num_arays=[3]
"run with TurbulencLayer"
for num_airy in num_arays:
    OA_Params=set_OA_params(wavelengths,num_airy)
    reconstruction_matrix=OA_Params["reconstruction_matrix"]
    deformable_mirror=OA_Params["deformable_mirror"]
    shwfs=OA_Params["shwfs"]
    magnifier=OA_Params["magnifier"]
    camera=OA_Params["camera"]
    shwfse=OA_Params["shwfse"]
    slopes_ref=OA_Params["slopes_ref"]
    for num_airy in [1,2,3,4,5]:
        for run_number in range(1,N+1):  # example run number
            ns = {
                            "__name__": "__main__",
                            "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                            "run_number": run_number,
                            "wavelength": 1.55e-6,
                            "r0_ref": 0.1,  # example value for r0_ref
                            "save_images": save_images,  # example value for r0_ref
                            "TurbulencLayer": True,
                            "USE_OA": True,
                            "num_airy": num_airy,
                            "reconstruction_matrix":reconstruction_matrix,
                            "deformable_mirror":deformable_mirror,
                            "shwfs":shwfs,
                            "magnifier":magnifier,
                            "camera":camera,
                            "shwfse":shwfse,
                            "slopes_ref":slopes_ref,
                            "r0_ref_list":r0_ref_list,
                        }
            number_of_runs=len(r0_list) * len(num_arays) * N
            print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
            exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
            current_time_for_run=time.time()
            count += 1
            T2FInish=utils.time_asstimate(current_time_for_run,time_for_start,count,number_of_runs)
            print("Time asstimate to finish: "+str(int(T2FInish/60))+" minutes")
"run without TurbulencLayer"
flag_without_TurbulencLayer=False
if flag_without_TurbulencLayer:
    count = 1
    for num_airy in num_arays:
            for r0_ref in r0_list :
                for run_number in range(1,N2+1):  # example run number
                    ns = {
                            "__name__": "__main__",
                            "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                            "run_number": run_number,
                            "wavelength": 1.55e-6,
                            "r0_ref": r0_ref,  # example value for r0_ref
                            "save_images": save_images,  # example value for r0_ref
                            "TurbulencLayer": False,
                            "num_airy": num_airy,
                            "USE_OA": True,
                            "num_airy": num_airy,
                            "reconstruction_matrix":reconstruction_matrix,
                            "deformable_mirror":deformable_mirror,
                            "shwfs":shwfs,
                            "magnifier":magnifier,
                            "camera":camera,
                            "shwfse":shwfse,
                            "slopes_ref":slopes_ref,
                            "r0_ref_list":r0_ref_list,
                            }
                    number_of_runs=len(r0_list) * len(num_arays) * N
                    print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                    exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                    count += 1
