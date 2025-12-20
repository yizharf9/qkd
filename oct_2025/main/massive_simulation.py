import utils
import numpy as np
import time
import math
from params import massive_simulation_begin_massage 
utils.check_dir()
try:
    print(massive_simulation_begin_massage)
except NameError:
    print("\"./params.py\" import executed incorrectly!")
    exit()

path_file="./main/single_simulation.py" #! <=== needs to be relative path to work
# edit as needed

Run_test_batch_prompt = input("run test batch (y/n) ?  ") 
if Run_test_batch_prompt == "y" :
    Run_test_batch = True
elif Run_test_batch_prompt == "n" :
    Run_test_batch = False
else : 
    exit("not a valid input!")

save_images_prompt = input("save images (y/n) ?  ") 
if save_images_prompt == "y" :
    save_images = True
elif save_images_prompt == "n" :
    save_images = False
else : 
    exit("not a valid input!")

TurbulencLayer_prompt = input("add layer with no turbulance (y/n) ?  ") 
if TurbulencLayer_prompt == "y" :
    TurbulencLayer = True
elif TurbulencLayer_prompt == "n" :
    TurbulencLayer = False
else : 
    exit("not a valid input!")


Add_Stellar_Noise_prompt = input("add stellar noise (y/n) ?  ") 
if Add_Stellar_Noise_prompt == "y" :
    Add_Stellar_Noise = True
elif Add_Stellar_Noise_prompt == "n" :
    Add_Stellar_Noise = False
else : 
    exit("not a valid input!")

USE_AO_prompt = input("use adaptive optics (y/n) ?  ") 
if USE_AO_prompt == "y" :
    USE_AO = True
elif USE_AO_prompt == "n" :
    USE_AO = False
else : 
    exit("not a valid input!")
if USE_AO:
    Multy_use_AO_prompt = input("USE multy mode AO (y/n)? ")
    if Multy_use_AO_prompt.lower() == "y":
        Multy_use_AO = True
    elif Multy_use_AO_prompt.lower() == "n":
        Multy_use_AO = False
    else:
        exit("Not a valid input! Please enter 'y' or 'n'.")


code = open(path_file,"r", encoding="utf-8").read()
print("code: \n",code[:60],"...\n")
code_obj = compile(code, "main/single_simulation.py", "exec")



if Run_test_batch :
    num_of_r0_samples = 3
    start = 0.01
    end = 0.15
    r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples)
    wavelengths = [1.55e-6]
    focal_dim=[8]
    N = 1
    N2=1
else :
    # r0 values
    num_of_r0_samples = 10
    start = 0.01
    end = 0.15
    r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples,)
    focal_dim=[8,1,1e-3,500e-6,9e-6]
    # wavelength values
    wavelengths = [1.55e-6, 0.500e-6]
    # num of runs for values specified values
    N = 5 
    N2=4
count = 1
wavelengths=1.55e-6
time_for_start=time.time()
num_arays=[1,2,3,4,5]
r0_ref_list=[0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02]
"run with TurbulencLayer"
for num_airy in [2,3,4]:
    for run_number in range(1,N+1):  # example run number
            ns = {
                            "__name__": "__main__",
                            "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                            "run_number": run_number,
                            "wavelength": 1.55e-6,
                            "r0_ref": 0.1,  # example value for r0_ref 
                            "save_images": save_images,  # example value for r0_ref
                            "TurbulencLayer": True,
                            "USE_AO": True, # !Will te a long run to run if True
                            "num_airy": num_airy,
                            "r0_ref_list":r0_ref_list,
                            "Add_Stellar_Noise":Add_Stellar_Noise,
                            "Multy_use_AO": Multy_use_AO,
                            #!if Multy_use_AO is True then AO use r0_ref_list no r0_ref
                            #!Only if r0_ref_list is none Then AO use r0_ref
                        }
            number_of_runs= len(num_arays) * N
            print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
            exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
            current_time_for_run=time.time()
            count += 1
            T2FInish=utils.time_estimate(current_time_for_run,time_for_start,count,number_of_runs)
            print("Time asstimate to finish: "+str(int(T2FInish/60))+" minutes")
"run without TurbulencLayer"
if TurbulencLayer == True:
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
                            "USE_AO": True,
                            "num_airy": num_airy,
                            "r0_ref_list":r0_ref_list,
                            "Add_Stellar_Noise":Add_Stellar_Noise,
                            "Multy_use_AO": Multy_use_AO,
                            #!if Multy_use_AO is True then AO use r0_ref_list no r0_ref
                            #!Only if r0_ref_list is none Then AO use r0_ref
                            }
                    number_of_runs=len(r0_list) * len(num_arays) * N
                    print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                    exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                    count += 1
