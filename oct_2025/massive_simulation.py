import utils
import numpy as np
import time
from params import massive_simulation_begin_massage 
utils.check_dir()
try:
    print(massive_simulation_begin_massage)
except NameError:
    print("\"./params.py\" import executed incorrectly!")
    exit()

path_file="./single_simulation.py" #! <=== needs to be relative path to work
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

code = open(path_file,"r", encoding="utf-8").read()
code_obj = compile(code, "single_simulation.py", "exec")



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

time_for_start=time.time()
# run with TurbulencLayer
for wavelength in wavelengths :
    for r0_ref in r0_list :
        for run_number in range(1,N+1):  # example run number
                ns = {
                        "__name__": "__main__",
                        "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                        "run_number": run_number,
                        "wavelength": wavelength,
                        "r0_ref": r0_ref,  # example value for r0_ref
                        "save_images": save_images,  # example value for r0_ref
                        "TurbulencLayer": TurbulencLayer,
                        "Add_Stellar_Noise": Add_Stellar_Noise,
                        "USE_AO": USE_AO,
                        "Run_test_batch": Run_test_batch,
                    }
                number_of_runs=len(r0_list) * len(wavelengths) * N *len(focal_dim)
                print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                current_time_for_run=time.time()
                count += 1
                T2FInish=utils.time_estimate(current_time_for_run,time_for_start,count,number_of_runs)
                print("Time estimate to finish: "+str(int(T2FInish/60))+" minutes")

# run without TurbulencLayer
count = 1
for focal in focal_dim:
    for wavelength in wavelengths :
        for r0_ref in r0_list :
            for run_number in range(1,N2+1):  # example run number
                ns = {
                        "__name__": "__main__",
                        "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                        "run_number": run_number,
                        "wavelength": wavelength,
                        "r0_ref": r0_ref,  # example value for r0_ref
                        "save_images": save_images,  # example value for r0_ref
                        "TurbulencLayer": False,
                        "Add_Stellar_Noise": Add_Stellar_Noise,
                        "USE_AO": USE_AO,
                        "Run_test_batch": Run_test_batch,
                        "focal_dim":focal,
                    }
                number_of_runs=len(r0_list) * len(wavelengths) * N2 *len(focal_dim)
                print(f"\n=== run_number={count - 1} finished out of {number_of_runs } ({count/(number_of_runs) * 100:.3f}%) ===")
                exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
                count += 1
