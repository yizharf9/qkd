import os
import utils
from pathlib import Path
import numpy as np
import datetime as dt
print("\n")
print("Starting massive simution runs...")
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
num_of_r0_samples = 10
start = 0.01
end = 0.15
r0_list = np.linspace(start=start,stop=end,num=num_of_r0_samples,)

<<<<<<< HEAD
# wavelength values
wavelengths = [1.55e-6, 0.500e-6]

# num of runs for values specified values
N = 4
count = 1
for wavelength in wavelengths :
=======
N = 100
N1=2
for wavelength in [1.55e-6, 0.500e-6] : #run with turbulence 100 {N} runs
>>>>>>> 80d6f64 (change massive run  to add a non turb data)
    for r0_ref in r0_list :
        for run_number in range(1,N+1):  # example run number
            ns = {
                    "__name__": "__main__",
                    "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                    "run_number": run_number,
                    "wavelength": wavelength,
                    "r0_ref": r0_ref,  # example value for r0_ref
                    "save_images": save_images,  # example value for r0_ref
<<<<<<< HEAD
                    "TurbulencLayer": TurbulencLayer,
                }
            print(f"\n=== run_number={count - 1} finished out of {len(r0_list) * len(wavelengths) * N  } ({count/(len(r0_list) * len(wavelengths) * N  ) * 100:.3f}%) ===")
=======
                    "TurbulencLayer": True,
               }

            print(f"\n=== run_number={run_number} ===")
>>>>>>> c30cd50 (look for what_is_new.ipynb)
            exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
<<<<<<< HEAD
            count += 1

=======

for wavelength in [1.55e-6, 0.500e-6] : #run without turbulence 2 {N1} runs
            for run_number in range(1,N1+1):  # example run number
                ns = {
                        "__name__": "__main__",
                        "__file__": str("single_simulation.py"),   # so Path(__file__) works inside turblance.py
                        "run_number": run_number,
                        "wavelength": wavelength,
                        "r0_ref": r0_ref,  # example value for r0_ref
                        "save_images": save_images,  # example value for r0_ref
                        "TurbulencLayer": False,
                }

                print(f"\n=== run_number={run_number} ===")
                exec(code_obj, ns,ns)  # fresh globals per run (no locals dict)
>>>>>>> 80d6f64 (change massive run  to add a non turb data)
