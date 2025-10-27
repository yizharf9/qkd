import plot_and_graph as pag
import pytest
print("All modules imported successfully.")
# save_images:
"✅only 1 group of images are saved per run of massive_simulation.py based on user input at the begining of massive_simulation.py"
#TODO
"""

#✅ plots_and_graph: Cn2 handling

# ✅flag_turbalence_layer: 

""""""
✅Add the flag to control whether turbulence layer is applied or not in single_simulation.py
"""
#-------tests for graphs and plots-------------------
#tests for plots_and_graph.py with Cn2 as r0_col§
# in your script/module
@pytest.mark.parametrize("flag,expected", [
    (True,  "Cn2"),
    (False, "r0_ref"),
])
def test_select_x_column(flag, expected):
    if pag.check_axis(run_Cn2=flag,r0_col="r0_ref") == expected:
        return f"✅Success"
    else:
        return f"❌Failure"

pag_t1=test_select_x_column(True, "Cn2")
pag_t2=test_select_x_column(False, "r0_ref")
print("tests for plots_and_graph.py with Cn2 as r0_col passed successfully.")
print("pag_t1: ",pag_t1)
print("pag_t2: ",pag_t2)