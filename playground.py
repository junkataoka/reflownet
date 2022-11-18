#%%
import numpy as np
import torch
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

# %% Create recipe csv files
recipe_num = 81
temps = [120, 150, 180]
sample = pd.read_csv("recipe_experiment/recipe_1_1.csv", header=None)
num_add_recipe = 3
recipe_list = { "M7":[105, 130, 160, 190, 230, 270, 290],
               "M8":[110, 140, 170, 200, 240, 280, 290],
                "M9":[120, 150, 180, 220, 260, 280, 300]}

c = 0
for key, val in recipe_list.items():
    for t in range(len(val)):
        file_name = f"recipe_experiment/recipe_{c + 1}_{t+1}.csv"
        sample.iloc[:, :] = val[t]
        sample.to_csv(file_name, index=False, header=False)
    c += 1

# %% Create
data = pd.read_csv("experiment_smooth_JunKataoka.csv")
result = []
result = [33, 66, 99, 132, 171, 204, 214, 224,
            234, 244, 254, 264, 274, 284, 294]

result = [i for i in result]

# %%
small_data = data.iloc[result, :]

die_M9 = small_data["M9_solder_measurement"].values
subtrate_M9 = small_data["M9_board_measurement"].values
pcb_M9 = small_data["M9_board_measurement"].values

die_M8 = small_data["M8_solder_measurement"].values
subtrate_M8 = small_data["M8_board_measurement"].values
pcb_M8 = small_data["M8_board_measurement"].values

die_M7 = small_data["M7_solder_measurement"].values
subtrate_M7 = small_data["M7_board_measurement"].values
pcb_M7 = small_data["M7_board_measurement"].values


# %%
def generate_heatmap(heatmap_root, die, subtrate, pcb, recipe_num, board_num):
    img = np.zeros((50, 50, 15))
    pcb_out = np.ones((30, 30, 15))
    subtrate_out = np.ones((24, 24, 15))
    die_out = np.ones((12, 17, 15))

    for i in range(15):
        die_out[:, :, i] = die[i] + 272.15
        subtrate_out[:, :, i] = subtrate[i] + 272.15
        pcb_out[:, :, i] = pcb[i] + 272.15

    img[:30, :30, :] = pcb_out
    img[:24, :24, :] = subtrate_out
    img[:12, :17, :] = die_out

    for i in range(15):
        df = pd.DataFrame(img[:, :, i])
        df.to_csv(f"{heatmap_root}/IMG_{board_num}_{recipe_num}_{i+1}.csv", index=False, header=False)

generate_heatmap("heatmap_experiment", die_M7, subtrate_M7, pcb_M7, 1, 1)
generate_heatmap("heatmap_experiment", die_M8, subtrate_M8, pcb_M8, 2, 1)
generate_heatmap("heatmap_experiment", die_M9, subtrate_M9, pcb_M9, 3, 1)
