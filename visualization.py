#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns

# %%
tar_input = torch.load("dataset/target_input.pt")

tar_input_die = tar_input[0, 0, 0].cpu().numpy()
tar_input_pcb = tar_input[0, 0, 1].cpu().numpy()
tar_input_trace = tar_input[0, 0, 2].cpu().numpy()
tar_input_temp = tar_input[0, 0, 3].cpu().numpy()

plt.imshow(tar_input_temp.reshape(50, 50), alpha=0.8)
# plt.imshow(tar_input_trace.reshape(50, 50), alpha=0.4)
# plt.imshow(tar_input_pcb.reshape(50, 50), alpha=0.3)
# plt.imshow(tar_input_die.reshape(50, 50), alpha=0.2)
#%%
tar_recipes = torch.load("dataset/target_target.pt")
tar_input_raw = torch.load("dataset/target_input_raw.pt")
#%%
src_recipes = torch.load("dataset/source_target.pt")
src_input_raw = torch.load("dataset/source_input_raw.pt")
src_input = torch.load("dataset/source_input.pt")
#%% 
res_raw = []
for i in range(len(src_input_raw)):
   src_input_raw_die = src_input_raw[i, :, 2, 0, 0].max().cpu().numpy().tolist()
   if src_input_raw_die not in res_raw:
      res_raw.append(src_input_raw_die)
#%%
from collections import defaultdict
recipe_unique = defaultdict(list)
for i in range(src_recipes.shape[0]):
    for j in range(src_recipes.shape[1]):
       if src_recipes[i, j].cpu().numpy().tolist() not in recipe_unique[f"zone{j}"]:
         recipe_unique[f"zone{j}"].append(src_recipes[i, j].cpu().numpy().tolist())

#%%
src_input.shape
972 * 15 = 14580


# %%
src_input = torch.load("dataset/source_input.pt")

src_input_die = src_input[0, 0, 0].cpu().numpy()
src_input_pcb = src_input[0, 0, 1].cpu().numpy()
src_input_trace = src_input[0, 0, 2].cpu().numpy()
src_input_temp = src_input[0, 0, 3].cpu().numpy()
src_input[0]

# %%
num_steps = 15
fig, axs = plt.subplots(nrows=1, ncols=num_steps, figsize=(15, 7))
for i in range(num_steps):
   heatmap = pd.read_csv(f"heatmap_simulation/IMG_1_1_{i+1}.csv")
   axs[i].imshow(heatmap)
   axs[i].set_axis_off()
fig.tight_layout()
plt.savefig("Figure/simulation_heatmap1.png", dpi=300)
# %%
fig, axs = plt.subplots(nrows=1, ncols=num_steps, figsize=(15, 7))
for i in range(num_steps):
   heatmap = pd.read_csv(f"heatmap_experiment/IMG_1_1_{i+1}.csv")
   axs[i].imshow(heatmap, interpolation = 'sinc', vmin=280, vmax=550)
   axs[i].set_axis_off()
fig.tight_layout()
plt.savefig("Figure/experiment_heatmap1.png", dpi=300)


# %%
recipe_list = []
num_area = 7
for i in range(num_area):
   df = pd.read_csv(f"recipe_simulation/recipe_1_{i+1}.csv")
   recipe = df.iloc[0, 0].item()
   recipe_list.append(recipe)

fig, ax = plt.subplots(figsize=(4,3))
ax.bar(x=np.arange(1, 8), height=recipe_list)
ax.set_xticks(np.arange(1,8))
ax.set_yticks(np.arange(0,350, 50))
ax.set_xlabel("Heating area", fontsize=14)
ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
fig.tight_layout()
plt.savefig("Figure/simulation_recipe1.eps", dpi=300, bbox_inches="tight")

# %%
recipe_list = []
num_area = 7
for i in range(num_area):
   df = pd.read_csv(f"recipe_experiment/recipe_1_{i+1}.csv")
   recipe = df.iloc[0, 0].item()
   recipe_list.append(recipe)

fig, ax = plt.subplots(figsize=(4,3))
ax.bar(x=np.arange(1, 8), height=recipe_list, color="orange")
ax.set_xticks(np.arange(1,8))
ax.set_yticks(np.arange(0,350, 50))
ax.set_xlabel("Heating area", fontsize=14)
ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
fig.tight_layout()
plt.savefig("Figure/experiment_recipe1.eps", dpi=300, bbox_inches="tight")
#%%
die = pd.read_csv(f"geo_img/M1_DIE.csv")
plt.imshow(die, vmin=0, vmax=3)
die.max().max()
#%%
pcb = pd.read_csv(f"geo_img/M1_PCB.csv")
plt.imshow(pcb, vmin=0, vmax=3)
pcb.max().max()
#%%
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(2,6))
sub = pd.read_csv(f"geo_img/M1_Substrate.csv")
pcb = pd.read_csv(f"geo_img/M1_PCB.csv")
die = pd.read_csv(f"geo_img/M1_DIE.csv")
axs[0].set_title("Die", fontsize=14)
axs[0].imshow(die, vmin=0, vmax=3)
axs[0].set_axis_off()
axs[1].set_title("Substrate", fontsize=14)
axs[1].imshow(sub, vmin=0, vmax=3)
axs[1].set_axis_off()
axs[2].set_title("PCB", fontsize=14)
axs[2].imshow(pcb, vmin=0, vmax=3)
axs[2].set_axis_off()
fig.tight_layout()
plt.savefig("Figure/M1_geom.eps", dpi=300)
# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6,2))
sub = pd.read_csv(f"geo_img/M12_Substrate.csv")
pcb = pd.read_csv(f"geo_img/M12_PCB.csv")
die = pd.read_csv(f"geo_img/M12_DIE.csv")
print(sub.max().max(), pcb.max().max(), die.max().max())
axs[0].set_title("Die", fontsize=14)
axs[0].imshow(die, vmin=0, vmax=2)
axs[0].set_axis_off()
axs[1].set_title("Substrate", fontsize=14)
axs[1].imshow(sub, vmin=0, vmax=2)
axs[1].set_axis_off()
axs[2].set_title("PCB", fontsize=14)
axs[2].imshow(pcb, vmin=0, vmax=2)
axs[2].set_axis_off()
fig.tight_layout()
plt.savefig("Figure/M12_geom.png", dpi=300)

# %% Recipe and temprature prodile curves
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
#%%
fig, ax = plt.subplots(figsize=(4,3))
data = pd.read_csv("experiment_smooth_JunKataoka.csv")
timestep = [0, 33, 66, 99, 132, 171, 204, 214, 224,
            234, 244, 254, 264, 274, 284, 294]
data = data.iloc[timestep, :]
data = data.reset_index()
ax.plot(timestep, data["M9_solder_measurement"], label="Solder temp.")
ax.plot(timestep, data["M9_board_measurement"], label="Board temp.")
recipe_timestep = [0,33,34,66,67,99,100,132,133,171,172,204,205,237,250,300]
recipe_temp = [120,120,150,150,180,180,220,220,260,260,280,280,300,300,96.85,96.85]
ax.plot(recipe_timestep, recipe_temp, label="Recipe")
ax.set_ylim(None, 500)
ax.set_xlabel("Time (sec)")
ax.set_ylabel(r"Temperature ($^\circ$C)")
ax.legend(loc="upper right")
plt.savefig("Figure/experiment_M9_tempviz.eps", dpi=300, bbox_inches='tight')

#%% Create color bar
npts = 1000
nbins = 15

x = np.random.uniform(low=368-273.15, high=543-273.15, size=(1000, 1000))

fig = plt.figure(figsize=(12,9))
ax = plt.gca()
im = plt.imshow(x, cmap="jet")
cbar = plt.colorbar(label=r"Temperature ($^\circ$C)");
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=18)
cbar.set_label("Temperature ($^\circ$C)", size=18)
plt.savefig("Figure/colorbar.eps", dpi=300)

# %%
