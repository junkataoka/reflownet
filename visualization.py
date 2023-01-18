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

# %%
src_input = torch.load("dataset/source_input.pt")

src_input_die = src_input[0, 0, 0].cpu().numpy()
src_input_pcb = src_input[0, 0, 1].cpu().numpy()
src_input_trace = src_input[0, 0, 2].cpu().numpy()
src_input_temp = src_input[0, 0, 3].cpu().numpy()

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
plt.savefig("Figure/simulation_recipe1.png", dpi=300)

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
plt.savefig("Figure/experiment_recipe1.png", dpi=300)
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
plt.savefig("Figure/M1_geom.png", dpi=300)
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

# %%
