#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# %%
tar_input = torch.load("dataset/target_input.pt")

tar_input_die = tar_input[0, 0, 0].cpu().numpy()
tar_input_pcb = tar_input[0, 0, 1].cpu().numpy()
tar_input_trace = tar_input[0, 0, 2].cpu().numpy()
tar_input_temp = tar_input[0, 0, 3].cpu().numpy()

# plt.imshow(tar_input_temp.reshape(50, 50), alpha=0.8)
plt.imshow(tar_input_trace.reshape(50, 50), alpha=0.4)
plt.imshow(tar_input_pcb.reshape(50, 50), alpha=0.3)
plt.imshow(tar_input_die.reshape(50, 50), alpha=0.2)

# %%
src_input = torch.load("dataset/source_input.pt")

src_input_die = src_input[0, 0, 0].cpu().numpy()
src_input_pcb = src_input[0, 0, 1].cpu().numpy()
src_input_trace = src_input[0, 0, 2].cpu().numpy()
src_input_temp = src_input[0, 0, 3].cpu().numpy()

# plt.imshow(src_input_temp.reshape(50, 50), alpha=0.8)
plt.imshow(src_input_trace.reshape(50, 50), alpha=0.4)
plt.imshow(src_input_pcb.reshape(50, 50), alpha=0.3)
plt.imshow(src_input_die.reshape(50, 50), alpha=0.2)
# %%
