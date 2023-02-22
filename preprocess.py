# %%
from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import ConvLSTMCell
import os
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split

class args:
    seed = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#%%
def generate_target(root_recipe, num_area, num_geom, num_recipe, remove_geom):


    res = []
    for j in range(num_geom):
        if j not in remove_geom:
            out = np.empty((num_recipe, num_area))
            for i in range(num_recipe):
                for k in range(num_area):
                    recipe_path = f"recipe_{i+1}_{k+1}.csv"
                    recipe_img = np.genfromtxt(os.path.join(root_recipe, recipe_path), delimiter=",")
                    out[i, k] = recipe_img[0, 0]
            res.append(out)


    res = np.concatenate(res, axis=0)

    return res


def generate_input(root_geom, root_heatmap, seq_len, num_geom, num_recipe, remove_geom):

    res = []

    for j in range(num_geom):
        if j not in remove_geom:
            out = np.empty((num_recipe, seq_len, 4, 50, 50))
            die_path = f"M{j+1}_DIE.csv"
            pcb_path = f"M{j+1}_PCB.csv"
            trace_path = f"M{j+1}_Substrate.csv"
            die_img = np.genfromtxt(os.path.join(root_geom, die_path), delimiter=",")
            pcb_img = np.genfromtxt(os.path.join(root_geom, pcb_path), delimiter=",")
            trace_img = np.genfromtxt(os.path.join(root_geom, trace_path), delimiter=",")
            print(die_img.max())

            for i in range(num_recipe):
                for k in range(seq_len):
                    heatmap_path = f"IMG_{j+1}_{i+1}_{k+1}.csv"
                    heatmap_img = np.genfromtxt(os.path.join(root_heatmap, heatmap_path), delimiter=",")
                    out[i, k, 0] = die_img
                    out[i, k, 1] = pcb_img
                    out[i, k, 2] = trace_img
                    out[i, k, 3] = heatmap_img
            res.append(out)
    
    res = np.concatenate(res, axis=0)
    return res

def generate_tardomain_input(root_geom, root_heatmap, seq_len, geom_id, num_recipe):

    res = []

    out = np.empty((num_recipe, seq_len, 4, 50, 50))

    for i in range(num_recipe):
        for k in range(seq_len):

            die_path = f"M{geom_id+1}_DIE.csv"
            pcb_path = f"M{geom_id+1}_PCB.csv"
            trace_path = f"M{geom_id+1}_Substrate.csv"
            heatmap_path = f"IMG_{geom_id+1}_{i+1}_{k+1}.csv"

            die_img = np.genfromtxt(os.path.join(root_geom, die_path), delimiter=",")
            pcb_img = np.genfromtxt(os.path.join(root_geom, pcb_path), delimiter=",")
            trace_img = np.genfromtxt(os.path.join(root_geom, trace_path), delimiter=",")
            heatmap_img = np.genfromtxt(os.path.join(root_heatmap, heatmap_path), delimiter=",")
            out[i, k, 0] = die_img
            out[i, k, 1] = pcb_img
            out[i, k, 2] = trace_img
            out[i, k, 3] = heatmap_img

        res.append(out)


    res = np.concatenate(res, axis=0)

    return res

#%%
remove_geom_list = []

a = generate_target(root_recipe="recipe_simulation", num_area=7, num_geom=12, num_recipe=81, remove_geom=remove_geom_list)
target_tensor = torch.tensor(a).cuda()
torch.save(target_tensor, f"./dataset/source_target.pt")

a = generate_target(root_recipe="recipe_experiment", num_area=7, num_geom=12, num_recipe=3, remove_geom=[i for i in range(12) if i != 0])
target_tensor = torch.tensor(a).cuda()
torch.save(target_tensor, f"./dataset/target_target.pt")

target_target_cv1_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,1]).cuda())
target_target_cv1_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([2]).cuda())
torch.save(target_target_cv1_train, "./dataset/target_target_cv1_train.pt")
torch.save(target_target_cv1_test, "./dataset/target_target_cv1_test.pt")

target_target_cv2_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,2]).cuda())
target_target_cv2_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([1]).cuda())

torch.save(target_target_cv2_train, "./dataset/target_target_cv2_train.pt")
torch.save(target_target_cv2_test, "./dataset/target_target_cv2_test.pt")

target_target_cv3_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([1,2]).cuda())
target_target_cv3_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([0]).cuda())
torch.save(target_target_cv3_train, "./dataset/target_target_cv3_train.pt")
torch.save(target_target_cv3_test, "./dataset/target_target_cv3_test.pt")

# %%
print("Generating input data")
def MinMaxNormalize(input_tensor):
    res_max = torch.FloatTensor([0, 0, 0, 0]).cuda()
    res_min = torch.FloatTensor([1e+10, 1e+10, 1e+10, 1e+10]).cuda()
    for i in range(input_tensor.shape[0]):

        for j in range(input_tensor.shape[1]):
            minimum = torch.min(input_tensor[i, j, :].view(4, -1), dim=1)[0]
            maximum = torch.max(input_tensor[i, j, :].view(4, -1), dim=1)[0]

            res_max = torch.FloatTensor([max(res_max[i], maximum[i]) for i in range(len(res_max))])
            res_min = torch.FloatTensor([min(res_min[i], minimum[i]) for i in range(len(res_min))])
        
    return res_min, res_max
#%%
a = generate_input(root_geom="./geo_img", root_heatmap="./heatmap_simulation",
                   seq_len=15, num_geom=12, num_recipe=81, remove_geom = remove_geom_list)
input_tensor = torch.tensor(a).cuda()
#%%
res_min, res_max = MinMaxNormalize(input_tensor)
res_min, res_max = res_min.cuda(), res_max.cuda()
input_tensor_normalized = (input_tensor - res_min[None, None, :, None, None]) / (res_max[None, None, :, None, None] - res_min[None, None, :, None, None])
#%%
torch.save(input_tensor_normalized, "./dataset/source_input.pt")
torch.save(input_tensor, "./dataset/source_input_raw.pt")
#%%
print("Generating target input data")
a = generate_tardomain_input(root_geom="./geo_img", root_heatmap="./heatmap_experiment",
                   seq_len=15, geom_id=0, num_recipe=3)
input_tensor = torch.tensor(a).cuda()
#%%
res_min, res_max = MinMaxNormalize(input_tensor)
res_min, res_max = res_min.cuda(), res_max.cuda()
input_tensor_normalized = (input_tensor - res_min[None, None, :, None, None]) / (res_max[None, None, :, None, None] - res_min[None, None, :, None, None])
torch.save(input_tensor_normalized, "./dataset/target_input.pt")
torch.save(input_tensor, "./dataset/target_input_raw.pt")

#%%

# %%
# target_input_cv1_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0,1]).cuda())
# target_input_cv1_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([2]).cuda())
# torch.save(target_input_cv1_train, "./dataset/target_input_cv1_train.pt")
# torch.save(target_input_cv1_test, "./dataset/target_input_cv1_test.pt")

# target_input_cv2_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0,2]).cuda())
# target_input_cv2_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([1]).cuda())
# torch.save(target_input_cv2_train, "./dataset/target_input_cv2_train.pt")
# torch.save(target_input_cv2_test, "./dataset/target_input_cv2_test.pt")

# target_input_cv3_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([1,2]).cuda())
# target_input_cv3_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0]).cuda())
# torch.save(target_input_cv3_train, "./dataset/target_input_cv3_train.pt")
# torch.save(target_input_cv3_test, "./dataset/target_input_cv3_test.pt")
