from gc import collect
import pickle as pkl
import datetime
from operator import mod
import os
from random import random
from statistics import mode
import sys
from tkinter import Y
from imageio import save
import numpy as np

from sklearn import datasets
import yaml
from _dataset.src.dataset import get_dataset_collate
from tqdm import *
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch
from model.PCMG import PCMG
from utils.furthestpointsample import farthest_point_sample,index_points

from utils.emd_loss import compute_emd_loss
from visuals.log import log_out_message

from  evaluate.ACTOR.models.get_model import get_model as get_ACTOR_model
from _parser.parser import get_parser
from _parser.ACTOR_parser import parser as get_ACOTR_parser
from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color
from openTSNE import TSNE
from matplotlib import cm

# ROOT_DIR = "D:\maweizhao\MyProgram\DeepLearning\myfile\1\PCMG"
# sys.path.append(ROOT_DIR)

humanact12_coarse_action_enumerator = {
    0: "Warm up",
    1: "Walk",
    2: "Run",
    3: "Jump",
    4: "Drink",
    5: "Lift dumbbell",
    6: "Sit",
    7: "Eat",
    8: "Turn steering wheel",
    9: "Phone",
    10: "Boxing",
    11: "Throw",
}

# humanact12_coarse_action_enumerator = {
#     0: "warm_up",
#     1: "walk",
#     2: "run",
#     3: "jump",
#     4: "drink",
#     5: "lift_dumbbell",
#     6: "sit",
#     7: "eat",
#     8: "turn steering wheel",
#     9: "phone",
#     10: "boxing",
#     11: "throw",
# }

_,args=get_parser()
print(args)

if args.model == "PCMG":
    model=PCMG(args).cuda()
    # 需要继续训练的模型的路径
    #begin_epoch=0
    ontraining_model_path="./check_point/done/modelpara.pth"
    #ontraining_model_path="./check_point/done/seq4.pth"
    if(os.path.exists(ontraining_model_path)):
        model_dict=torch.load(ontraining_model_path)
        #begin_epoch=model_dict["epoch"]
        model.load_state_dict(model_dict["net"])
        #optimizer.load_state_dict(model_dict["optimizer"])
if args.model == "ACTOR":
    folder="D:\maweizhao\MyProgram\DeepLearning\myfile/1\doing/1024vertices\person\smpl_cls_PCMG_pointnet_Transformer\evaluate\ACTOR\pretrained_models/humanact12"
    checkpointname='checkpoint_5000.pth.tar'
    def load_args(filename):
        with open(filename, "rb") as optfile:
            opt = yaml.load(optfile, Loader=yaml.Loader)
        return opt
    opt,parameters=get_ACOTR_parser()
    newparameters = {key: val for key, val in vars(opt).items() if val is not None}
    #folder, checkpointname = os.path.split(newparameters["checkpointname"])
    parameters_load = load_args(os.path.join(folder, "opt.yaml"))
    parameters.update(newparameters)
    parameters.update(parameters_load)
    # parameters["njoints"]=25
    # parameters["device"]=device
    parameters["pose_rep"]="rot6d"
    parameters["jointstype"]="vertices"
    model=get_ACTOR_model(parameters)
    model.outputxyz = True
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    epoch=checkpointname.split("_")[1].split(".")[0]
    if(os.path.exists(checkpointpath)):
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        #print(type(state_dict))
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("pretrain model not found!")
        
    
dataset,collate=get_dataset_collate(args)
train_iterator = DataLoader(dataset, batch_size=10,shuffle=True, num_workers=0,collate_fn=collate)
import pickle as pkl
#points_index_path="D:\maweizhao\MyProgram\DeepLearning\myfile/1\doing/1024vertices\person\smpl_cls_PCMG_pointnet_Transformer\_dataset\data\smpl_1024vertices_fps_index/_1024_point_index.pkl"
#points_index = pkl.load(open(points_index_path, "rb"))
#print(points_index.shape)



smallest_loss=99999

tsne = TSNE(perplexity=3)
all_latent=0
all_color=0
with torch.no_grad():
    for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):

        model.eval()
        output=model(batch)
        latent=batch["bias_latent_z"]
        y=batch["y"]
        if i==0:
            all_color=y
            all_latent=latent
        else:
            all_color=torch.cat([all_color,y],dim=0)
            all_latent=torch.cat([all_latent,latent],dim=0)
        
        
        #break
        
import matplotlib.pyplot as plt

# t-sne参考:https://distill.pub/2016/misread-tsne/
tsne = TSNE(
    perplexity=2.0,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
embedding = tsne.fit(all_latent.cpu().numpy())
colors=all_color.cpu().numpy()


font={'family':'Times New Roman', 'size': 10}
fig, ax = plt.subplots()
scatter=ax.scatter(embedding[:,0],embedding[:,1],c=colors, cmap='Paired')
labels=ax.get_xticklabels()+ax.get_yticklabels()
[labels_temp.set_fontname('Times New Roman') for labels_temp in labels]
ax.tick_params(labelsize=10)

legend_list=[]
for i in range(12):
    legend_color=cm.get_cmap('Paired')(i)
    #print(legend_color)
    temp_legend= ax.scatter([], [], c=[legend_color],
                           label=humanact12_coarse_action_enumerator[i])
    legend_list.append(temp_legend)



legend=ax.legend(handles=legend_list,bbox_to_anchor=(1.00,1.00),loc=0,prop=font)
ax.add_artist(legend)
save_path='./example/temp/latent_space'+str(datetime.datetime.now())+'.png'
save_path=save_path.replace(':','-')
plt.savefig(save_path,format="png",dpi=1200,bbox_inches="tight")





