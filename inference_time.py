
from gc import collect
import pickle as pkl
import datetime
from operator import mod
import os
from random import random
from statistics import mode
import sys
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
import time
from matplotlib import cm

# ROOT_DIR = "D:\maweizhao\MyProgram\DeepLearning\myfile\1\PCMG"
# sys.path.append(ROOT_DIR)

humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}

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
    folder=".\evaluate\ACTOR\pretrained_models/humanact12"
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
        
    
# dataset,collate=get_dataset_collate(args)
# train_iterator = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0,collate_fn=collate)
import pickle as pkl
#points_index_path="D:\maweizhao\MyProgram\DeepLearning\myfile/1\doing/1024vertices\person\smpl_cls_PCMG_pointnet_Transformer\_dataset\data\smpl_1024vertices_fps_index/_1024_point_index.pkl"
#points_index = pkl.load(open(points_index_path, "rb"))
#print(points_index.shape)


def evaluate_onebyone():
    batch_size = args.batch_size

    batch_size = 1000

    range = torch.range(0,batch_size-1).tolist()

    #print(range)
    time_list  = []
    test_times = 20
    PCMG_time = time.time()


    batch_size = 128

    model.eval()
    cls=torch.tensor([0],device='cuda:0')
    y=torch.tensor([11],device='cuda:0')
    frames_num=torch.tensor([60],device='cuda:0')

    with torch.no_grad():
        for i,batch in tqdm(enumerate(range), desc="Computing batch",total=len(range)):
            now = time.time()
            #print(y)
            if args.model == "PCMG":
                output=model.generate(y,cls,frames_num,args.points_num)["output"]
            if args.model == "ACTOR":
                gen_points=model.generate(y,frames_num)
                #gen_points=gen_points['output_xyz'][0].permute(2,0,1)
                #print(gen_points.shape)


            if(i > (len(range)-test_times)):
                time_list.append(time.time()-now)
        infer_time = time.time() - PCMG_time


    print(infer_time/batch_size)
    print(torch.tensor(time_list).mean())


def evaluate_batchsize():
    batch_size = args.batch_size

    batch_size = 1000


    #print(range)
    time_list  = []
    test_times = 20
    #PCMG_time = time.time()


    batch_size = 128

    model.eval()
    cls=torch.tensor([0],device='cuda:0').repeat(batch_size)
    y=torch.tensor([11],device='cuda:0').repeat(batch_size)
    frames_num=torch.tensor([60],device='cuda:0').repeat(batch_size)

    #print(cls.shape)
    
    warm_up_times=20

    with torch.no_grad():
        #print(y)
        for i in range(0,warm_up_times):
            if args.model == "PCMG":
                output=model.generate(y,cls,frames_num,args.points_num)["output"]
            if args.model == "ACTOR":
                gen_points=model.generate(y,frames_num)
                #gen_points=gen_points['output_xyz'][0].permute(2,0,1)
                #print(gen_points.shape)

    with torch.no_grad():
        PCMG_time = time.time()
        #print(y)
        if args.model == "PCMG":
            output=model.generate(y,cls,frames_num,args.points_num)["output"]
        if args.model == "ACTOR":
            gen_points=model.generate(y,frames_num)
            #gen_points=gen_points['output_xyz'][0].permute(2,0,1)
            #print(gen_points.shape)
        infer_time = time.time() - PCMG_time


    print(infer_time/batch_size)
    #print(torch.tensor(time_list).mean())

evaluate_batchsize()

#evaluate_onebyone()