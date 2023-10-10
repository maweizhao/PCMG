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
#from utils.furthestpointsample import farthest_point_sample,index_points

# from utils.emd_loss import compute_emd_loss
# from visuals.log import log_out_message

from  evaluate.ACTOR.models.get_model import get_model as get_ACTOR_model
from _parser.parser import get_parser
from _parser.ACTOR_parser import parser as get_ACOTR_parser
from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color

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
        
    
dataset,collate=get_dataset_collate(args)
train_iterator = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=0,collate_fn=collate)
import pickle as pkl
#points_index_path="D:\maweizhao\MyProgram\DeepLearning\myfile/1\doing/1024vertices\person\smpl_cls_PCMG_pointnet_Transformer\_dataset\data\smpl_1024vertices_fps_index/_1024_point_index.pkl"
#points_index = pkl.load(open(points_index_path, "rb"))
#print(points_index.shape)


def pick_1class2seq(train_iterator,y_class,cls=0):
    seq_1=0
    seq_2=0
    for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):
        if batch["y"][0]== y_class:
            if seq_1 == 0:
                #seq_1=tuple([batch["xyz"][0],batch["y"][0],batch["cls"][0]])
                #print(batch)
                batch["cls"][0]=cls
                seq_1=batch
                continue
            if seq_2==0:
                #seq_2=tuple([batch["xyz"][0],batch["y"][0],batch["cls"][0]])
                batch["cls"][0]=cls
                seq_2=batch
                break
    return seq_1,seq_2

def down_sample_seq(seq,down_rate,num_cls,color=None):
    '''
    input:
        seq:tensor[L,N,3]
        down_rate:int
        color:string,default:'red'
    output:
        down_seq[L//down_rate,N,3]
        down_color[L//down_rate,N,3]
    example:
        down_rate=2,down_seq[0,2,4,...]
    '''
    #down_rate=2,[0,2]
    L,N,C=seq.shape
    index=torch.range(0,L-1,down_rate,dtype=torch.long)
    down_seq=seq[index,:,:]
    down_seq=down_seq.unsqueeze(1)
    #color=seq[29,:,1].cpu().numpy()
    import pickle as pkl
    if color==None and num_cls==1:
        data_path="./_dataset/data/_1024vertices_color/point_color_cls1.pkl"
        color=pkl.load(open(data_path, "rb"))[0]
    elif color==None and num_cls==3:
        #color=seq[28,:,1].cpu().numpy()
        
        data_path="./_dataset/data/_1024vertices_color/point_color_cls3.pkl"
        color=pkl.load(open(data_path, "rb"))
        
        
        #print(color)
        # with open(data_path, 'wb') as f:
        #     pkl.dump(color, f)
    #color=seq[28,:,1].cpu().numpy()
    #color=seq[0].cpu().numpy()*10
    #color=np.ones((1024,3))*np.array([72.0/255.0,25.0/255.0,40.0/255.0])
    #color=np.ones((1024,3))*np.array([0.0,1.0,0.5])
    #color[3]=np.array([1.0,0.0,0.0])
    #x = np.linspace(0.0, 1.0, 100)
    #print(color)
    color = (color - np.min(color)) / (np.max(color) - np.min(color))
    color=cm.get_cmap('viridis')(color)[:, :3]
    #color=cm.get_cmap('gist_rainbow')(color)[:, :3]
    # 3cls: 2：左手手指,46:腋下,7：左手手肘,19：左手小臂中
    #color[2]=np.array([1.0,0.0,0.0])
    # 1cls:2：右手手指，17：右手小臂中
    #color[17]=np.array([1.0,0.0,0.0])
    down_color=[color]*(L//down_rate)

    #rgb=cm.get_cmap('viridis')(color)
    

    # print(down_seq.shape)
    # print(down_color)
    return down_seq,down_color




smallest_loss=99999

with torch.no_grad():
    for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):
        
        gt=batch["xyz"][0].cuda()
        # meshplot_visuals_n_joint_seq_color([gt],["red"],"smpl")

        # print(batch["lengths"][0])
        # print(batch["mask"][0].shape)
        # print(batch["mask"][0])
        # gt=batch["xyz"][0].cuda()
        # meshplot_visuals_n_joint_seq([gt])
        # break
        
        # gt=batch["xyz"][0].cuda()
        # if(i==5):
        #     break
        
        # ------------------interpolate------------------
        # y=torch.tensor([3],device='cuda:0')
        # cls=torch.tensor([0],device='cuda:0')
        # y_class=y[0].item()
        # seq_1,seq_2=pick_1class2seq(train_iterator,y_class,cls)
        # print(seq_1["xyz"].shape)
        
        # y_2=torch.tensor([3],device='cuda:0')
        # y_class=y_2[0].item()
        # seq_1_temp,seq_2_temp=pick_1class2seq(train_iterator,y_class,cls)
        
        # model.eval()
        # output=model.interpolate_latent_generate(seq_1,seq_2)
        # output=model.interpolate_latent_generate(seq_1,seq_1_temp)
        
        # ------------------interpolate------------------
        
        #print(seq_1['xyz'].shape)
        #meshplot_visuals_n_seq_color( [-seq_1['xyz'][0],output[1],-seq_2['xyz'][0],],['red','red','red'])
        #meshplot_visuals_n_seq_color( [output[0],output[1],output[2]],['red','red','red'])
        #break
        
        
        # #------------reconstruct-----------------
        # model.eval()
        # output=model(batch)
        # gt=batch["xyz"][0].cuda()
        # points=output[0].cuda()
        

        
        # import pickle as pkl
        # save_name="D:\maweizhao\MyProgram\DeepLearning\myfile\PCMG\compare/PCMG_gen.pkl"
        # with open(save_name, 'wb') as f:
        #     pkl.dump(gt, f)
        #     print("save at: "+ save_name)
        # print(gt.shape)
        
        # print(batch["cls"][0])
        

        #meshplot_visuals_n_joint_seq_color([-gt,-points],["red","blue"])
        #meshplot_visuals_n_seq_color([-gt,-points],["red","blue"])
        #break
        # #if(i==5):
        # break
        # #------------reconstruct-----------------

        # # -------------- generate------------------
        model.eval()
        y=batch["y"].cuda()
        cls=torch.tensor([2],device='cuda:0')
        y=torch.tensor([11],device='cuda:0')
        frames_num=torch.tensor([60],device='cuda:0')
        print(y)
        args.model='PCMG'
        if args.model == "PCMG":
            output=model.generate(y,cls,frames_num,args.points_num)["output"]
            
        #print(output.shape)
        # # -------------- generate------------------

        # # -------------- mocap mosh++------------------
        # mosh_output_name='throw_5_60frame_output.pkl'
        # mosh_output_path='./temp/mosh/output/select/'+mosh_output_name
        # mosh_output = pkl.load(open(mosh_output_path, "rb"))

        # pc=mosh_output['points'].cuda()
        # faces=mosh_output['faces'].cuda()
        # markers=mosh_output['markers'].cuda()

        # output=pc.unsqueeze(0)
        # from visuals.visuals import meshplot_visuals_n_seq_mesh_color,meshplot_visuals_n_seq_pointcloudandmesh_color

        # #meshplot_visuals_n_seq_mesh_color([-pc],[faces],['red'])
        # meshplot_visuals_n_seq_pointcloudandmesh_color([-markers],[-pc],[faces],['red'])
 
        # # -------------- mocap mosh++------------------
 


        # if args.model == "ACTOR":
        #     gen_points=model.generate(y,frames_num)
        #     gen_points=gen_points['output_xyz'][0].permute(2,0,1)
        #     points_index
        #     from _dataset.src.utils import farthest_point_sample,index_points
        #     point_index=points_index.repeat(args.num_frames,1)
            
        #     gen_points=index_points(gen_points,point_index)  #[num_frames,num_frames,3]
            
        #     print(gen_points.shape)
            
        # meshplot_visuals_n_seq_color([-gen_points],
        #                 ["red"])
        
        #-----------------------------
        
        for output_index in range(len(output)) :
            gen_points=output[output_index]
            ori_gen_point=gen_points
            # # ----------save as html---------------
            gen_points,color_list=down_sample_seq(gen_points,4,args.num_animal_classes)
            #markers,color_list=down_sample_seq(markers,4,args.num_animal_classes)
            #print(markers.shape)
            if args.point_decoder=='Transformer_pointdecoder':
                #save_path='./example/base/'+humanact12_coarse_action_enumerator[y[0].item()]+'_'+str(datetime.datetime.now())
                save_path='./example/temp/'+str(cls[0].item())+'_cls_'+humanact12_coarse_action_enumerator[y[0].item()]+'_'+str(datetime.datetime.now())
                #save_path='./example/temp/'+str(output_index)+'_'+humanact12_coarse_action_enumerator[y[0].item()]+'_'+str(datetime.datetime.now())
            else:
                save_path='./example/compare/'+humanact12_coarse_action_enumerator[y[0].item()]+str(datetime.datetime.now())
            save_path=save_path.replace(':','-')
            #color_list= gen_points[0][0].cpu().detach().numpy()[:, 1]*15
            #faces=[faces]*15
            #meshplot_visuals_n_seq_color(-gen_points,color_list,save_path)
            #meshplot_visuals_n_seq_mesh_color(-gen_points,faces,color_list,save_path)
            #meshplot_visuals_n_seq_pointcloudandmesh_color(-gen_points,faces,color_list)
            #meshplot_visuals_n_seq_pointcloudandmesh_color(-markers,-gen_points,faces,color_list,save_path)
            # # ----------save as html---------------
            
            #gen_points=gen_points.permute(1,0,2,3)[0]
            #color_list=["red"]*15
            #meshplot_visuals_n_joint_seq_color(-gen_points,color_list,"smpl",save_path)
            #meshplot_visuals_n_seq_color([-output[0]],[color_list[0]])

        
        # # ----------save as c3d---------------
        # from _dataset.src.process_c3d import write_mocap_c3d
        # print(str(datetime.datetime.now()))
        # #output=gt*torch.tensor([-1,-1,-1]).cuda()
        # points=output[0].cpu().numpy()*1000
        # #print(points.shape)
        # c3d_path="./temp/c3d/temp/"+humanact12_coarse_action_enumerator[y[0].item()]+"_"+str(datetime.datetime.now())+"_60frame.c3d"
        # c3d_path=c3d_path.replace(':','-')
        # write_mocap_c3d(points,c3d_path,frame_rate=10)
        
        meshplot_visuals_n_seq_color([-output[0]],[color_list[0]])
        
        # # ----------save as c3d---------------
        
        #mosh_points_path="./temp/output.pkl"
        #mosh_points = pkl.load(open(mosh_points_path, "rb"))
        #meshplot_visuals_n_seq_color([-mosh_points],['red'])
        
        # length_hand=output[1,:,2,:]-output[1,:,17,:]
        # #print(length_hand)
        # length_hand=length_hand*length_hand
        # #print(length_hand.shape)
        # length_hand=torch.sum(length_hand,dim=1)
        # print(length_hand)


        # length_hand=seq_1["xyz"][0]
        # length_hand=length_hand[:,2,:]-length_hand[:,7,:]
        # #print(length_hand)
        # length_hand=length_hand*length_hand
        # length_hand=torch.sum(length_hand,dim=1)
        # print(length_hand)
        
        # length_hand=seq_2["xyz"][0]
        # length_hand=length_hand[:,2,:]-length_hand[:,7,:]
        # #print(length_hand)
        # length_hand=length_hand*length_hand
        # length_hand=torch.sum(length_hand,dim=1)
        # print(length_hand)
        
        
        # meshplot_visuals_n_seq_color([-seq_1["xyz"][0],-output[1],-seq_2["xyz"][0]],[color_list[0],color_list[0],color_list[0]])
        #meshplot_visuals_n_seq_color([-output[0],-output[1],-output[2]],[color_list[0],color_list[0],color_list[0]])
        #meshplot_visuals_n_seq_color([-output[0],-output[1],-output[2]],[color_list[0],color_list[0],color_list[0]])
        #meshplot_visuals_n_joint_seq_color([-output[0]],['red'])
        if i==0:
            break
        
  
            

        # import pickle as pkl
        # save_name="D:/maweizhao/MyProgram/DeepLearning/myfile/1/doing/1024vertices/person/smpl_cls_PCMG_pointnet_Transformer/compare/ACTOR_gen.pkl"
        # with open(save_name, 'wb') as f:
        #     pkl.dump(gen_points, f)
        #     print("save at: "+ save_name)


