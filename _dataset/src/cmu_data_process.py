from argparse import ArgumentParser
import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import * 
import pickle as pkl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)



def add_options(parser):
    group = parser.add_argument_group('options')
    
    # dataset
    group.add_argument("--dataset", default="cmu",choices=[ "uestc","4Dcompelte", "humanact12","cmu"], help="Dataset to load")
    group.add_argument("--jointstype", default="smpl",choices=[ "vertices","smpl"], help="Dataset to load")
    group.add_argument("--pose_rep", default="xyz", choices=["xyz", "rotvec", "rotmat", "rotquat", "rot6d"], help="xyz or rotvec etc")
    group.add_argument("--num_frames", default=100,type=int, help="Dataset to load")
    group.add_argument("--datapath", default="E:\dataset\datasets\CMU Mocap\mocap", help="Dataset to load")
    group.add_argument("--save_path", default="./temp/", help="Dataset to load")


parser = ArgumentParser()
add_options(parser)
args = parser.parse_args(args=[])



def sample(xyz,sample_num_frames):
    '''
        xyz:torch.tensor(n_frame,njoint,channel)
        sample_num_frames:int
    '''
    n_frame,njont,channel=xyz.shape
    output=torch.zeros(size=(sample_num_frames,njont,channel))
    if(n_frame>=sample_num_frames):
        output[0:sample_num_frames,:,:]=xyz[0:sample_num_frames,:,:]
    elif(n_frame<sample_num_frames):
        end_frame=xyz[n_frame-1].unsqueeze(0)   #[1,njoint,channel]
        cat_num=sample_num_frames-n_frame
        end_frame=end_frame.repeat(cat_num,1,1)
        output=torch.cat([xyz[:n_frame,:,:],end_frame],dim=0)
        
    return output




data_root_path=args.datapath
vertices_data_path=os.path.join(data_root_path,"mocap_3djoints")
#data_path="E:\dataset\datasets\CMU Mocap\mocap\mocap_3djoints"
cls_path=os.path.join(data_root_path,"pose_clip.csv")
#files=os.listdir(data_path)
#cls_data=csv.reader(cls_path)
#cls_data=list(cls_data)

motion_data=pd.read_csv (cls_path)
motion_data=motion_data.to_dict(orient='list')
motion_dict = {}
for i,motion in enumerate(motion_data['motion']):
    motion_dict[motion]=motion_data['action_type'][i]
    
#print(cls_dict['13_20'])
action_num_cls_dict = {}
action=0

x_list=list()
y_list=list()
cls_list=list()
mask_list=list()
lengths_list=list()

for index,motion_ind in tqdm(enumerate(motion_dict),total=len(motion_dict)):
    vertices_path=os.path.join(vertices_data_path,motion_ind+'.npy')
    vertices_seq=np.load(vertices_path)
    vertices_seq=torch.from_numpy(vertices_seq).cuda()
    
    action_type=motion_dict[motion_ind]
    if action_type not in action_num_cls_dict.keys():
        action_num_cls_dict[action_type]=action
        action+=1   #action:pre define number 
        
        

    if(args.num_frames!=-1):
        mask=torch.full((1,args.num_frames),True,dtype=bool)[0]
        lengths=torch.tensor(args.num_frames) 
        vertices_seq=sample(vertices_seq,args.num_frames)
        # print(mask)
        # print(lengths)
    else:
        #mask=batch["mask"][0]
        lengths=torch.tensor(vertices_seq.shape[0])
        mask=torch.full((1,lengths),True,dtype=bool)[0]
        
    #x=vertices_seq.type(torch.FloatTensor)*0.05   #[L,N,3]
    x=vertices_seq.type(torch.FloatTensor)/20   #[L,N,3]
    #print(x)
    # from visuals.visuals import meshplot_visuals_n_joint_seq_color
    # meshplot_visuals_n_joint_seq_color([x.cuda()],["red"],"cmu")
    # break
    
    # translation the root joint in the frist frame to point(0,0,0)
    translation=x[:1,:1,:]   #[1,1,3]
    x=x-translation
    
    # from visuals.visuals import meshplot_visuals_n_joint_seq_color
    # meshplot_visuals_n_joint_seq_color([x.cuda()],["red"],"cmu")
    # break
    
    # print(x.shape)
    # print(x)
        
    y=action_num_cls_dict[action_type]
    # print(action_type)
    # print(y)
    # print(motion_ind)
    cls=0 
    
    x_list.append(x)
    y_list.append(y)
    cls_list.append(cls)
    mask_list.append(mask)
    lengths_list.append(lengths)
    
    
data_dic=dict()
data_dic["x"]=x_list
data_dic["y"]=y_list
data_dic["cls"]=cls_list
data_dic["mask"]=mask_list
data_dic["lengths"]=lengths_list
    
save_name="cmu_"+str(args.num_frames)+"frames"+".pkl"
save_path=os.path.join(PROJECT_ROOT_DIR,args.save_path)
print(save_path)
with open(save_path+save_name, 'wb') as f:
    pkl.dump(data_dic, f)
print(save_name)
print("done!")

print(action_num_cls_dict)