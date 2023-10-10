import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)

from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import * 
import pickle as pkl



from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color
from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz
from _dataset.src.utils import farthest_point_sample,index_points


def add_options(parser):
    group = parser.add_argument_group('options')
    
    # dataset
    group.add_argument("--dataset", default="uestc",choices=[ "uestc","4Dcompelte", "humanact12","cmu"], help="Dataset to load")
    group.add_argument("--num_frames", default=60,type=int, help="Dataset to load")
    group.add_argument("--num_points", default=1024,type=int, help="Dataset to load")
    group.add_argument("--datapath", default="E:/dataset/datasets/uestc", help="Dataset to load")
    
    group.add_argument("--jointstype", default="vertices",choices=[ "vertices","smpl"], help="Dataset to load")
    group.add_argument("--pose_rep", default="rotvec", choices=["xyz", "rotvec", "rotmat", "rotquat", "rot6d"], help="xyz or rotvec etc")
    group.add_argument("--glob",default=True, dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument("--glob_rot", type=int, nargs="+", default=[0, 0, 0],help="Default rotation, usefull if glob is False")
    group.add_argument("--translation",default=False, dest='translation', action='store_true',help="if we want to output translation")
    group.add_argument('--vertstrans',default=False, dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")


def rot2xyz(args,x,mask=None):
    param2xyz = {"pose_rep": args.pose_rep,
                "glob_rot": args.glob_rot,
                "glob": args.glob,
                "jointstype": args.jointstype,
                "translation": args.translation,
                "vertstrans": args.vertstrans}
    rotation2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    #args.update(param2xyz)
    #kargs.update(kwargs)
    
    return rotation2xyz(x, mask, **param2xyz)
    # return rotation2xyz(x, mask,pose_rep="rot6d",
    #                     translation=True,
    #                     glob=True,
    #                     jointstype="smpl",
    #                     vertstrans=True
    #                     )


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


def main():

    parser = ArgumentParser()
    add_options(parser)
    args = parser.parse_args(args=[])

    with open(os.path.join(args.datapath, 'info', 'num_frames_min.txt'), 'r') as f:
        num_frames_video = np.asarray([int(s) for s in f.read().splitlines()])
        print(len(num_frames_video))
        
    # Out of 118 subjects -> 51 training, 67 in test
    all_subjects = np.arange(1, 119)
    _tr_subjects = [
        1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50,
        52, 54, 55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88,
        90, 91, 93, 96, 99, 102, 103, 104, 107, 108, 112, 113]
    _test_subjects = [s for s in all_subjects if s not in _tr_subjects]
        
    pkldatafilepath = os.path.join(args.datapath, "humanact12poses.pkl")
    data = pkl.load(open(pkldatafilepath, "rb"))   #dict_keys:(['poses', 'oldposes', 'joints3D', 'y'])
    
    zero_pose=torch.zeros((1,24,3,1))
    zero_vertices=rot2xyz(args,zero_pose.cuda())
    zero_vertices=zero_vertices.squeeze(0).permute(2,0,1)  #[1,6890,3]
    point_index=farthest_point_sample(zero_vertices,args.num_points)  #[1,1024]
    
    if args.num_frames != -1:
        point_index=point_index.repeat(args.num_frames,1)
    
    #print(point_index.shape)
    
    
    x_list=list()
    y_list=list()
    cls_list=list()
    mask_list=list()
    lengths_list=list()
    
    #rand_index=np.random.randint(0,len(data['poses']))
    #print(len(data['poses']))
        
    for index,pose in tqdm(enumerate(data['poses']),total=len(data['poses'])):
        #pose=data['poses'][rand_index]
        
        if args.num_frames == -1:
            point_index=point_index.repeat(lengths,1)
        pose=torch.from_numpy(pose)
        pose=pose.clone().detach().float()
        #pose=torch.tensor(pose.clone().detach(),dtype=torch.float) 
        
        lengths,jointnum_mul_3=pose.shape
        pose=pose.reshape(lengths,jointnum_mul_3//3,3)   #[L,njoint,3]
        #pose=pose.permute(1,2,0)
        #print(pose.shape)
        if args.num_frames != -1:
            pose=sample(pose,args.num_frames)  #[num_frames,njoint,3]
            mask=torch.full((1,args.num_frames),True,dtype=bool)[0]
            lengths=torch.tensor(args.num_frames) 
        
        pose=pose.permute(1,2,0)
        pose=pose.unsqueeze(0)  #[1,njoint,3,num_frames]
        # print(pose.shape)
        vertices=rot2xyz(args,pose.cuda())  #[1,6890,3,num_frames]
        #print(vertices.shape)
        vertices=vertices.permute(0,3,1,2).squeeze(0) #[num_frames,6890,3]
        #print(vertices.shape)
        vertices=index_points(vertices,point_index)  #[num_frames,num_frames,3]
        
        # print(vertices.shape)
        # meshplot_visuals_n_seq_color([vertices],["red"])
        # break
        
        cls=0
        x=vertices
        y=data['y'][index]
        
        
        x_list.append(x)
        y_list.append(y)
        cls_list.append(cls)
        mask_list.append(mask)
        lengths_list.append(lengths)
        break
        
    data_dic=dict()
    data_dic["x"]=x_list
    data_dic["y"]=y_list
    data_dic["cls"]=cls_list
    data_dic["mask"]=mask_list
    data_dic["lengths"]=lengths_list
        
        
    # save_name="humanact12_"+str(args.jointstype)+"_"+str(args.num_points)+"points"+str(args.num_frames)+"frames"+".pkl"
    # with open(PROJECT_ROOT_DIR+"/temp/"+save_name, 'wb') as f:
    #     pkl.dump(data_dic, f)
    # print(save_name)
    # print("done!")





if __name__ == '__main__':
    main()
    
    