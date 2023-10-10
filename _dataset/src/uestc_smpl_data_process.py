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
    group.add_argument("--num_points", default=24,type=int, help="Dataset to load")
    group.add_argument("--datapath", default="C:/usersdata/maweizhao/dataset/uestc", help="Dataset to load")
    
    group.add_argument("--jointstype", default="smpl",choices=[ "vertices","smpl"], help="Dataset to load")
    group.add_argument("--pose_rep", default="rot6d", choices=["xyz", "rotvec", "rotmat", "rotquat", "rot6d"], help="xyz or rotvec etc")
    group.add_argument("--glob",default=True, dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument("--glob_rot", type=int, nargs="+", default=None,help="Default rotation, usefull if glob is False")
    group.add_argument("--translation",default=True, dest='translation', action='store_true',help="if we want to output translation")
    group.add_argument('--vertstrans',default=True, dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")

# # x:[B,25,6,L]
# def rot2xyz(args,x,mask=None):
#     param2xyz = {"pose_rep": args.pose_rep,
#                 "glob_rot": args.glob_rot,
#                 "glob": args.glob,
#                 "jointstype": args.jointstype,
#                 "translation": args.translation,
#                 "vertstrans": args.vertstrans}
#     rotation2xyz = Rotation2xyz(device=torch.device("cuda:0"))
#     #args.update(param2xyz)
#     #kargs.update(kwargs)
    
#     return rotation2xyz(x, mask, **param2xyz)
#     # return rotation2xyz(x, mask,pose_rep="rot6d",
#     #                     translation=True,
#     #                     glob=True,
#     #                     jointstype="smpl",
#     #                     vertstrans=True
#     #                     )


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

    from _dataset.src.uestc.uestc import UESTC

    
    split = "train"
    dataset = UESTC(num_frames=args.num_frames,split=split,pose_rep=args.pose_rep)

    #dataset.reset_shuffle()
    #print(point_index.shape)

    x_list=list()
    y_list=list()
    cls_list=list()
    mask_list=list()
    lengths_list=list()
    
    #rand_index=np.random.randint(0,len(data['poses']))
    #print(len(data['poses']))
    #randomIndex = torch.randint(0,len(dataset),1).item()

    from _dataset.src.uestc.rotation2xyz import Rotation2xyz
    rot2xyz = Rotation2xyz(device="cuda")


    if args.num_frames != -1:
        mask=torch.full((1,args.num_frames),True,dtype=bool)
        lengths=torch.tensor(args.num_frames) 
        cls = 0

    for index,data in tqdm(enumerate(dataset),total=len(dataset)):
        
        print(data['inp'].shape)
        
        xyz = rot2xyz(x=data['inp'].unsqueeze(0).cuda(), mask=mask, pose_rep='rot6d', glob=True,
                                    translation=True, jointstype='smpl', vertstrans=True, betas=None,
                                    beta=0, glob_rot=None, get_rotations_back=False)


        #xyz = rot2xyz(args,data['inp'].unsqueeze(0).cuda())  # [B,25,6,L] -> [B,24,3,L]
        #xyz  = data['inp'].unsqueeze(0).cuda()
        xyz = xyz.permute(0,3,1,2)  #[B,L,N,3]
        y = data["action"]
        print(xyz[0])
        meshplot_visuals_n_joint_seq_color([-xyz[0]],["red"])

        #print(xyz.shape)

        
        x_list.append(xyz)
        y_list.append(y)
        cls_list.append(cls)
        mask_list.append(mask[0])
        lengths_list.append(lengths)
        break

        
    data_dic=dict()
    data_dic["x"]=x_list
    data_dic["y"]=y_list
    data_dic["cls"]=cls_list
    data_dic["mask"]=mask_list
    data_dic["lengths"]=lengths_list
        
        
    # save_name=split+"_" + args.dataset + "_"+str(args.jointstype)+"_"+str(args.num_points)+"points"+str(args.num_frames)+"frames"+".pkl"
    # with open(PROJECT_ROOT_DIR+"/temp/"+save_name, 'wb') as f:
    #     pkl.dump(data_dic, f)
    # print(save_name)
    # print("done!")





if __name__ == '__main__':
    main()
    
    