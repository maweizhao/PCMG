from ast import arg
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
#PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR=BASE_DIR
print(BASE_DIR)
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
    group.add_argument("--dataset", default="humanact12",choices=[ "uestc","4Dcompelte", "humanact12","cmu"], help="Dataset to load")
    group.add_argument("--num_frames", default=60,type=int, help="Dataset to load")
    group.add_argument("--num_points", default=53,type=int, help="Dataset to load")
    group.add_argument("--datapath", default="E:\dataset\datasets\HumanAct12Poses", help="Dataset to load")
    
    group.add_argument("--num_cls", default=1,type=int, help="Dataset to load")
    
    group.add_argument("--mocap_marker", type=bool,default=True, help="Dataset to load")
    group.add_argument("--jointstype", default="vertices",choices=[ "vertices","smpl"], help="Dataset to load")
    group.add_argument("--pose_rep", default="rotvec", choices=["xyz", "rotvec", "rotmat", "rotquat", "rot6d"], help="xyz or rotvec etc")
    group.add_argument("--glob",default=True, dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument("--glob_rot", type=int, nargs="+", default=[0, 0, 0],help="Default rotation, usefull if glob is False")
    group.add_argument("--translation",default=False, dest='translation', action='store_true',help="if we want to output translation")
    group.add_argument('--vertstrans',default=False, dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument("--root_translation",type=bool,default=False,help="if we want to output translation")
    group.add_argument('--pointcloud_in_order',type=bool,default=True, help="Training with vertex translations in the SMPL mesh")

def rot2xyz(args,x,betas,mask=None):
    param2xyz = {"pose_rep": args.pose_rep,
                "glob_rot": args.glob_rot,
                "glob": args.glob,
                "jointstype": args.jointstype,
                "translation": args.translation,
                "vertstrans": args.vertstrans}
    rotation2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    #args.update(param2xyz)
    #kargs.update(kwargs)
    
    return rotation2xyz(x, mask,betas=betas, **param2xyz)
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


    pkldatafilepath = os.path.join(args.datapath, "humanact12poses.pkl")
    data = pkl.load(open(pkldatafilepath, "rb"))   #dict_keys:(['poses', 'oldposes', 'joints3D', 'y'])
    
    # zero_pose=torch.zeros((1,24,3,1))
    # zero_vertices=rot2xyz(args,zero_pose.cuda(),betas)
    # zero_vertices=zero_vertices.squeeze(0).permute(2,0,1)  #[1,6890,3]
    # point_index=farthest_point_sample(zero_vertices,args.num_points)  #[1,1024]
    
    # with open(PROJECT_ROOT_DIR+"/temp/"+"_1024_point_index.pkl", 'wb') as f:
    #     pkl.dump(point_index, f)
    
    data_path=PROJECT_ROOT_DIR+"/_dataset/data/smpl_1024vertices_fps_index/"+"_m_J_regressor_1024_point_index.pkl"
    #open(data_path, "rb")
    if args.mocap_marker:
        from _dataset.src.mocap_marker import PICK_INDEX
        point_index=PICK_INDEX.unsqueeze(0)
    else:
        point_index=pkl.load(open(data_path, "rb"))
    if args.num_frames != -1:
        point_index=point_index.repeat(args.num_frames,1)
        
    m_J_regressor=PROJECT_ROOT_DIR+"/_dataset/data/smpl_1024vertices_fps_index/"+"_m_1024_J_regressor.pkl"
    #open(data_path, "rb")
    m_J_regressor=pkl.load(open(m_J_regressor, "rb"))

    
    #print(point_index.shape)
    
    joint_list=list()
    x_list=list()
    y_list=list()
    cls_list=list()
    mask_list=list()
    lengths_list=list()
    
    #rand_index=np.random.randint(0,len(data['poses']))
    #print(len(data['poses']))
    for cls in range(args.num_cls):
        
        #betas=torch.ones((10)).cuda()
        betas_offset=(args.num_cls-1)/2
        beta=(cls-betas_offset)*2
        betas=torch.ones((10)).cuda()*beta
        #print(betas)
        # betas=torch.tensor([
        #     0.0,  #1
        #     beta,  #2
        #     beta,  #3
        #     beta,  #4
        #     beta,  #5
        #     beta,  #6
        #     beta,  #7
        #     beta,  #8
        #     beta,  #9
        #     beta#  10
        # ]).cuda()
        
        
        #print(cls)
        
        if args.num_frames!=-1:
            betas=betas[None].repeat(args.num_frames,1)

        # import random  
        # random.shuffle(data['poses'],)
        # random.shuffle(data['joints3D'])
        for index,pose in tqdm(enumerate(data['poses']),total=len(data['poses'])):
            #pose=data['poses'][rand_index]
            lengths,jointnum_mul_3=pose.shape
            if args.num_frames == -1:
                point_index=point_index.repeat(lengths,1)
                
            joint=data['joints3D'][index]#[num_frames,njoint,3]
            joint=torch.from_numpy(joint)
            joint=joint.clone().detach().float()

            
            pose=torch.from_numpy(pose)
            pose=pose.clone().detach().float()

            pose=pose.reshape(lengths,jointnum_mul_3//3,3)   #[L,njoint,3]
            if args.num_frames != -1:
                pose=sample(pose,args.num_frames)  #[num_frames,njoint,3]
                joint=sample(joint,args.num_frames).cuda()
                mask=torch.full((1,args.num_frames),True,dtype=bool)[0]
                lengths=torch.tensor(args.num_frames) 
            
            joint=joint-joint[:1,:1,:]   #将第一帧的根节点设置为0
            root_translate=joint[:,:1,:]-joint[:1,:1,:]  #[n_frame,1,3]
            
            pose=pose.permute(1,2,0)
            pose=pose.unsqueeze(0)  #[1,njoint,3,num_frames]
            #print(pose.shape)
            args.jointstype="vertices"
            vertices=rot2xyz(args,pose.cuda(),betas)  #[1,6890,3,num_frames]
            
            
            vertices=vertices.permute(0,3,1,2).squeeze(0) #[num_frames,6890,3]
            if args.pointcloud_in_order !=True :
                point_index=farthest_point_sample(vertices,args.num_points)  #[1,1024]
            #print(vertices.shape)
            vertices=index_points(vertices,point_index)  #[num_frames,1024,3]
            
            if(args.root_translation):
                vertices=vertices+root_translate
            else:
                joint=joint-joint[:,:1,:]
            
            # print(vertices.shape)
            # meshplot_visuals_n_seq_color([-vertices],["red"])
            # from smplx.lbs import vertices2joints
            # joint_1=vertices2joints(m_J_regressor,vertices).cuda()
            # joint_1=joint_1-joint_1[:1,:1,:]
            # print(joint_1[:,:1,:])
            # # print(vertices.shape)
            # # meshplot_visuals_n_seq_color([-vertices],["red"])
            # temp=torch.cat([joint,joint_1],dim=1)
            
            # meshplot_visuals_n_joint_seq_color([-temp],["red"])
            # #meshplot_visuals_n_joint_seq_color([-joint.cuda(),-joint_1,-vertices],["red","blue","gray"])
            # if index==5:
            #     break
            # # temp = torch.cat([joint,vertices],dim=1)
            # # temp_joint=temp[:,:24,:]
            # # temp_vertices=temp[:,24:,:]
            # # print(temp.shape)
            # # print(temp_joint.shape)
            # # print(temp_vertices.shape)
            
            x=vertices
            y=data['y'][index]
            
            joint_list.append(joint)
            x_list.append(x)
            y_list.append(y)
            cls_list.append(cls)
            mask_list.append(mask)
            lengths_list.append(lengths)
            #break
        
    data_dic=dict()
    data_dic["joint"]=joint_list
    data_dic["x"]=x_list
    data_dic["y"]=y_list
    data_dic["cls"]=cls_list
    data_dic["mask"]=mask_list
    data_dic["lengths"]=lengths_list
        
        
    save_name="humanact12_"+str(args.num_cls)+'_'+str(args.jointstype)+"_"+str(args.num_points)+"points"+str(args.num_frames)+"frames"+".pkl"
    with open(PROJECT_ROOT_DIR+"/temp/"+save_name, 'wb') as f:
        pkl.dump(data_dic, f)
    print(save_name)
    print("done!")





if __name__ == '__main__':
    main()
    
    