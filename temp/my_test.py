

import pickle as pkl
import numpy as np
import torch
from lib.hybrik.SMPL import SMPL_layer
from visuals.visuals import meshplot_visuals_n_seq_color,meshplot_visuals_n_joint_seq_color
import evaluate.rotation_conversions.rotation_conversions as geometry
from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz

from evaluate.rotation_conversions.ik import joint3dcoordinates_to_rot6d




smpl_dtype = torch.float32
h36m_jregressor = np.load('./lib/smpl/J_regressor_h36m.npy')
smpl = SMPL_layer(
    './lib/smpl/basicModel_neutral_lbs_10_207_0_v1.1.0.pkl',
    h36m_jregressor=h36m_jregressor,
    dtype=smpl_dtype,
    num_joints=24
)

#print(h36m_jregressor.shape)

B=60


# phis=torch.rand((B,23,2))
# #phis=torch.full((B,23,2),+180.0)
# #phis=torch.full((B,23,2),-3.1415926)
# betas=torch.full((B,10),0.0)
# #leaf_thetas=torch.zeros((B,5,4))
# leaf_thetas=torch.full((B,5,4),-99*3.1415926)
# leaf_thetas[:,:,[0,1]]=0.0


ACTOR_gen_path="D:/maweizhao/MyProgram/DeepLearning/myfile/1/doing/1024vertices/person/smpl_cls_PCMG_pointnet_Transformer/compare/ACTOR_gen.pkl"
# ACTOR_gen_data = pkl.load(open(ACTOR_gen_path, "rb")).unsqueeze(0)
# #print(ACTOR_gen_data.shape)
# #print(ACTOR_gen_data.device)
# ACTOR_gen_data=torch.tensor([1,1,1]).cuda()*ACTOR_gen_data
# rot6d,output=joint3dcoordinates_to_rot6d(ACTOR_gen_data,True)
# #print(rot6d.shape)
# rot6d=rot6d.permute(0,2,3,1)
# #print(rot6d.shape)

# rot2xyz = Rotation2xyz(device="cuda")

# params = {"pose_rep": "rot6d",
#             "translation": True,
#             "glob": True,
#             "jointstype": "smpl",   #vertices,smpl
#             "vertstrans": True,
#             "num_frames": 60,
#             "sampling": "conseq",
#             "sampling_step": 1}

# #mask=torch.full((B),True)

# #rot6d=rot6d.unsqueeze(0).permute(0,2,3,1) #不带平移时[B,24,6,L]
# #print(rot6d.shape)

# #rot6d=torch.rand((1,25,6,60))

# rot6d_2smpl=rot2xyz(rot6d.cuda(),mask=None,**params).permute(0,3,1,2)[0]
# output_vertice=output.vertices


# #rot6d_2smpl=-rot6d_2smpl
# #print(joint_point_est.shape)
# #meshplot_visuals_n_joint_seq([rot6d_2smpl,ACTOR_gen_data[0]])

# #meshplot_visuals_n_seq_color([-rot6d_2smpl,-ACTOR_gen_data[0] ],["red","blue"])
# meshplot_visuals_n_joint_seq_color([-ACTOR_gen_data[0],-rot6d_2smpl,-output_vertice ],["blue","red","red"])

mosh_output_name='throw_001_60frame_output.pkl'
mosh_output_path='./temp/mosh/output/'+mosh_output_name
mosh_output = pkl.load(open(mosh_output_path, "rb"))

pc=mosh_output['points']
faces=mosh_output['faces']
markers=mosh_output['markers']

from visuals.visuals import meshplot_visuals_n_seq_mesh_color,meshplot_visuals_n_seq_pointcloudandmesh_color



#meshplot_visuals_n_seq_mesh_color([-pc],[faces],['red'])
meshplot_visuals_n_seq_pointcloudandmesh_color([-markers],[-pc],[faces],['red'])


