import torch

import evaluate.rotation_conversions.rotation_conversions as geometry
from evaluate.rotation_conversions.smpl import SMPL

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)

SMPL_MODEL_PATH="./lib/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
#SMPL_MODEL_PATH="./lib/smpl/SMPL_NEUTRAL.pkl"
SMPL_MODEL_PATH=os.path.join(PROJECT_ROOT_DIR,SMPL_MODEL_PATH)

JOINTSTYPE_ROOT = {"a2m": 0, # action2motion
                   "smpl": 0,
                   "a2mpl": 0, # set(smpl, a2m)
                   "vibe": 8}  # 0 is the 8 position: OP MidHip below
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]


class Rotation2xyz:
    def __init__(self, device,model_path=SMPL_MODEL_PATH):
        self.device = device
        self.smpl_model = SMPL(model_path=model_path).eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        # print("  ")
        # print("gg,xshape:")
        # print(x.shape)
        
        # print(x.shape)
        # print(x[:, -1, :3].shape)
        # print(x[:, :-1].shape)
        
        # 平移时 x:[B,25,6,L],最后三行的前三个坐标为平移
        if translation:
            
            x_translations = x[:, -1, :3]   #[B,3,L]
            x_rotations = x[:, :-1]         #[B,24,6,L]    
        else:
            x_rotations = x
        # print("x_translations.shape:")
        # print(x_translations.shape)
        #print(x_rotations.shape)

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        # print("x_rotations.shape")
        # print(x_rotations.shape)
        nsamples, time, njoints, feats = x_rotations.shape
        #print(x_rotations.shape)

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device,dtype=torch.float)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],   #[N_frame,10]
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
            
            
        # if betas is None:
        #     betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
        #                     dtype=rotations.dtype, device=rotations.device)
        #     betas[:, 1] = beta
        # betas=torch.rand([rotations.shape[0], self.smpl_model.num_betas],
        #                     dtype=rotations.dtype, device=rotations.device)    
        
        # smpl输出dic,key:vertices,vibe,a2m,smpl,a2mpl
        # print("smpl in")
        # print(rotations.shape)
        # print(global_orient.shape)
        #print(betas.shape)
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        # print("smpl out")
        # print(out)

        # get the desirable joints
        # jointstype为vertices
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        # print("empty")
        #print(x.device)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        # x_xyz的shape:(bs,n_vertices,3,n_frame)，bs为batch_size，n_vertices为顶点个数,3为三维顶点,n_frame为总帧数
        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # print(joints.shape)
        # print(x_xyz.shape)

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            #print(x_translations.shape)
            x_translations = x_translations - x_translations[:, :, [0]]
            # print(x_translations.shape)
            # print(x_translations[:, None, :, :].shape)
            # add the translation to all the joints
            
            x_xyz = x_xyz + x_translations[:, None, :, :]

        return x_xyz
