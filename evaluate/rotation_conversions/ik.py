import numpy as np
import torch
from zmq import device
from lib.hybrik.SMPL import SMPL_layer
import evaluate.rotation_conversions.rotation_conversions as geometry

# B,L,24,3
def joint3dcoordinates_to_rot6d(joint3dcoordinates,translation):
    '''hybrik : convert 3d coordinates joint to rot6d
    
        Parameters
        ----------
        joint3dcoordinates: torch.tensor
            [B,L,24,3]
        translation: bool
        
        Returns
        -------
        rot6d:  torch.tensor,if translation is True [B,L,25,6] else [B,L,24,6].The last row in dim:2 is translation.
        output:
            hybrik output
    '''
    
    
    _device=joint3dcoordinates.device
    B,L,njoint,channel=joint3dcoordinates.shape
    pose_skeleton=joint3dcoordinates.view(B*L,njoint,channel)
    
    transl=pose_skeleton[:, 0, :].unsqueeze(1)   #translation  [B*L,1,3]
    pose_skeleton = pose_skeleton - pose_skeleton[:, 0, :].unsqueeze(1)
    

    smpl_dtype = torch.float32
    h36m_jregressor = np.load('./lib/smpl/J_regressor_h36m.npy')
    smpl = SMPL_layer(
        './lib/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        h36m_jregressor=h36m_jregressor,
        dtype=smpl_dtype,
        num_joints=24
    )
    

    phis=torch.full((B*L,23,2),1.0).to(_device)
    phis=phis*torch.tensor([1.0,0.0]).cuda()
    
    #phis=torch.randn((B*L,23,2)).to(_device)
    #phis=torch.full((B,23,2),-3.1415926)
    betas=torch.full((B*L,10),0.0).to(_device)
    #leaf_thetas=torch.randn((B*L,5,4)).to(_device)
    leaf_thetas=torch.full((B*L,5,4),1.0).to(_device)
    leaf_thetas=leaf_thetas*torch.tensor([1.0,0.0,0.0,0.0]).cuda()
    #leaf_thetas=torch.full((B*L,5,4),-99*3.1415926).to(_device)
    #leaf_thetas[:,[0,3,4],:]=-1

    
    output = smpl.hybrik(
        pose_skeleton=pose_skeleton.type(smpl_dtype),
        betas=betas,
        phis=phis,
        global_orient=None,
        transl=None,
        return_verts=True,
        leaf_thetas=leaf_thetas
    )
        
    rot6d=geometry.matrix_to_rotation_6d(output.rot_mats)  #[B*L,24,6]
    if translation:
        tran_temp=torch.zeros((rot6d.shape[0],1,rot6d.shape[2]),dtype=rot6d.dtype).to(_device)
        #print(tran_temp.shape)
        #print(transl.shape)
        tran_temp[:,:,:3]=transl
        # print(tran_temp.shape)
        # print(rot6d.shape)
        rot6d=torch.cat((rot6d,tran_temp),dim=1)  #[B*L,25,6]
    rot6d=rot6d.view(B,L,rot6d.shape[1],rot6d.shape[2])
    
    return rot6d,output
    
    
    