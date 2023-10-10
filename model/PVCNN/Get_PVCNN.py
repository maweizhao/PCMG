import functools
import torch.nn as nn
import torch

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)

import  model.PVCNN.modules.functional as F
from model.PVCNN.modules.voxelization import Voxelization
from model.PVCNN.modules.shared_mlp import SharedMLP
from model.PVCNN.modules.se import SE3d
# from models.utils import create_mlp_components, create_pointnet_components

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      with_se=with_se, normalize=normalize, eps=eps)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels

def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))

def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])

class PVCNN(nn.Module):
    

    def __init__(self,latent_dim = 256, extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.latent_dim = latent_dim
        self.blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point, out_channels=[256, 128],
            classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        self.outcov = torch.nn.Linear((int)(128*width_multiplier), self.latent_dim)
        #self.outcov = torch.nn.Conv1d(184, self.latent_dim,1)
    # inputs["xyz"]:[B,L,N,3]
    def forward(self, batch):  # [B,n_channel,n_point]

        B,L,N,channel=batch["xyz"].shape
        inputs = batch["xyz"].cuda().contiguous()

        inputs=inputs.view(B*L,N,channel)  #[B*L,N,3]
        
        inputs=inputs.permute(0,2,1) #[B*L,3,N]

        #print(inputs.shape)

        coords = inputs[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            inputs, _ = self.point_features[i]((inputs, coords))
            out_features_list.append(inputs)

        inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
        #print(inputs.shape)
        point_encoder_z = self.outcov(inputs)

        # out_features_list.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        # outputs = torch.cat(out_features_list, dim=1)
        # print(outputs.shape)
        # outputs = self.outcov(outputs)
        # point_encoder_z = outputs.max(dim=-1, keepdim=False).values #[B*L,256]


        point_encoder_z = point_encoder_z.view(B,L, self.latent_dim)     #[B*L,latent_dim]
        # print(inputs.max(dim=-1, keepdim=False).values.shape)
        # # inputs: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
        # outputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)   # num_batches * 256

        return {"point_encoder_z":point_encoder_z}


def get_PVCNN(args):
    return PVCNN(latent_dim=args.latent_dim ,width_multiplier=args.PVCNN_width_multiplier).cuda()


def testPVCNN():
    point_features = PVCNN(width_multiplier=0.125).cuda()
    batch = dict()
    batch["xyz"] = torch.rand(20,30,24,3)
    batch.update(point_features(batch)) 
    #print(batch["point_encoder_z"].shape)


#testPVCNN()
