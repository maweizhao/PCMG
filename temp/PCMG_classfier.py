import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from modules.point_4d_convolution import *
from modules.transformer import *
from _parser.p4_transformer import p4transfomer_parse_args

class P4Transformer_MotionDiscriminator(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Linear(1024, mlp_dim),
            nn.GELU()
        )
        
        self.output_linear =nn.Linear(mlp_dim, num_classes)
        

    def forward(self, input,lengths=None):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        input=input.permute(0,3,1,2)  #[B,N,3,L]->[B,L,N,3]
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        lin1 = self.transformer(embedding)
        lin1 = torch.max(input=lin1, dim=1, keepdim=False, out=None)[0]
        lin1 = self.mlp_head(lin1)
        output=self.output_linear(lin1)

        return output


class P4Transformer_MotionDiscriminatorForFID(P4Transformer_MotionDiscriminator):
    def forward(self, input,lengths=None):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        input=input.permute(0,3,1,2)  #[B,N,3,L]->[B,L,N,3]
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        lin1 = self.transformer(embedding)
        lin1 = torch.max(input=lin1, dim=1, keepdim=False, out=None)[0]
        lin1 = self.mlp_head(lin1)

        return lin1
    
    
    
classifier_model_files = {
    "humanact12": "lib/actionrecognition/humanact12_p4transformer_60.tar",
    "4Dcompelte": "lib/actionrecognition/4Dcomplete_p4transformer_30.tar",
}
    
def load_classifier(dataset_type, input_size_raw, num_classes, device):
    p4_parameters,p4_args=p4transfomer_parse_args()
    model = torch.load(classifier_model_files[dataset_type], map_location=device)
    classifier=P4Transformer_MotionDiscriminator(radius=p4_args.radius, nsamples=p4_args.nsamples, spatial_stride=p4_args.spatial_stride,
                temporal_kernel_size=p4_args.temporal_kernel_size, temporal_stride=p4_args.temporal_stride,
                emb_relu=p4_args.emb_relu,
                dim=p4_args.dim, depth=p4_args.depth, heads=p4_args.heads, dim_head=p4_args.dim_head,
                mlp_dim=p4_args.mlp_dim, num_classes=num_classes).to(device)
    
    if("model" in model.keys()):
        classifier.load_state_dict(model["model"])
    if("net" in model.keys()):
        classifier.load_state_dict(model["net"])
    classifier.eval()
    return classifier


def load_classifier_for_fid(dataset_type, input_size_raw, num_classes, device):
    model = torch.load(classifier_model_files[dataset_type], map_location=device)
    #print(model.keys())
    p4_parameters,p4_args=p4transfomer_parse_args()
    classifier=P4Transformer_MotionDiscriminatorForFID(radius=p4_args.radius, nsamples=p4_args.nsamples, spatial_stride=p4_args.spatial_stride,
                temporal_kernel_size=p4_args.temporal_kernel_size, temporal_stride=p4_args.temporal_stride,
                emb_relu=p4_args.emb_relu,
                dim=p4_args.dim, depth=p4_args.depth, heads=p4_args.heads, dim_head=p4_args.dim_head,
                mlp_dim=p4_args.mlp_dim, num_classes=num_classes).to(device)
    if("model" in model.keys()):
        classifier.load_state_dict(model["model"])
    if("net" in model.keys()):
        classifier.load_state_dict(model["net"])
    classifier.eval()
    return classifier