import torch
import numpy as np
#import robust_laplacian
import torch.nn.functional as F
from tqdm import *

def laplace_vec(batch_xyz,k):
    '''
    input:
        batch_xyz:[B,N,C,L]
        k:int
    output:
        laplace:[B,N,C,L]
    '''
    B,N,C,L=batch_xyz.shape
    device=batch_xyz.device
    #print(device)
    batch_xyz=batch_xyz.permute(0,3,1,2)  #[B,L,N,3]
    #print(batch_xyz[0][0][0])
    from utils._utils import batch_pairwise_dist,knn
    batch_xyz=batch_xyz.reshape(B*L,N,C)  #[B*L,N,3]
    batch_xyz=batch_xyz.permute(0,2,1)
    idx=knn(batch_xyz,k+1).to(device)
    #print(idx.device)
    idx_base = torch.arange(0, B*L, device=device).view(-1, 1, 1)*N
    idx = idx + idx_base
    idx = idx.view(-1)
    #print(idx.shape)
    batch_xyz = batch_xyz.transpose(2, 1).contiguous()   # (batch_size,num_dims, num_points )  -> (batch_size,num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    #print(batch_xyz.shape)
    knn_points = batch_xyz.view(B*L*N, -1)[idx, :]
    knn_points = knn_points.view(B*L, N, k+1, C)   #[B*L,N,k+1,3]
    laplace=torch.zeros(B*L,N,C).to(device)
    for i in range(k):
        laplace+=knn_points[:,:,i+1,:]
        #print(knn_points[:,:,i+1,:].shape)
    laplace=knn_points[:,:,0,:]-laplace/k
    laplace=laplace.view(B,L,N,C)
    laplace=laplace.permute(0,2,3,1)
    #print(laplace.shape)
    
    #print(batch_xyz[0][0][0])
    #print(knn_points[0][0][0])
    #val_1
    
    
    return laplace




def pointcloud_seq_laplace_loss(motion_loader,device):
    with torch.no_grad():
    #print(len(motion_loader))
        all_loss=0
        #batch_num=0
        for batch in tqdm(motion_loader,total=motion_loader.len()):
            #batch_num+=1
            # print("gggg")
            batch_xyz=batch["output_xyz"].cuda()
            B,N,C,L=batch_xyz.shape
            laplace=laplace_vec(batch_xyz,5)
            laplace=laplace.permute(3,0,1,2)
            
            laplace_m_offset=torch.cat([laplace[:1,:,:,:],laplace[:L-1,:,:,:]],dim=0)
            loss=F.mse_loss(laplace,laplace_m_offset)
            all_loss+=loss
            
            # batch_xyz=batch_xyz.permute(0,3,1,2)  #[B,L,N,3]
            
            # for point_seq in batch_xyz:
            #     laplace_m=np.zeros((L,N,C))
            #     #print(laplace_m.shape)
            #     for i in range(0,L):
            #         points_i=point_seq[i].cpu().numpy()
            #         #print(points_i.shape)
            #         L_i,_=  robust_laplacian.point_cloud_laplacian(points_i,n_neighbors=5)
            #         laplace_m[i]=L_i*points_i
            #     laplace_m=torch.tensor(laplace_m)
            #     laplace_m_offset=torch.cat([laplace_m[:1,:,:],laplace_m[:L-1,:,:]],dim=0)
            #     loss=F.mse_loss(laplace_m,laplace_m_offset)
            #     all_loss+=loss
    #print(batch_num)
        
        
    return all_loss.item()/motion_loader.len()