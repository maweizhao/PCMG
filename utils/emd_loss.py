import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

#print(ROOT_DIR)

from utils.emd.emd import earth_mover_distance


def compute_emd_loss(gt_seq,pred_seq):
    """
    Calculate Density-aware Chamfer Distance between two point sets
    :param gt_seq: size[B,L, N, C]
    :param pred_seq: size[B,L, M, C]
    :return: sum of Density-aware Chamfer Distance of two point sets
    """
    B,L,N,C=gt_seq.shape
    _,_,M,_=pred_seq.shape
    gt_seq=gt_seq.permute(1,0,2,3)      #[L，B，N，C]
    pred_seq=pred_seq.permute(1,0,2,3)  #[L，B，M，C]
    
    gt_seq=gt_seq.reshape(L*B,N,C)
    pred_seq=pred_seq.reshape(L*B,M,C)
    emd_loss= earth_mover_distance(pred_seq,gt_seq, transpose=False)
    #emd_loss=emd_loss/(L*B*N)
    emd_loss=torch.mean(emd_loss)
    #print(emd_loss)
    
    return emd_loss

# def test():
#     gt_seq=torch.rand(5,2,512,3)
#     pred_seq=torch.rand(5,2,512,3)
#     a=compute_emd_loss(gt_seq.cuda(),pred_seq.cuda())
#     print(a)
    
# test()