#测时间
from tqdm import tqdm
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.dataset import SMPLTestPairDataset,SMALTestPairDataset
from model.SkinningNet import SkinningNet
from model.NeuralSkinningPoseTransfer import poseTransfer, ysg_poseTransfer
from model.utils import PoseTransferLoss, getFacesOneRingIdx, getLaplacianMatrix
from model.ysg_utils import ysg_getLaplacianMatrix

from meshplot import plot
import igl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:%s"%device)

batchSize = 1
dataset = SMPLTestPairDataset()
dataIter = DataLoader(dataset, num_workers=0, batch_size=batchSize, shuffle=False, drop_last=True)
allDataNUM = len(dataset)

net = SkinningNet(jointNum=24)
net = net.to(device)
net.load_state_dict(torch.load(".\stateDict\skinningNet_finetune_noise.pkl"))
net = net.eval()


for sV, rV, tV, _ in dataIter:
    sV = sV.to(device).float()
    rV = rV.to(device).float()
    tV = tV.to(device).float()

    F = torch.tensor(dataset.faces).unsqueeze(0).to(device).long().repeat(batchSize, 1, 1)
    facesOneRingIdx = torch.tensor(getFacesOneRingIdx(dataset.faces)).to(device).long().unsqueeze(0).repeat(batchSize, 1, 1)

    with torch.no_grad():
        for _ in tqdm(range(100),desc='warm up'):
            _, _ = net(sV, facesOneRingIdx)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    repeatition = 300 #重复测300次
    timings = np.zeros((repeatition,1))
    timelist=np.zeros((repeatition,1))

    torch.cuda.synchronize()# 等待GPU
    with torch.no_grad():
        for i in tqdm(range(repeatition), desc='test of net'):
            time1=time.time()
            starter.record()
            _, _ = net(sV, facesOneRingIdx)
            _, _ = net(rV, facesOneRingIdx)
            ender.record()
            torch.cuda.synchronize() # 等待GPU
            time2=time.time()
            curr_time = starter.elapsed_time(ender) # 从starter到ender的用时，单位为ms
            timings[i] = curr_time
            timelist[i]=time2-time1
    avg_time = np.mean(timings)
    print('Ours网络平均推理耗时:{}'.format(avg_time))
    print('Ours网络平均推理耗时(time测试):{}'.format(np.mean(timelist)))

    starter2 = torch.cuda.Event(enable_timing=True)
    ender2 = torch.cuda.Event(enable_timing=True)
    repeatition2 = 300 #重复测300次
    timings2 = np.zeros((repeatition2,1))
    torch.cuda.synchronize()
    for i in tqdm(range(repeatition2), desc='test of full method'):
        starter2.record()
        time1=time.time()
        facesOneRingIdx = torch.tensor(getFacesOneRingIdx(dataset.faces)).to(device).long().unsqueeze(0).repeat(batchSize, 1, 1)
        laplacian = ysg_getLaplacianMatrix(sV, F, weight = "cotangent")
        preV, _, _, _, _, _ = ysg_poseTransfer(net, sV, facesOneRingIdx, rV, facesOneRingIdx, laplacian, blendShape = "lbs", dLambda = 20, modelType = "human")
        
        ender2.record()
        torch.cuda.synchronize()
        time2=time.time()
        curr_time = starter2.elapsed_time(ender2)
        timings2[i] = curr_time
        timelist[i]=time2-time1
    avg_time2 = np.mean(timings2)
    print('Ours整个方法姿态迁移耗时:{}'.format(avg_time2))
    print('Ours整个方法姿态迁移耗时(time测试):{}'.format(np.mean(timelist)))


    from torchinfo import summary
    print('用torchinfo统计参数',summary(net, input_data=(sV, facesOneRingIdx, )))
    from thop import profile
    macs, params = profile(net, inputs=(sV, facesOneRingIdx, ))
    from thop import clever_format
    macs, params = clever_format([macs, params], "%.3f")
    print('用thop统计参数macs:',macs)
    print('用thop统计参数params',params)
    print('用numel()统计参数:',sum(p.numel() for p in net.parameters() if p.requires_grad))
    break
