import argparse
import os

from _parser.parser import get_parser

def parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpointname",type=str,
    #                     default="D:/usersdata/maweizhao/Myprogram/release/PCMG/check_point\ontraining\\"+
    #                     "done_uestc_60_l2_vertice_smpl_uestc_2023-05-31 14-15-10.055990"
    #                     +"/modelpara.pth")
    parser.add_argument("--checkpointname",type=str,
                        default="D:/usersdata/maweizhao/Myprogram/release/PCMG/evaluate\ACTOR/pretrained_models/uestc\\"+
                        ""
                        +"/checkpoint_1000.pth.tar")
    #parser.add_argument("--checkpointname",type=str, default="D:\maweizhao\MyProgram\DeepLearning\myfile/1\doing/1024vertices\person\smpl_cls_PCMG_pointnet_Transformer\evaluate\ACTOR\pretrained_models/humanact12/checkpoint_5000.pth.tar")
    parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
    parser.add_argument("--num_frames", default=60, type=int, help="number of frames or -1")
    parser.add_argument("--niter", default=20, type=int, help="number of iterations")
    parser.add_argument("--num_seq_max", default=3000, type=int, help="number of sequences maximum to load or -1")

    
    parameters,args=get_parser()    
    opt = parser.parse_args(args=[])
    newparameters = {key: val for key, val in vars(opt).items() if val is not None}
    
    folder, checkpoint = os.path.split(newparameters["checkpointname"])
    parameters.update(newparameters)

    #epoch=1000
    #epoch = int(checkpoint.split("_")[1].split('.')[0])
    return args,parameters, folder, checkpoint, opt.niter


