import argparse
import os

from _parser.parser import get_parser
from evaluate.ACTOR.models.get_model import JOINTSTYPES



# def add_generation_options(parser):
#     group = parser.add_argument_group('Generation options')
#     group.add_argument("--num_samples_per_action", default=5, type=int, help="num samples per action")
#     group.add_argument("--num_frames", default=60, type=int, help="The number of frames considered (overrided if duration mode is chosen)")
#     #group.add_argument("--fact_latent", default=1, type=int, help="Fact latent")

#     group.add_argument("--jointstype", default="smpl", choices=JOINTSTYPES,
#                     help="Jointstype for training with xyz")

#     group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Add the vertex translations")
#     group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Do not add the vertex translations")
#     group.set_defaults(vertstrans=False)

#     group.add_argument("--mode", default="gen", choices=["interpolate", "gen", "duration", "reconstruction"],
#                     help="The kind of generation considered.")
    
#     #parser.add_argument("--checkpointname",default="D:/maweizhao/MyProgram/DeepLearning/myfile/ACTOR-master/pretrained_models/humanact12/checkpoint_5000.pth.tar")
#     parser.add_argument("--checkpointname",default="D:/maweizhao/MyProgram/DeepLearning/myfile/ACTOR-master/pretrained_models/uestc/checkpoint_1000.pth.tar")
#     group.add_argument("--njoints", default=25, type=int, help="sampling step")
#     group.add_argument("--nfeats", default=6, type=int, help="sampling step")
#     group.add_argument("--num_classes", default=40, type=int, help="number of classes")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeltype",type=str, default="cvae")
    parser.add_argument("--archiname",type=str, default="transformer")
    parser.add_argument("--njoints",type=int, default=25)
    parser.add_argument("--glob_rot", type=int, nargs="+", default=[3.141592653589793, 0, 0],
                       help="Default rotation, usefull if glob is False")
    # parser.add_argument("--lambdas", type=int, nargs="+", default=[3.141592653589793, 0, 0],
    #                 help="Default rotation, usefull if glob is False")
    parser.add_argument('--vertstrans',default=True, dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    parser.add_argument("--jointstype", default="smpl", choices=JOINTSTYPES,
                help="Jointstype for training with xyz")
    
    
    parameters,args=get_parser()
    
    # update lambdas params
    parameters["lambdas"] = {  "kl": 1.0e-05,"rc": 1.0,"rcxyz": 1.0}
    #parameters["check"]
    
    opt = parser.parse_args(args=[])
    newparameters = {key: val for key, val in vars(opt).items() if val is not None}
    
    parameters.update(newparameters)

    #epoch=1000
    #epoch = int(checkpoint.split("_")[1].split('.')[0])
    return args,parameters


