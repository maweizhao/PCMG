import torch
from model.PCMG import PCMG
from _parser.parser import get_parser
import os

_,args=get_parser()
print(args)

model=PCMG(args).cuda()
ontraining_model_path="./check_point/done/modelpara.pth"
#ontraining_model_path="./check_point/done/seq4.pth"
if(os.path.exists(ontraining_model_path)):
    model_dict=torch.load(ontraining_model_path)
    #begin_epoch=model_dict["epoch"]
    model.load_state_dict(model_dict["net"])

model.eval()
cls=torch.tensor([0],device='cuda:0')
y=torch.tensor([4],device='cuda:0')
frames_num=torch.tensor([60],device='cuda:0')
print(y)
args.model='PCMG'
if args.model == "PCMG":
    output=model.generate(y,cls,frames_num,args.points_num)["output"]


# use meshplot to visual
from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color
meshplot_visuals_n_seq_color([-output[0]],["red"])