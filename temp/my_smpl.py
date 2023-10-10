import torch
import os
import igl
from evaluate.rotation_conversions.smpl import SMPL
import pickle as pkl

SMPL_MODEL_PATH="./lib/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

smpl_model = SMPL(model_path=SMPL_MODEL_PATH).eval()
pose=zero_pose=torch.zeros((1,23,3,3))
global_orient=torch.zeros(1,1,3,3)
betas=torch.zeros((1,10))

out = smpl_model(body_pose=pose, global_orient=global_orient, betas=betas)
print(out)

#ret = igl.write_triangle_mesh(os.path.join(root_folder, "data", "bunny_out.obj"), v, f)

smpl_data = pkl.load(open(SMPL_MODEL_PATH, "rb"))

