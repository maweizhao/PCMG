import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)

SMPL_DATA_PATH =  os.path.join(PROJECT_ROOT_DIR, "lib/smpl")

# SMPL_DATA_PATH = "D:\maweizhao\MyProgram\DeepLearning\myfile\ACTOR-master\models\smpl/"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
# SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
