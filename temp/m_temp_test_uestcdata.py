from _parser.parser import get_parser
from _dataset.src.dataset import HumanAct12_smpl_VerticesDataset, get_dataset_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

parameters,args  = get_parser()

args.split="train"
#args.pose_rep="xyz"
dataset,collate=get_dataset_collate(args)
print(type(dataset))
from _dataset.src.dataset import smpl_collate
#collate=smpl_collate
print(type(dataset))
#dataset=HumanAct12_smpl_VerticesDataset(args=args,data_path="./_dataset/data/smpl_humanact12_cls_60frames_1024vertices.pkl")
train_iterator = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,collate_fn=collate)

for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):

    from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz
    rot2xyz = Rotation2xyz(device="cuda")
    print(batch["xyz"].shape)
    batch["xyz"] = rot2xyz(x=batch["xyz"].permute(0,2,3,1).cuda(), mask=batch["mask"], pose_rep='rot6d', glob=args.glob,
                            translation=args.translation, jointstype='smpl', vertstrans=args.vertstrans, betas=None,
                            beta=0, glob_rot=args.glob_rot, get_rotations_back=False).permute(0,3,1,2)
    # print(batch["xyz"].shape)
    print(batch["xyz"])
    gen_points = batch["xyz"][5].cuda()
    from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color
    meshplot_visuals_n_seq_color([-gen_points],
                ["red"])
    break


# for motion in train_iterator:
    
#     # print(batch["xyz"].shape)
#     # print(batch["xyz"][0])
#     gen_points = motion["xyz"][5]
#     from visuals.visuals import meshplot_visuals_n_joint_seq_color, meshplot_visuals_n_seq_color
#     meshplot_visuals_n_joint_seq_color([-gen_points],
#                 ["red"])
#     break