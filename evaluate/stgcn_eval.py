import torch
from tqdm import tqdm

from evaluate.tools import fixseed

from evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
# from src.evaluate.othermetrics.evaluation import Evaluation

from torch.utils.data import DataLoader

from _dataset.src.dataset import get_dataset_collate

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from .tools import save_metrics, format_metrics

# from src.models.get_model import get_model as get_gen_model
# from src.datasets.get_dataset import get_datasets
import evaluate.rotation_conversions.rotation_conversions as geometry

from model.PCMG import PCMG
from evaluate.rotation_conversions.ik import joint3dcoordinates_to_rot6d

def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self,args, mode, model, parameters, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]

        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    #classes = databatch["y"]
                    #gendurations = databatch["lengths"]
                    #batch = model.generate(classes, gendurations)
                    
                    classes = databatch["y"].cuda()
                    gendurations = databatch["lengths"]
                    cls=torch.tensor([0],device='cuda:0')
                    
                    #y=torch.tensor([5],device='cuda:0')
                    batch = model.generate(classes,cls,gendurations,args.generate_num_vertices)  # batch["output_xyz"]  : [B,L,24,3]
                    #print(batch["output_xyz"].device)
                    batch["output"],_=joint3dcoordinates_to_rot6d(batch["output"],args.translation)  #batch["output"] : [B,L,25,6]
                    batch["output"]=batch["output"].permute(0,2,3,1) #batch["output"] : [B,25,6,L]
                    feats = "output"
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}   # batch["xyz"]  : [B,L,25,6]
                    #print(batch["xyz"].shape)
                    if args.pose_rep!="rot6d":
                        batch["xyz"],_=joint3dcoordinates_to_rot6d(batch["xyz"],args.translation)  #[B,L,25,6]
                    batch["xyz"]=batch["xyz"].permute(0,2,3,1)  # [B,25,6,L]
                    #print(batch["xyz"].shape)
                    feats = "xyz"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output_xyz"

                batch = {key: val.to(device) for key, val in batch.items()}

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]

                # x = x.permute(0, 3, 1, 2)
                # x = convert_x_to_rot6d(x, pose_rep)
                # x = x.permute(0, 2, 3, 1)

                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(args,parameters, folder, checkpointname, niter):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = parameters["batch_size"]
    doing_recons = False

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    # get_datasets(parameters)
    # faster: hardcode value for uestc

    parameters["num_classes"] = 40
    parameters["nfeats"] = 6
    parameters["njoints"] = 25

    model=PCMG(args).cuda()

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    print(checkpointpath)
    if(os.path.exists(checkpointpath)):
        model_dict=torch.load(checkpointpath)
        epoch=model_dict["epoch"]
        model.load_state_dict(model_dict["net"])
    else:
        raise NotImplementedError("pretrain model not found!")
    
    model.eval()

    model.outputxyz = False

    recogparameters = parameters.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

    stgcn_metrics = {}
    # joints_metrics = {}
    # pose_metrics = {}

    
    args.split="train"
    #args.pose_rep="xyz"
    #args.pose_rep="rot6d"
    train_dataset,collate=get_dataset_collate(args)
    args.split="test"
    test_dataset,_=get_dataset_collate(args)
    compute_gt_gt = False
    # if compute_gt_gt:
    #     datasetGT = {key: [get_datasets(parameters)[key],
    #                        get_datasets(parameters)[key]]
    #                  for key in ["train", "test"]}
    # else:
    #     datasetGT = {key: [get_datasets(parameters)[key]]
    #                  for key in ["train", "test"]}
    if compute_gt_gt:
        args.split="train"
        #args.pose_rep="xyz"
        #args.pose_rep="rot6d"
        train_dataset_1,collate=get_dataset_collate(args)
        args.split="test"
        test_dataset_1,_=get_dataset_collate(args)
        datasetGT = {"train":[train_dataset,train_dataset_1],
                     "test":[test_dataset,test_dataset_1]}
    else:
        datasetGT = {"train":[train_dataset],
                    "test":[test_dataset]}
    

    print("Dataset loaded")

    allseeds = list(range(niter))


    for seed in allseeds:
        fixseed(seed)
        for key in ["train", "test"]:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs,
                                         shuffle=False, num_workers=0,
                                         collate_fn=collate)
                              for data in datasetGT[key]]
                        for key in ["train", "test"]}

        if doing_recons:
            reconsLoaders = {key: NewDataloader(args,"rc", model, parameters,
                                                dataiterator[key][0],
                                                device)
                             for key in ["train", "test"]}

        gtLoaders = {key: NewDataloader(args,"gt", model, parameters,
                                        dataiterator[key][0],
                                        device)
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: NewDataloader(args,"gt", model, parameters,
                                             dataiterator[key][1],
                                             device)
                          for key in ["train", "test"]}

        genLoaders = {key: NewDataloader(args,"gen", model, parameters,
                                         dataiterator[key][0],
                                         device)
                      for key in ["train", "test"]}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}
        if doing_recons:
            loaders["recons"] = reconsLoaders

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

        # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
        # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = epoch
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
