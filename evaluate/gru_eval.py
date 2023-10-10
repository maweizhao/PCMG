from tomlkit import key
import torch
from tqdm import tqdm
from evaluate.ACTOR.models.rotation2xyz import Rotation2xyz

from evaluate.tools import fixseed

from evaluate.action2motion.evaluate import A2MEvaluation
# from src.evaluate.othermetrics.evaluation import OtherMetricsEvaluation

from torch.utils.data import DataLoader

from _dataset.src.dataset import get_dataset_collate

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from .tools import save_metrics, format_metrics

from model.PCMG import PCMG
from  evaluate.ACTOR.models.get_model import get_model as get_ACTOR_model
from _parser.ACTOR_parser import parser as get_ACOTR_parser

from visuals.visuals import meshplot_visuals_n_joint_seq_color
import yaml

def load_args(filename):
    with open(filename, "rb") as optfile:
        opt = yaml.load(optfile, Loader=yaml.Loader)
    return opt


# def rot2xyz(x,mask=None):
#     # param2xyz = {"pose_rep": args.pose_rep,
#     #             "glob_rot": None,
#     #             "glob": args.glob_rot,
#     #             "jointstype": args.jointstype,
#     #             "translation": args.translation,
#     #             "vertstrans": args.vertstrans}
#     rotation2xyz = Rotation2xyz(device=torch.device("cuda"))
#     #args.update(param2xyz)
    
#     return rotation2xyz(x, mask,pose_rep="rot6d",
#                         translation=True,
#                         glob=True,
#                         jointstype="smpl",
#                         vertstrans=True
#                         )

from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz
rot2xyz = Rotation2xyz(device="cuda")


class NewDataloader:
    def __init__(self,args, mode, model, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    if args.model == "PCMG":
                        classes = databatch["y"].cuda()
                        gendurations = databatch["lengths"]
                        cls=torch.tensor([0],device='cuda:0')
                        #y=torch.tensor([5],device='cuda:0')
                        batch = model.generate(classes,cls,gendurations,args.points_num)
                        #print(batch["output_xyz"].shape)
                        
                        if args.dataset =="cmu":
                            batch["output"]=batch["output"]-batch["output"][:,:1,:1,:]
                        
                        batch["output_xyz"]=batch["output"].permute(0,2,3,1) #[B,L,N,3]->[B,N,3,L]
                        batch = {key: val.to(device) for key, val in batch.items()}
                        
                        #print(batch["output"].shape)
                        # meshplot_visuals_n_joint_seq_color([batch["output"][0]],["red"],"cmu")
                        # break
                        
                        
                    elif args.model == "ACTOR":
                        classes = databatch["y"]
                        gendurations = databatch["lengths"]
                        #print(gendurations)
                        batch = model.generate(classes, gendurations)  #[B, 24, 6, N_frames]
                        
                        #print(batch["output_xyz"].shape)
                        #meshplot_visuals_n_joint_seq([batch["output_xyz"][0].permute(2,0,1)])
                        #break
                        #print(batch["output_xyz"].shape)   #[B, 24, 3, N_frames]
                        batch = {key: val.to(device) for key, val in batch.items()}
                elif mode == "gt":
                    #print( type(databatch) )

                    batch = {key: val.to(device) for key, val in databatch.items()}
                    # batch["x_xyz"] = model.rot2xyz(batch["x"].to(device),
                    #                                batch["mask"].to(device))
                    #batch["x_xyz"]=batch["xyz"]
                    #batch["output"] = batch["xyz"]

                    
                    batch["output_xyz"] = batch["xyz"].permute(0,2,3,1)
                    #print(batch["output_xyz"].shape)
                    batch["output_xyz"] = rot2xyz(x=batch["output_xyz"], mask=batch["mask"], pose_rep='rot6d', glob=True,
                                    translation=True, jointstype='smpl', vertstrans=False, betas=None,
                                    beta=0, glob_rot=[3.141592653589793,0,0], get_rotations_back=False)
                    
                    #print(batch["output_xyz"].shape)
                    # meshplot_visuals_n_joint_seq_color([batch["output"][3]],["red"],"cmu")
                    # break
                    #print(batch["output_xyz"].shape)
                    #break
                
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    batch["output_xyz"] = model.rot2xyz(batch["output"],
                                                        batch["mask"])
                    batch["x_xyz"] = model.rot2xyz(batch["x"],
                                                   batch["mask"])

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)
    def len(self):
        return len(self.batches)


def evaluate(args,parameters, folder, checkpointname, niter):
    num_frames = 60

    # fix parameters for action2motion evaluation
    parameters["num_frames"] = num_frames
    if parameters["dataset"] == "ntu13":
        parameters["jointstype"] = "a2m"
        parameters["vertstrans"] = False  # No "real" translation in this dataset
    elif parameters["dataset"] == "humanact12":
        parameters["jointstype"] = "smpl"
        parameters["vertstrans"] = True
    elif parameters["dataset"] == "uestc":
        parameters["jointstype"] = "smpl"
        parameters["vertstrans"] = True
    #else:
        #raise NotImplementedError("Not in this file.")

    device = parameters["device"]
    dataname = parameters["dataset"]

    print(folder)
    
    # dummy => update parameters info
    if args.model == "PCMG":
        model=PCMG(args).cuda()
        print("Restore weights..")
        checkpointpath = os.path.join(folder, checkpointname)
        if(os.path.exists(checkpointpath)):
            model_dict=torch.load(checkpointpath)
            if "epoch" in model_dict.keys():
                epoch=model_dict["epoch"]
            if "model" in model_dict.keys():
                model.load_state_dict(model_dict["model"])
            elif  "net" in model_dict.keys():
                model.load_state_dict(model_dict["net"])
            else:
                raise NotImplementedError("pretrain model not found!")
        else:
            raise NotImplementedError("pretrain model not found!")
        
        
    elif args.model == "ACTOR":
        opt,parameters=get_ACOTR_parser()
        
        newparameters = {key: val for key, val in vars(opt).items() if val is not None}
        #folder, checkpointname = os.path.split(newparameters["checkpointname"])
        parameters_load = load_args(os.path.join(folder, "opt.yaml"))
        parameters.update(newparameters)
        parameters.update(parameters_load)
        # parameters["njoints"]=25
        # parameters["device"]=device
        parameters["pose_rep"]="rot6d"
        parameters["jointstype"]="smpl"
        model=get_ACTOR_model(parameters)
        model.outputxyz = True
        print("Restore weights..")
        checkpointpath = os.path.join(folder, checkpointname)
        epoch=checkpointname.split("_")[1].split(".")[0]
        if(os.path.exists(checkpointpath)):
            state_dict = torch.load(checkpointpath, map_location=parameters["device"])
            #print(type(state_dict))
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError("pretrain model not found!")
        
    
    model.eval()
    #model.outputxyz = True

    a2mevaluation = A2MEvaluation(args,dataname, device)
    a2mmetrics = {}

    # evaluation = OtherMetricsEvaluation(device)
    # joints_metrics = {}, pose_metrics = {}

    # datasetGT1,collate=get_dataset_collate(args)
    # datasetGT2,_=get_dataset_collate(args)
    compute_gt_gt = False
    if args.dataset == "uestc":
        args.split="train"
        #args.pose_rep="xyz"
        train_dataset,collate=get_dataset_collate(args)
        args.split="test"
        test_dataset,_=get_dataset_collate(args)
        compute_gt_gt = False
        if compute_gt_gt:
            args.split="train"
            train_dataset_1,collate=get_dataset_collate(args)
            args.split="test"
            test_dataset_1,_=get_dataset_collate(args)
            datasetGT = {"train":[train_dataset,train_dataset_1],
                            "test":[test_dataset,test_dataset_1]}
        else:
            datasetGT = {"train":[train_dataset],
                        "test":[test_dataset]}
            #print(datasetGT.keys())
    else:
        args.split="train"
        #print(args.dataset)
        #args.pose_rep="xyz"
        train_dataset,collate=get_dataset_collate(args)
        datasetGT = {"train":[train_dataset]}

    
    

    # datasetGT1 = get_datasets(parameters)["train"]
    # datasetGT2 = get_datasets(parameters)["train"]

    allseeds = list(range(niter))
    #print(allseeds)
    doing_recons = False

    try:
        for index, seed in enumerate(allseeds):
            fixseed(seed)
            for key in datasetGT.keys():
                for data in datasetGT[key]:
                    data.reset_shuffle()
                    data.shuffle()
            
            print(f"Evaluation number: {index+1}/{niter}")
            
            #print(datasetGT.keys())
            dataiterator = {key: [DataLoader(data, batch_size=parameters["batch_size"],
                                    shuffle=False, num_workers=0,
                                    collate_fn=collate)
                        for data in datasetGT[key]]
                    for key in datasetGT.keys()}
            
            if doing_recons:
                reconsLoaders = {key: NewDataloader(args,"rc", model,
                                                    dataiterator[key][0],
                                                    device)
                                for key in datasetGT.keys()}

            gtLoaders = {key: NewDataloader(args,"gt", model,
                                            dataiterator[key][0],
                                            device)
                        for key in datasetGT.keys()}

            if compute_gt_gt:
                gtLoaders2 = {key: NewDataloader(args,"gt", model,
                                                dataiterator[key][1],
                                                device)
                            for key in datasetGT.keys()}

            genLoaders = {key: NewDataloader(args,"gen", model,
                                            dataiterator[key][0],
                                            device)
                        for key in datasetGT.keys()}

            loaders = {"gen": genLoaders,
                    "gt": gtLoaders}
            if doing_recons:
                loaders["recons"] = reconsLoaders

            if compute_gt_gt:
                loaders["gt2"] = gtLoaders2

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)
            del loaders

            # dataiterator = DataLoader(datasetGT1, batch_size=parameters["batch_size"],
            #                           shuffle=False, num_workers=0, collate_fn=collate)
            # dataiterator2 = DataLoader(datasetGT2, batch_size=parameters["batch_size"],
            #                            shuffle=False, num_workers=0, collate_fn=collate)

            # # reconstructedloader = NewDataloader("rc", model, dataiterator, device)
            # motionloader = NewDataloader(args,"gen", model, dataiterator, device)
            # gt_motionloader = NewDataloader(args,"gt", model, dataiterator, device)
            # gt_motionloader2 = NewDataloader(args,"gt", model, dataiterator2, device)
            # # for batch in motionloader:
            # #     print(batch["output_xyz"].shape)
            # #     import pickle as pkl
            # #     save_name="D:\maweizhao\MyProgram\DeepLearning\myfile\ACTOR-master\my_file\my_test/ACTOR_gen.pkl"
            # #     with open(save_name, 'wb') as f:
            # #         pkl.dump(batch["output_xyz"], f)
            # #         print("save at: "+ save_name)
            # #     break

            # # Action2motionEvaluation
            # loaders = {"gen": motionloader,
            #            # "recons": reconstructedloader,
            #            "gt": gt_motionloader,
            #            "gt2": gt_motionloader2}

            # a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            # joints_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=True)
            # pose_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=False)

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    #print(metrics)
    
    mean_metrics = {"mean":{},"std":{}}
    for key in metrics["feats"].keys():
        values_tensor=torch.zeros(len(metrics["feats"][key]))
        for index,value in enumerate(metrics["feats"][key]):
            #print(value)
            #mean_value+=float(value)
            values_tensor[index]=float(value)
        #print(mean_value)
        
        mean_value=values_tensor.mean(dim=0)
        std_value=values_tensor.std(dim=0)
        mean_metrics["mean"][key]=format(mean_value,'.6e') 
        mean_metrics["std"][key]=format(std_value,'.6e') 
        
    #metrics.update(mean_metrics)
    #metrics = dict( mean_metrics.items() + metrics.items() )
    #metrics = dict( mean_metrics,**metrics )
    merge_metrics = {**mean_metrics, **metrics}
    #print(merge_metrics)
    #metrics=merge_metrics
    # mean_metrics.update(metrics)
    # metrics=mean_metrics
            
    #mean_metrics={"mean": {key:[[metrics[0][key][index]] for index in len(metrics[0][key])  ]  for key in metrics[0].keys() }}
    #print(mean_metrics)
    #metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    #epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, merge_metrics)
