
import datetime
import math
from operator import mod
import os
from statistics import mode
import sys
from _dataset.src.dataset import get_dataset_collate
from tqdm import *
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch
from evaluate.action2motion.models import load_classifier as load_gru_classifier
from model.PCMG import PCMG
from utils.furthestpointsample import farthest_point_sample,index_points

from evaluate.action2motion.accuracy import calculate_accuracy

from utils.emd_loss import compute_emd_loss
from visuals.log import log_out_message

# ROOT_DIR = "D:\maweizhao\MyProgram\DeepLearning\myfile\1\PCMG"
# sys.path.append(ROOT_DIR)

from _parser.parser import get_parser


# ROOT_DIR = "D:\maweizhao\MyProgram\DeepLearning\myfile\1\PCMG"
# sys.path.append(ROOT_DIR)


def calculate_acc(args,model,evalute_lassifier,iterator,model_save_path,epoch):
    with torch.no_grad():
        motion_loader=[]
        model.eval()
        for i, batch in tqdm(enumerate(iterator), desc="Computing accuracy...",total=len(iterator)):
            classes = batch["y"].cuda()
            gendurations = batch["lengths"]
            cls=torch.tensor([0],device='cuda:0')
            #y=torch.tensor([5],device='cuda:0')
            batch = model.generate(classes,cls,gendurations,args.points_num)
            #print(batch["output_xyz"].shape)
            # cmu offset
            #batch["output"]=batch["output"]-batch["output"][:,:1,:1,:]
            batch["output_xyz"]=batch["output"].permute(0,2,3,1) #[B,L,N,3]->[B,N,3,L]
            if(args.jointstype=="vertices+smpl"):
                batch["output_xyz"]=batch["output_xyz"][:,:24,:,:]
            batch = {key: val.to('cuda:0') for key, val in batch.items()}
            motion_loader.append(batch)
            
        acc,_=calculate_accuracy(model=None,motion_loader=motion_loader,num_labels=args.num_classes,classifier=evalute_lassifier,device="cuda:0")
        log_info="----------------------------------------------------------------"
        log_out_message(model_save_path,log_info)
        log_info="epoch:%d,acc:%10.8f"%(epoch+1,acc)
        log_out_message(model_save_path,log_info)
        log_info="----------------------------------------------------------------"
        log_out_message(model_save_path,log_info)



def main():
    parameters,args=get_parser()
    model=PCMG(args).cuda()
    dataset,collate=get_dataset_collate(args)
    train_iterator = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.save_model_path="./check_point/"+str(datetime.datetime.now().year)+"_"+str(datetime.datetime.now().month)+"_"+str(datetime.datetime.now().day)+"_modelpara.pth"


    continue_train=True
    continue_train=False
    if not continue_train:
        # 需要继续训练的模型的路径
        begin_epoch=0
        model_save_path="./check_point/ontraining/"+"_"+str(args.dataset)+"_"+str(args.num_frames)+"_"+str(args.rec_loss)+"_"+str(args.jointstype)+"_"+str(args.dataset)+"_"+str(datetime.datetime.now())+"/"
        model_save_path=model_save_path.replace(':','-')
        if not(os.path.exists(model_save_path)):
            os.mkdir(model_save_path)
        ontraining_model_path=model_save_path+"modelpara.pth"
    else:
        model_save_path="./check_point/ontraining/_humanact12_30_l2_vertice_vertices_humanact12_2022-06-24 02-29-13.616135_2000/"
        ontraining_model_path=model_save_path+"modelpara.pth"

    log_out_message(model_save_path,str(parameters))

    #ontraining_model_path="./check_point/ontraining/modelpara.pth"
    if(os.path.exists(ontraining_model_path)):
        model_dict=torch.load(ontraining_model_path)
        begin_epoch=model_dict["epoch"]
        model.load_state_dict(model_dict["net"])
        if begin_epoch>=4.0* args.num_epochs//5:  
            lr=args.lr*0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        #optimizer.load_state_dict(model_dict["optimizer"])

    if args.evaluate_model == "GRU":
        if(args.jointstype=="smpl"):
            evalute_classifier=load_gru_classifier(args.dataset, args.points_num*3,args.num_classes, device="cuda:0").eval()
        elif(args.jointstype=="vertices+smpl"):
            evalute_classifier=load_gru_classifier(args.dataset, 24*3,args.num_classes, device="cuda:0").eval()
    elif args.evaluate_model == "P4Transformer":
        from evaluate.P4Transformer.models.PCMG_classfier import load_classifier as load_P4Transformer_classifier
        evalute_classifier=load_P4Transformer_classifier(args.dataset, args.points_num*3,args.num_classes, device="cuda:0").eval()

    smallest_loss=99999
    from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz
    rot2xyz = Rotation2xyz(device="cuda")
    for epoch in range(begin_epoch, args.num_epochs):

        all_loss=0
        all_loss_vertices=0
        all_loss_kl=0
        all_loss_density=0
        #model.train()
        for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):
            
            #print(batch["xyz"].shape)
            #print(batch["mask"].shape)

            #print(batch["xyz"].shape)
            if args.dataset == "uestc":
                batch["xyz"] = rot2xyz(x=batch["xyz"].permute(0,2,3,1).cuda(), mask=batch["mask"], pose_rep='rot6d', glob=args.glob,
                                translation=args.translation, jointstype='smpl', vertstrans=args.vertstrans, betas=None,
                                beta=0, glob_rot=args.glob_rot, get_rotations_back=False).permute(0,3,1,2)
            
            #print( batch["xyz"].shape )

            if epoch==4.0* args.num_epochs//5:
                lr=args.lr*0.1
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            # if epoch==4.5* args.num_epochs//5:
            #     lr=args.lr*0.01
            #     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            #     #for param_group in optimizer.param_groups:
            #         #param_group["lr"] = lr

            
            optimizer.zero_grad()
            output=model(batch)
            #print(output.shape)
            
            
            vertice_loss=0
            density_loss=0
            
            if args.rec_loss == "cd_density":
                vertice_loss=model.chamfer_distance_loss(batch["xyz"].cuda(),batch["output"].cuda())
                density_loss=model.compute_density_loss(batch["xyz"].cuda(),batch["output"].cuda())
            elif args.rec_loss == "emd":
                vertice_loss=compute_emd_loss(batch["xyz"].cuda(),batch["output"].cuda())
            elif args.rec_loss == "l2_vertice":
                vertice_loss=model.compute_vertices_loss( batch["xyz"],batch["output"])
            elif args.rec_loss == "l2_joint_vertices":
                
                joint_output=batch["output"][:,:,:24,:]  #[B,L,24,3]
                joint_output=joint_output-joint_output[:,:,:1,:]

                vertices_output=batch["output"][:,:,24:,:]

                B,L,N,C=vertices_output.shape
                joint_gt=batch["xyz"][:,:,:24,:]
                joint_gt=joint_gt-joint_gt[:,:,:1,:]

                vertices_gt=batch["xyz"][:,:,24:,:]

                import pickle as pkl
                m_J_regressor_path="./_dataset/data/smpl_1024vertices_fps_index/"+"_m_1024_J_regressor.pkl"
                m_J_regressor=pkl.load(open(m_J_regressor_path, "rb"))
                from smplx.lbs import vertices2joints
                temp_vertices_output=vertices_output.reshape(B*L,N,C)
                vertices_2_joint=vertices2joints(m_J_regressor,temp_vertices_output)
                vertices_2_joint=vertices_2_joint.reshape(B,L,24,C)
                vertices_2_joint=vertices_2_joint-vertices_2_joint[:,:,:1,:]



                vertice_loss =args.lambda_joint* model.compute_vertices_loss(joint_gt,joint_output
                                )+args.lambda_vertices* model.compute_vertices_loss(vertices_gt,vertices_output
                                ) +args.lambda_vertices2joint* model.compute_vertices_loss(joint_gt,vertices_2_joint)
            
            #vertice_loss=compute_emd_loss(batch["xyz"].cuda(),batch["output"].cuda())
            #loss=model.chamfer_distance_loss(output,batch["xyz"].cuda())
            #vertice_loss=model.compute_vertices_loss(batch)
            #vertice_loss=model.chamfer_distance_loss(batch["xyz"].cuda(),batch["output"].cuda())
            #density_loss=model.compute_density_loss(batch["xyz"].cuda(),batch["output"].cuda())
            #vertice_loss,density_loss=model.chamfer_density_loss(batch["xyz"].cuda(),batch["output"].cuda())
            #vertice_loss=model.compute_vertices_loss(batch)+args.lambda_cd*model.chamfer_distance_loss(batch["xyz"].cuda(),batch["output"].cuda())
            kl_loss=model.compute_kl_loss(batch)
            
            loss=args.lambda_kl*kl_loss+vertice_loss+args.lambda_density*density_loss
            
            all_loss+=loss
            all_loss_vertices+=vertice_loss
            all_loss_density+=density_loss
            all_loss_kl+=kl_loss
            loss.backward()
            optimizer.step()

            # elif epoch==4.5* args.num_epochs//5:
            #     lr=args.lr*0.01
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = lr
            
            
            
            #break
        #break
        # 保存参数
        all_loss=all_loss/len(train_iterator)
        all_loss_vertices=all_loss_vertices/len(train_iterator)
        all_loss_kl=all_loss_kl/len(train_iterator)
        all_loss_density=all_loss_density/len(train_iterator)
        # for p in model.parameters():
        #     print(p)
        if(smallest_loss>all_loss):
            smallest_loss=all_loss
            state = {'net':model.state_dict(), 'epoch':epoch}
            torch.save(state, ontraining_model_path)
            #print("save at:%s"%(ontraining_model_path))
        log_info="----------------out train-------------------%d个epoch,lr:%10.8f,min_loss:%10.8f"%((epoch+1),optimizer.state_dict()['param_groups'][0]['lr'],smallest_loss)
        log_out_message(model_save_path,log_info)
        log_info="loss:%10.8f,vertices loss:%10.8f,density loss:%10.8f,kl loss:%10.8f"%(all_loss,all_loss_vertices,all_loss_density,all_loss_kl)
        log_out_message(model_save_path,log_info)

        # evalute accuracy
        if((epoch+1)%args.train_per_evalute_epoch==0):
            # if(os.path.exists(ontraining_model_path)):
            #     model_dict=torch.load(ontraining_model_path)
            #     #begin_epoch=model_dict["epoch"]
            #     model.load_state_dict(model_dict["net"])
            calculate_acc(args,model,evalute_classifier,train_iterator,model_save_path,epoch)
            model.train()
            

if __name__ == '__main__':
    main()