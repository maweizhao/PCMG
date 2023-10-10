import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)


from evaluate.action2motion.models import MotionDiscriminator
from _parser.parser import get_parser
from _dataset.src.dataset import get_dataset_collate
from torch.utils.data import DataLoader
import datetime
import torch
from tqdm import *

from visuals.log import log_out_message




parameters,args  = get_parser()

dataset_opt = {"ntu13": {"joints_num": 18,
                        "input_size_raw": 54,
                        "num_classes": 13},
            'humanact12': {"input_size_raw": 72,
                            "joints_num": 24,
                            "num_classes": 12},
            'uestc': {"input_size_raw": 72,
                        "joints_num": 24,
                        "num_classes": 40},
            }
dataname=args.dataset
input_size_raw = dataset_opt[dataname]["input_size_raw"]
num_classes = dataset_opt[dataname]["num_classes"]
device = parameters["device"]
model=MotionDiscriminator(input_size_raw, 128, 2, device=device, output_size=num_classes).to(device)


args.split="train"
args.pose_rep="xyz"
dataset,collate=get_dataset_collate(args)
train_iterator = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,collate_fn=collate)
args.split="test"
args.pose_rep="xyz"
dataset,_=get_dataset_collate(args)
test_iterator = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,collate_fn=collate)


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

args.save_model_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/gru_check_point/"+str(datetime.datetime.now().year)+"_"+str(datetime.datetime.now().month)+"_"+str(datetime.datetime.now().day)+"_modelpara.pth")

continue_train=True
continue_train=False
if not continue_train:
    # 需要继续训练的模型的路径
    begin_epoch=0
    model_save_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/check_point/gru_check_point/ontraining/"+"_"+str(args.num_frames)+"_"+str(args.rec_loss)+"_"+str(args.jointstype)+"_"+str(args.dataset)+"_"+str(datetime.datetime.now())+"/")
    model_save_path=model_save_path.replace(':','-')
    
    model_save_path = list(model_save_path)
    model_save_path[1] = ':'
    model_save_path = ''.join(model_save_path)
    
    print(model_save_path)
    if not(os.path.exists(model_save_path)):
        os.mkdir(model_save_path)
    ontraining_model_path=model_save_path+"modelpara.pth"
else:
    model_save_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/gru_check_point/ontraining/_60_l2_vertice_smpl_uestc_2022-06-08 00-26-40.469124 -1/")
    ontraining_model_path=model_save_path+"modelpara.pth"

log_out_message(model_save_path,str(parameters))

#ontraining_model_path="./check_point/ontraining/modelpara.pth"
if(os.path.exists(ontraining_model_path)):
    model_dict=torch.load(ontraining_model_path)
    begin_epoch=model_dict["epoch"]
    model.load_state_dict(model_dict["model"])
    #optimizer.load_state_dict(model_dict["optimizer"])
    
from evaluate.rotation_conversions.rotation2xyz import Rotation2xyz
rot2xyz = Rotation2xyz(device="cuda")    
    
def train_epoch(epoch,smallest_loss,optimizer):
    all_loss=0
    all_accuracy=0
    for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):
        model.train()

        batch["xyz"] = rot2xyz(x=batch["xyz"].permute(0,2,3,1).cuda(), mask=batch["mask"], pose_rep='rot6d', glob=args.glob,
                translation=args.translation, jointstype='smpl', vertstrans=args.vertstrans, betas=None,
                beta=0, glob_rot=args.glob_rot, get_rotations_back=False).permute(0,3,1,2)

        optimizer.zero_grad()
        # B,L,N,3
        x=batch["xyz"].permute(0,2,3,1).to(device)
        #print(x.shape)
        length=batch["lengths"].to(device)
        output=model(motion_sequence=x,lengths=length)
        #print(output.shape)
        criterion=torch.nn.CrossEntropyLoss(reduction='mean')
        loss=criterion(output,batch["y"].to(device))

        confusion = torch.zeros(args.num_classes, args.num_classes, dtype=int)
        yhat = output.max(dim=1).indices
        ygt = batch["y"]
        for label, pred in zip(ygt, yhat):
            confusion[label][pred] += 1
        accuracy = torch.trace(confusion)/torch.sum(confusion)
        
        
        all_loss+=loss
        all_accuracy+=accuracy
        loss.backward()
        optimizer.step()
        
        #break
    #break
    # 保存参数
    all_loss=all_loss/len(train_iterator)
    all_accuracy=all_accuracy/len(train_iterator)
    # for p in model.parameters():
    #     print(p)
    if(smallest_loss>all_loss):
        smallest_loss=all_loss
    #     state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    #     torch.save(state, ontraining_model_path)
    #     #print("save at:%s"%(ontraining_model_path))
    log_info="----------------out train-------------------%d个epoch,lr:%10.8f,min_loss:%10.8f"%((epoch+1),optimizer.state_dict()['param_groups'][0]['lr'],smallest_loss)
    log_out_message(model_save_path,log_info)
    log_info="loss:%10.8f,acc:%10.8f"%(all_loss,all_accuracy)
    log_out_message(model_save_path,log_info)
    return smallest_loss
    

def test_epoch(epoch,smallest_loss,highest_testacc):
    with torch.no_grad():
        all_loss=0
        all_accuracy=0
        for i, batch in tqdm(enumerate(test_iterator), desc="Computing batch",total=len(test_iterator)):
            
            batch["xyz"] = rot2xyz(x=batch["xyz"].permute(0,2,3,1).cuda(), mask=batch["mask"], pose_rep='rot6d', glob=args.glob,
                translation=args.translation, jointstype='smpl', vertstrans=args.vertstrans, betas=None,
                beta=0, glob_rot=args.glob_rot, get_rotations_back=False).permute(0,3,1,2)

            model.eval()
            # B,L,N,3
            x=batch["xyz"].permute(0,2,3,1).to(device)
            length=batch["lengths"].to(device)
            output=model(motion_sequence=x,lengths=length)
            #print(output.shape)
            criterion=torch.nn.CrossEntropyLoss(reduction='mean')
            loss=criterion(output,batch["y"].to(device))

            confusion = torch.zeros(args.num_classes, args.num_classes, dtype=int)
            yhat = output.max(dim=1).indices
            ygt = batch["y"]
            for label, pred in zip(ygt, yhat):
                confusion[label][pred] += 1
            accuracy = torch.trace(confusion)/torch.sum(confusion)
            
            
            all_loss+=loss
            all_accuracy+=accuracy

            
            #break
        #break
        # 保存参数
        all_loss=all_loss/len(test_iterator)
        all_accuracy=all_accuracy/len(test_iterator)
        # for p in model.parameters():
        #     print(p)
        if(smallest_loss>all_loss):
            smallest_loss=all_loss
        if(highest_testacc<all_accuracy):
            highest_testacc=all_accuracy
            state = {'net':model.state_dict(), 'epoch':epoch}
            torch.save(state, ontraining_model_path)
            #print("save at:%s"%(ontraining_model_path))
        log_info="----------out test----------%d个epoch,highest_acc:%10.8f,min_loss:%10.8f"%((
                                                        epoch+1),highest_testacc,smallest_loss)
        log_out_message(model_save_path,log_info)
        log_info="loss:%10.8f,acc:%10.8f"%(all_loss,all_accuracy)
        log_out_message(model_save_path,log_info)
        return smallest_loss,highest_testacc
    

smallest_trainloss=99999
smallest_testloss=99999
highest_testacc=0
for epoch in range(begin_epoch, args.num_epochs):
    
    if epoch==4.0* args.num_epochs//5:
        lr=args.lr*0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    smallest_trainloss=train_epoch(epoch,smallest_trainloss,optimizer)
    smallest_testloss,highest_testacc=test_epoch(epoch,smallest_testloss,highest_testacc)

    
    
    
    
