import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)


from evaluate.action2motion.models import MotionDiscriminator
from _parser.parser import get_parser
from _parser.p4_transformer import p4transfomer_parse_args
from _dataset.src.dataset import get_dataset_collate
from torch.utils.data import DataLoader
import datetime
import torch
from tqdm import *

from visuals.log import log_out_message
from evaluate.P4Transformer.models.PCMG_classfier import P4Transformer_MotionDiscriminator


    
    
def train_epoch(model,train_iterator,epoch,smallest_loss,highest_acc,optimizer,device,args,ontraining_model_path,model_save_path):
    all_loss=0
    all_accuracy=0
    for i, batch in tqdm(enumerate(train_iterator), desc="Computing batch",total=len(train_iterator)):
        #model.train()
    

        optimizer.zero_grad()
        # B,L,N,3
        x=batch["xyz"].permute(0,2,3,1).to(device)  #B,L,N,3->B,N,3,L
        #x=batch["xyz"].permute(0,2,3,1).to(device)
        #print(x.shape)
        #print(device)
        output=model(x).cuda()
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
    if(highest_acc<all_accuracy):
        highest_acc=all_accuracy
        state = {'net':model.state_dict(), 'epoch':epoch}
        torch.save(state, ontraining_model_path)
    #     state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    #     torch.save(state, ontraining_model_path)
    #     #print("save at:%s"%(ontraining_model_path))
    log_info="----------------out train-------------------%d个epoch,lr:%10.8f,highest_acc:%10.8f,min_loss:%10.8f"%((
                                                    epoch+1),optimizer.state_dict()['param_groups'][0]['lr'],highest_acc,smallest_loss)
    log_out_message(model_save_path,log_info)
    log_info="loss:%10.8f,acc:%10.8f"%(all_loss,all_accuracy)
    log_out_message(model_save_path,log_info)
    return smallest_loss,highest_acc
    

def main():



    parameters,args  = get_parser()
    p4_parameters,p4_args=p4transfomer_parse_args()
    
    device = parameters["device"]
    model=P4Transformer_MotionDiscriminator(radius=p4_args.radius, nsamples=p4_args.nsamples, spatial_stride=p4_args.spatial_stride,
                  temporal_kernel_size=p4_args.temporal_kernel_size, temporal_stride=p4_args.temporal_stride,
                  emb_relu=p4_args.emb_relu,
                  dim=p4_args.dim, depth=p4_args.depth, heads=p4_args.heads, dim_head=p4_args.dim_head,
                  mlp_dim=p4_args.mlp_dim, num_classes=args.num_classes).to(device)

    dataset,collate=get_dataset_collate(args)
    train_iterator = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.save_model_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/check_point/p4transformer/"+str(datetime.datetime.now().year)+"_"+str(datetime.datetime.now().month)+"_"+str(datetime.datetime.now().day)+"_modelpara.pth")

    continue_train=True
    continue_train=False
    if not continue_train:
        # 需要继续训练的模型的路径
        begin_epoch=0
        model_save_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/check_point/p4transformer/ontraining/"+"_"+str(args.num_frames)+"_"+str(args.rec_loss)+"_"+str(args.jointstype)+"_"+str(args.dataset)+"_"+str(datetime.datetime.now())+"/")
        model_save_path=model_save_path.replace(':','-')
        
        model_save_path = list(model_save_path)
        model_save_path[1] = ':'
        model_save_path = ''.join(model_save_path)
        
        print(model_save_path)
        if not(os.path.exists(model_save_path)):
            os.mkdir(model_save_path)
        ontraining_model_path=model_save_path+"modelpara.pth"
    else:
        model_save_path=os.path.join(PROJECT_ROOT_DIR,"evaluate/check_point/p4transformer/ontraining/_60_l2_vertice_smpl_uestc_2022-06-08 00-26-40.469124 -1/")
        ontraining_model_path=model_save_path+"modelpara.pth"

    log_out_message(model_save_path,str(parameters))
    log_out_message(model_save_path,str(p4_parameters))

    #ontraining_model_path="./check_point/ontraining/modelpara.pth"
    if(os.path.exists(ontraining_model_path)):
        model_dict=torch.load(ontraining_model_path)
        begin_epoch=model_dict["epoch"]
        model.load_state_dict(model_dict["model"])
        #optimizer.load_state_dict(model_dict["optimizer"])


    smallest_trainloss=99999
    smallest_testloss=99999
    highest_testacc=0
    model.train()
    for epoch in range(begin_epoch, args.num_epochs):
        
        if epoch==4.0* args.num_epochs//5:
            lr=args.lr*0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        smallest_trainloss,highest_testacc=train_epoch(model, train_iterator,epoch,smallest_trainloss,highest_testacc,optimizer,device,args,ontraining_model_path,model_save_path)
        
        
        
        #smallest_testloss,highest_testacc=test_epoch(model,test_iterator,epoch,smallest_testloss,highest_testacc,device,args,ontraining_model_path,model_save_path)

    
if __name__ == '__main__':
    main()
    
    
