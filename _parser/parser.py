
from argparse import ArgumentParser
from email.policy import default

def add_options(parser):
    group = parser.add_argument_group('options')
    
    group.add_argument("--num_epochs", type=int, default=50000, help="AdamW: learning rate")
    group.add_argument("--lr", type=float, default=0.0001, help="AdamW: learning rate")
    group.add_argument("--batch_size", default=20,type=int, help="batch size")
    group.add_argument("--latent_dim", default=256,type=int, help="batch size")
    
    group.add_argument("--model",default="PCMG",choices=[ "PCMG","ACTOR"], help="model name")
    group.add_argument("--evaluate_model",default="GRU",choices=[ "GRU","STGCN","P4Transformer"], help="model name")
    
    # dataset
    group.add_argument("--dataset", default="humanact12",choices=[ "uestc","4Dcompelte", "humanact12","cmu"], help="Dataset to load")
    group.add_argument("--jointstype", default="vertices",choices=[ "vertices","smpl","mocap_surface","vertices+smpl"], help="Dataset to load")
    group.add_argument("--pose_rep", default="xyz", choices=["xyz", "rotvec", "rotmat", "rotquat", "rot6d"], help="xyz or rotvec etc")
    group.add_argument("--num_frames", default=60,type=int, help="Dataset to load")      #cmu:100,smpl:60,vertice:30/60 
    group.add_argument("--mask", default=True,type=bool, help="Dataset to load")
    group.add_argument("--split", default="train",type=str, help="Dataset spilt")
    group.add_argument('--pointcloud_in_order',type=bool,default=True, help="Training with vertex translations in the SMPL mesh")

    
    group.add_argument("--glob",default=True, dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument("--nfeats", default=6, type=int, help="sampling step")
    group.add_argument("--translation",default=True, dest='translation', action='store_true',help="if we want to output translation")
    group.add_argument('--vertstrans',default=False, dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument('--glob_rot',default=[3.141592653589793, 0, 0], help="Training with vertex translations in the SMPL mesh")
    
    group.add_argument("--num_classes", default=12,type=int, help="number of motion classes")  #uestc=40,humanact12=12,cmu=8
    group.add_argument("--num_animal_classes", default=1,type=int, help="number of motion classes")
    group.add_argument("--input_dim", default=3,type=int, help="number of motion classes")
    group.add_argument("--points_num", default=1024,type=int, help="number of motion classes")   #cmu:20,smpl:24,vertice:1024,mocap_surface:53

    #Ablation study
    group.add_argument("--rec_loss", default="l2_vertice", choices=[ "cd_density", "emd","l2_vertice","l2_joint_vertices"],help="reconstruction loss")
    group.add_argument("--point_encoder", default="pointnet", choices=[ "pointnet", "pointnet_res","linear","PVCNN"],help="reconstruction loss")
    group.add_argument("--point_decoder", default="Transformer_pointdecoder", choices=[ "AXform_pointdecoder", "Transformer_pointdecoder","linear"],help="reconstruction loss")
    
    group.add_argument("--PVCNN_width_multiplier", default=0.125,type=float, help="number of knn in decoder")

    group.add_argument("--AXform_pointdecoder_num_branch", default=16,type=int, help="number of knn in decoder")
    group.add_argument("--AXform_K2", default=32,type=int, help="number of knn in decoder")
    group.add_argument("--AXform_N", default=128,type=int, help="number of knn in decoder")
    
    #loss
    #group.add_argument("--lambda_cd", type=float, default=1e-6, help="lambda_kl")  #l2取1e-2,cd取5e-2
    group.add_argument("--lambda_kl", type=float, default=1e-2, help="lambda_kl")  #l2取1e-2,cd取1e-5,emd取1e-0
    group.add_argument("--density_k", type=int, default=1024, help="lambda_kl")  
    group.add_argument("--lambda_density", type=float, default=1, help="lambda_kl")
    group.add_argument("--lambda_joint", type=float, default=1, help="lambda_kl")
    group.add_argument("--lambda_vertices", type=float, default=1, help="lambda_kl")
    group.add_argument("--lambda_vertices2joint", type=float, default=1, help="lambda_kl")
    
    #transformer
    group.add_argument("--ff_size", default=1024, type=int, help="sampling step")
    group.add_argument("--num_layers", default=6, type=int, help="sampling step")
    group.add_argument("--num_heads", default=4, type=int, help="sampling step")
    group.add_argument("--dropout", default=0.1, type=int, help="sampling step")
    group.add_argument("--activation", default="gelu", help="Activation for function for the transformer layers")
    group.add_argument("--ablation", default=None, choices=[None, "average_encoder", "zandtime", "time_encoding", "concat_bias"],
                       help="Ablations for the transformer architechture")
    
    #decoder
    group.add_argument("--Transformer_pointdecoder_k", default=32,type=int, help="number of knn in decoder")
    group.add_argument("--Transformer_pointdecoder_m", default=64,type=int, help="number of knn in decoder")
    group.add_argument("--Transformer_pointdecoder_num_branch", default=1,type=int, help="number of knn in decoder")
    group.add_argument("--point_decoder_version", default="v2",choices=["v1", "v2"], help="number of knn in decoder")
    group.add_argument("--mapping_net", default=True, type=bool,help="reconstruction loss")
    
    # generate
    group.add_argument("--generate_num_frame",default=60, type=int, help="sampling step")         #cmu:100,smpl:60,vertice:30/60     
    group.add_argument('--generate_num_vertices',default=1024, type=int, help="sampling step")   #cmu:20,smpl:24,vertice:1024
    #group.set_defaults(cuda=True)
    
    # cuda
    group.add_argument("--cuda",default=True, dest='cuda', action='store_true', help="if we want to try to use gpu")
    group.add_argument('--cpu',default=False, dest='cuda', action='store_false', help="if we want to use cpu")
    #group.set_defaults(cuda=True)
    
    # train_evalute_
    group.add_argument("--train_per_evalute_epoch", default=50,type=int, help="number of knn in decoder")
    group.add_argument("--train_evalute_gen_num", default=3000,type=int, help="number of knn in decoder")
    
    

def adding_cuda(parameters):
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")


def get_parser():
    parser = ArgumentParser()
    add_options(parser)
    args = parser.parse_args(args=[])
    parameters = {key: val for key, val in vars(args).items() if val is not None}
    adding_cuda(parameters)
    return parameters,args