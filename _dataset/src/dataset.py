import random
import pickle as pkl
import sys
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)


humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}

# cmu
# (0, 'Walk')
# (1, 'Wash')
# (2, 'Run')
# (3, 'Jump')
# (4, 'Animal Behavior')
# (5, 'Dance')
# (6, 'Step')
# (7, 'Climb')



# UESTC_coarse_action_enumerator = {
# 0:"punching-and-knee-lifting",
# 1 marking-time-and-knee-lifting
# 2 jumping-jack
# 3 squatting
# 4 forward-lunging
# 5 left-lunging
# 6 left-stretching
# 7  raising-hand-and-jumping
# 8 left-kicking
# 9 rotation-clapping
# 10 front-raising
# 11pulling-chest-expanders
# 12punching
# 13wrist-circling
# 14single-dumbbell-raising
# 15shoulder-raising
# 16elbow-circling
# 17dumbbell-one-arm-shoulder-pressing
# 18arm-circling
# 19dumbbell-shrugging
# 20pinching-back
# 21head-anticlockwise-circling
# 22shoulder-abduction
# 23deltoid-muscle-stretching
# 24straight-forward-flexion
# 25spinal-stretching
# 26dumbbell-side-bend
# 27standing-opposite-elbow-to-knee-crunch
# 28standing-rotation
# 29overhead-stretching
# 30upper-back-stretching
# 31knee-to-chest
# 32knee-circling
# 33alternate-knee-lifting
# 34bent-over-twist
# 35rope-skipping
# 36standing-toe-touches
# 37standing-gastrocnemius-calf
# 38single-leg-lateral-hopping
# 39high-knees-running
# }

class HumanAct12VerticesDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path=data_path
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        
    def __len__(self):
        return len(self.vertices_list)
    
    def __getitem__(self,index):
        vertices=self.vertices_list[index]
        motion_calsses=self.motion_calsses[index] 
        cls= self.cls_list[index]
        return vertices,motion_calsses,cls
    
    
class HumanAct12_smpl_VerticesDataset(torch.utils.data.Dataset):
    def __init__(self,args,data_path):
        super().__init__()
        self.data_path=data_path
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        self.mask_list=load_data["mask"]
        self.lengths_list=load_data["lengths"]
        self.args=args
        self.split = args.split
        
        self._train = list(range(len(self.vertices_list)))
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None
        
    def __len__(self):
        return len(self.vertices_list)
    
    def __getitem__(self,index):
        if self.split == 'train':
            index = self._train[index]
        else:
            index = self._test[index]
        
        
        vertices=self.vertices_list[index]
        motion_calsses=self.motion_calsses[index] 
        cls= self.cls_list[index]
        lengths=self.lengths_list[index]
        mask=self.mask_list[index]
        
        #print(mask)
        # max_len=self.args.num_frames
        # #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
        # mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
        # mask=mask.squeeze(0).to(bool)
        
        # print("vertices.shape")
        # print(vertices.shape)
        if(self.args.num_frames!=-1):
            
            num_vertices,num_channel=vertices.shape[1],vertices.shape[2]
            shape = (self.args.num_frames,num_vertices,num_channel)
            canvas = vertices.new_zeros(size=shape)
            #print(canvas.shape)
            for i, b in enumerate(vertices):
                if(i<self.args.num_frames):
                    canvas[i]=b
            vertices=canvas
            # print(mask.shape)
            # print(self.num_frames)
            max_len=self.args.num_frames
            #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
            mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
            mask=mask.squeeze(0).to(bool)
            #mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            # print(lengths)
            #print(mask.shape)
            #print(vertices.shape)
        
        
        return vertices,motion_calsses,cls,mask,lengths
    
    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
    
    
class UESTC_smpl_VerticesDataset(torch.utils.data.Dataset):
    def __init__(self,args,data_path):
        super().__init__()
        self.data_path=data_path
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        self.mask_list=load_data["mask"]
        self.lengths_list=load_data["lengths"]
        self.args=args
        self.split = args.split
        
        self._train = list(range(len(self.vertices_list)))
        self._test = list(range(len(self.vertices_list)))
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None
        
    def __len__(self):
        return len(self.vertices_list)
    
    def __getitem__(self,index):
        if self.split == 'train':
            index = self._train[index]
        else:
            index = self._test[index]
        
        
        vertices=self.vertices_list[index][0]
        motion_calsses=self.motion_calsses[index] 
        cls= self.cls_list[index]
        lengths=self.lengths_list[index]
        mask=self.mask_list[index]
        
        #print(mask)
        # max_len=self.args.num_frames
        # #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
        # mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
        # mask=mask.squeeze(0).to(bool)
        
        # print("vertices.shape")
        # print(vertices.shape)
        if(self.args.num_frames!=-1):
            #print(vertices.shape)
            num_vertices,num_channel=vertices.shape[1],vertices.shape[2]
            shape = (self.args.num_frames,num_vertices,num_channel)
            canvas = vertices.new_zeros(size=shape)
            #print(canvas.shape)
            for i, b in enumerate(vertices):
                if(i<self.args.num_frames):
                    canvas[i]=b
            vertices=canvas
            # print(mask.shape)
            # print(self.num_frames)
            max_len=self.args.num_frames
            #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
            mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
            mask=mask.squeeze(0).to(bool)
            #mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            # print(lengths)
            #print(mask.shape)
            #print(vertices.shape)
        
        
        return vertices,motion_calsses,cls,mask,lengths
    
    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
    
 
 
 
class dateset_joint_VerticesDataset(torch.utils.data.Dataset):
    def __init__(self,args,data_path):
        super().__init__()
        self.data_path=data_path
        
        if(args.cuda):
            self.device=torch.device('cuda:0')
        else:
            self.device=torch.device('cpu')
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        self.mask_list=load_data["mask"]
        #print(self.mask_list[0])
        self.lengths_list=load_data["lengths"]
        self.args=args
        self.split = args.split
        
        self._train = list(range(len(self.vertices_list)))
        self._test = list(range(len(self.vertices_list)))
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None
        
    def __len__(self):
        return len(self.vertices_list)

    def __getitem__(self,index):
        if self.split == 'train':
            index = self._train[index]
        else:
            index = self._test[index]
        
        
        vertices=self.vertices_list[index].to(self.device)
        motion_calsses=self.motion_calsses[index]
        cls= self.cls_list[index]
        lengths=self.lengths_list[index].to(self.device)
        #print(lengths)
        mask=self.mask_list[index].to(self.device)
        
        #print(mask)
        # max_len=self.args.num_frames
        # #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
        # mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
        # mask=mask.squeeze(0).to(bool)
        
        # print("vertices.shape")
        # print(vertices.shape)
        if(self.args.num_frames!=-1):
            
            num_vertices,num_channel=vertices.shape[1],vertices.shape[2]
            shape = (self.args.num_frames,num_vertices,num_channel)
            canvas = vertices.new_zeros(size=shape)
            #print(canvas.shape)
            for i, b in enumerate(vertices):
                if(i<self.args.num_frames):
                    canvas[i]=b
            vertices=canvas
            # print(mask.shape)
            # print(self.num_frames)
            max_len=self.args.num_frames
            #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
            mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
            mask=mask.squeeze(0).to(bool)
            #mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            # print(lengths)
            #print(mask.shape)
            #print(vertices.shape)
        
        
        return vertices,motion_calsses,cls,mask,lengths

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
 
 
class dateset_joint_plus_verticesDataset(torch.utils.data.Dataset):
    def __init__(self,args,data_path):
        super().__init__()
        self.data_path=data_path
        
        if(args.cuda):
            self.device=torch.device('cuda:0')
        else:
            self.device=torch.device('cpu')
        load_data = pkl.load(open(data_path, "rb"))
        self.joint_list=load_data["joint"]
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        self.mask_list=load_data["mask"]
        #print(self.mask_list[0])
        self.lengths_list=load_data["lengths"]
        self.args=args
        self.split = args.split
        
        self._train = list(range(len(self.vertices_list)))
        self._test = list(range(len(self.vertices_list)))
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None
        
    def __len__(self):
        return len(self.vertices_list)

    def __getitem__(self,index):
        if self.split == 'train':
            index = self._train[index]
        else:
            index = self._test[index]
        
        
        vertices=torch.cat([self.joint_list[index].to(self.device),self.vertices_list[index].to(self.device)],dim=1)  #[n_frame,1024+24,3]
        motion_calsses=self.motion_calsses[index]
        cls= self.cls_list[index]
        lengths=self.lengths_list[index].to(self.device)
        #print(lengths)
        mask=self.mask_list[index].to(self.device)
        
        #print(mask)
        # max_len=self.args.num_frames
        # #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
        # mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
        # mask=mask.squeeze(0).to(bool)
        
        # print("vertices.shape")
        # print(vertices.shape)
        if(self.args.num_frames!=-1):
            
            num_vertices,num_channel=vertices.shape[1],vertices.shape[2]
            shape = (self.args.num_frames,num_vertices,num_channel)
            canvas = vertices.new_zeros(size=shape)
            #print(canvas.shape)
            for i, b in enumerate(vertices):
                if(i<self.args.num_frames):
                    canvas[i]=b
            vertices=canvas
            # print(mask.shape)
            # print(self.num_frames)
            max_len=self.args.num_frames
            #mask = torch.arange(max_len, device=torch.device("cuda")).expand(lengths, max_len) < lengths
            mask = torch.arange(max_len, device=torch.device("cuda")).expand(1, max_len) < lengths  
            mask=mask.squeeze(0).to(bool)
            #mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            # print(lengths)
            #print(mask.shape)
            #print(vertices.shape)
        
        
        return vertices,motion_calsses,cls,mask,lengths

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test
    
    
class UESTC_VerticesDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path=data_path
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.cls_list=load_data["cls"]
        self.mask_list=load_data["mask"]
        self.lengths_list=load_data["lengths"]
        
    def __len__(self):
        return len(self.vertices_list)
    
    def __getitem__(self,index):
        vertices=self.vertices_list[index]
        motion_calsses=self.motion_calsses[index] 
        cls= self.cls_list[index]
        
        return vertices,motion_calsses,cls
    
    
class _4DcompelteVerticesDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path=data_path
        load_data = pkl.load(open(data_path, "rb"))
        self.vertices_list=load_data["x"]
        self.motion_calsses=load_data["y"]
        self.motion_dic=load_data["y_dic"]
        self.class_dic=load_data["animal_class_dic"]
        self.animal_class_list=load_data["animal_class_list"]
        #print(load_data["animal_class_list"])
        
    def __len__(self):
        return len(self.vertices_list)
    
    def __getitem__(self,index):
        vertices=self.vertices_list[index]
        motion_calsses=self.motion_calsses[index]
        #motion_calsses=0
        animal_class_list=self.animal_class_list[index]
        return vertices,motion_calsses,animal_class_list
    
    def get_motion_dic(self):
        return self.motion_dic
    def get_cls_dic(self):
        return self.class_dic


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    #print(batch.device())
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def vertice_collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    animaclsbatch = [b[2] for b in batch]
    bs=len(databatch)
    
    #print(labelbatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    animaclsbatchTensor = torch.as_tensor(animaclsbatch)
    xyz=torch.zeros(size=(bs,databatch[0].shape[0],databatch[0].shape[1],databatch[0].shape[2]))  #[B,L,N,3]
    for i in range(0,bs):
        xyz[i]=databatch[i]
    
    batch = {"xyz": xyz,"y":labelbatchTensor,"cls":animaclsbatchTensor}
    return batch


def smpl_collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    animaclsbatch = [b[2] for b in batch]
    bs=len(databatch)
    maskbatch=[b[3] for b in batch]
    temp_lenbatch = [len(b[0]) for b in batch]   #用来计算mask的长度，取决于batch里面最大的length
    lenbatch = [b[4] for b in batch]     #用来计算mask的True的长度
    
    #maskbatch= torch.tensor((b[3] for b in batch),dtype=bool)
    #print(maskbatch[0].shape)
    #print(maskbatch)
    
    
    #print(labelbatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    animaclsbatchTensor = torch.as_tensor(animaclsbatch)
    temp_lenbatchTensor = torch.as_tensor(temp_lenbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    databatchTensor = collate_tensors(databatch)
    
    max_len = max(temp_lenbatchTensor)
    #print(lenbatch)
    
    maskbatchTensor=torch.full(size=(len(maskbatch),max_len),fill_value=False)  #[B,L]
    for i in range(0,len(maskbatch)):
        #maskbatchTensor[i]=maskbatch[i]
        lengths=lenbatch[i]
        mask = torch.arange(max_len, device="cuda:0").expand(1, max_len) < lengths
        mask=mask.squeeze(0).to(bool)
        maskbatchTensor[i]=mask

    #maskbatchTensor = lengths_to_mask(lenbatchTensor)
    #maskbatchTensor=torch.tensor(np.array(maskbatch))
    
    
    batch = {"xyz": databatchTensor,"y":labelbatchTensor,"cls":animaclsbatchTensor,
              "mask": maskbatchTensor, "lengths": temp_lenbatchTensor}
    return batch


def ma2m_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    #print(databatch)
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    
    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting


    labelbatch = [b[1] for b in batch]
    animaclsbatch = [b[2] for b in batch]
    bs=len(databatch)
    maskbatch=[b[3] for b in batch]
    temp_lenbatch = [len(b[0]) for b in batch]   #用来计算mask的长度，取决于batch里面最大的length
    lenbatch = [b[4] for b in batch]     #用来计算mask的True的长度
    
    #maskbatch= torch.tensor((b[3] for b in batch),dtype=bool)
    #print(maskbatch[0].shape)
    #print(maskbatch)
    print(databatch)
    
    #print(labelbatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    animaclsbatchTensor = torch.as_tensor(animaclsbatch)
    temp_lenbatchTensor = torch.as_tensor(temp_lenbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    databatchTensor = collate_tensors(databatch)
    
    max_len = max(temp_lenbatchTensor)
    #print(lenbatch)
    
    maskbatchTensor=torch.full(size=(len(maskbatch),max_len),fill_value=False)  #[B,L]
    for i in range(0,len(maskbatch)):
        #maskbatchTensor[i]=maskbatch[i]
        lengths=lenbatch[i]
        mask = torch.arange(max_len, device="cuda:0").expand(1, max_len) < lengths
        mask=mask.squeeze(0).to(bool)
        maskbatchTensor[i]=mask
        
    #maskbatchTensor = lengths_to_mask(lenbatchTensor)
    #maskbatchTensor=torch.tensor(np.array(maskbatch))
    
    
    batch = {"xyz": databatchTensor,"y":labelbatchTensor,"cls":animaclsbatchTensor,
              "mask": maskbatchTensor, "lengths": temp_lenbatchTensor}
    return batch

def a2m_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    actionbatch = [b['action'] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)  #[B,nJoint,nfeature,nframe]
    lenbatchTensor = torch.as_tensor(lenbatch)    #[B]
    maskbatchTensor = lengths_to_mask(lenbatchTensor)  #[B,nframe]
    actionbatch= torch.as_tensor(actionbatch)   #[B]
    animaclsbatchTensor = torch.zeros(size=actionbatch.shape)  #[B]

    databatchTensor = databatchTensor.permute(0,3,1,2) #[B,nJoint,nfeature,nframe]->[B,nframe,nJoint,nfeature]

    # print(databatchTensor.shape)
    # print(lenbatchTensor.shape)
    # print(maskbatchTensor.shape)
    # print(actionbatch.shape)
    # print(animaclsbatchTensor.shape)
    # print(databatchTensor.shape)

    # print(databatchTensor.shape)
    # print(lenbatchTensor.shape)
    # print(maskbatchTensor.shape)
    # print(actionbatch.shape)
    # print(animaclsbatchTensor.shape)
    #print(databatchTensor.shape)

    batch = {"xyz": databatchTensor,"y":actionbatch,"cls":animaclsbatchTensor,
            "mask": maskbatchTensor, "lengths": lenbatchTensor}

    return batch


# def collate(batch):
#     databatch = [b[0] for b in batch]
#     labelbatch = [b[1] for b in batch]
#     lenbatch = [len(b[0][0][0]) for b in batch]

#     databatchTensor = collate_tensors(databatch)
#     labelbatchTensor = torch.as_tensor(labelbatch)
#     lenbatchTensor = torch.as_tensor(lenbatch)

#     maskbatchTensor = lengths_to_mask(lenbatchTensor)
#     batch = {"x": databatchTensor, "y": labelbatchTensor,
#              "mask": maskbatchTensor, "lengths": lenbatchTensor}
#     return batch

def get_dataset_collate(args):
    
    # print(args.pose_rep)
    # print(args.jointstype)
    if args.dataset=="uestc":
        args.num_classes=40
        
        if args.jointstype=="vertices":
            data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/uestc_cls_30frames_1024vertices.pkl")
            dataset=UESTC_VerticesDataset(data_path)   #num_class=40
        elif args.jointstype=="smpl":
            from _dataset.src.actiontomotion.uestc import UESTC
            dataset = UESTC(split= args.split ,num_frames=args.num_frames,
                                      pose_rep = "rot6d" ,translation = args.translation,glob = args.glob )
            # if args.pose_rep=="xyz":
            #     from _dataset.src.actiontomotion.uestc import UESTC
            #     dataset = UESTC(split= args.split ,num_frames=args.num_frames,
            #                           pose_rep = args.pose_rep ,translation = args.translation,glob = args.glob )
            #     #print("ggggg")
            #     # if args.split == "train":

                    
            #     #     #print(PROJECT_ROOT_DIR)
            #     #     data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/train_uestc_smpl_24points60frames.pkl")
            #     #     dataset=UESTC_smpl_VerticesDataset(args=args,data_path=data_path)   #num_class=40
            #     # elif args.split == "test":
            #     #     data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/test_uestc_smpl_24points60frames.pkl")
            #     #     dataset=UESTC_smpl_VerticesDataset(args=args,data_path=data_path)   #num_class=40
            # elif args.pose_rep=="rot6d":
            #     if args.split == "train":
            #         data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/train_rot6d_smpl_uestc_cls_60frames_1024vertices.pkl")
            #         dataset=UESTC_smpl_VerticesDataset(args=args,data_path=data_path)   #num_class=40
            #     elif args.split == "test":
            #         data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/test_rot6d_smpl_uestc_cls_60frames_1024vertices.pkl")
            #         dataset=UESTC_smpl_VerticesDataset(args=args,data_path=data_path)   #num_class=40
            
            # else :
            #     raise NotImplementedError("This dataset is not supported.")
        #elif args.jointstype=="smpl":
            
            
    elif args.dataset=="humanact12":
        args.num_classes=12
        if args.jointstype=="vertices":
            if args.num_animal_classes == 1 and args.pointcloud_in_order==True:
                data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/humanact12_vertices_1024points60frames.pkl")
            elif args.num_animal_classes == 3:
                data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/humanact12_3_vertices_1024points60frames.pkl")
            elif args.num_animal_classes == 1 and args.pointcloud_in_order==False:
                data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/humanact12_unorder1_vertices_1024points60frames.pkl")
            dataset=dateset_joint_VerticesDataset(args=args,data_path=data_path)
        elif args.jointstype=="smpl":
            from _dataset.src.actiontomotion.humanact12poses import HumanAct12Poses

            dataset = HumanAct12Poses(split= args.split ,num_frames=args.num_frames,
                                      pose_rep = "xyz" ,translation = args.translation,glob = args.glob )

            # dataset=HumanAct12_smpl_VerticesDataset(args=args,data_path="./_dataset/data/smpl_humanact12_cls_60frames_1024vertices.pkl")
        elif args.jointstype=="vertices+smpl":
            dataset=dateset_joint_plus_verticesDataset(args=args,data_path="./_dataset/data/humanact12_J_joint_vertices_1024points60frames.pkl")
        elif args.jointstype=="mocap_surface":
            dataset=HumanAct12_smpl_VerticesDataset(args=args,data_path="./_dataset/data/humanact12_inorder1_vertices_53points60frames.pkl")
    elif args.dataset=="4Dcompelte":
        dataset=_4DcompelteVerticesDataset("./_dataset/data/_4Dcomplete_puma_tiger_mix_30frames_1024vertices.pkl")
        args.num_classes=len(dataset.get_motion_dic())
        #args.num_classes=1
        args.num_animal_classes=len(dataset.get_cls_dic())
        
    elif args.dataset=="cmu":
        args.num_classes=8
        data_path=os.path.join(PROJECT_ROOT_DIR,"_dataset/data/cmu_100frames.pkl")
        dataset=dateset_joint_VerticesDataset(args=args,data_path=data_path)   #num_class=40
        
    if args.jointstype=="vertices":
        collate=smpl_collate
    else:
        collate=a2m_collate
    
    return dataset,collate