import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from pdb import set_trace as brk

class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self,gts,preds):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.mean(mins)
		mins, _ = torch.min(P, 2)  #[B*L,N]
		loss_2 = torch.mean(mins)
		#print(mins.shape)
		return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P


class Chamfer_Density_Loss(nn.Module):
    
	def __init__(self):
		super(Chamfer_Density_Loss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self,gts,preds,density_k):
		dist1 = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(dist1, 1)
		loss_1 = torch.mean(mins)
		mins, _ = torch.min(dist1, 2)  #[B*L,N]
		loss_2 = torch.mean(mins)

		#print(mins.shape)
		charmfer_loss=loss_1 + loss_2
  
		dist2=self.batch_pairwise_dist(gts,gts)  #[L*B,N,N]
  
		val_1,_=dist1.topk(k=density_k,dim=2,largest=False,sorted=True) #[L*B,N,k]
		val_2,_=dist2.topk(k=density_k,dim=2,largest=False,sorted=True) #[L*B,N,k]
        
		density_loss=F.mse_loss(val_1,val_2, reduction='mean')
		

		return charmfer_loss,density_loss


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P