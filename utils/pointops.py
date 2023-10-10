import torch
import torch.nn as nn


class Gen_QueryAndGroupXYZ(nn.Module):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """
    def __init__(self, radius=None, nsample=32, use_xyz=True):
        super(Gen_QueryAndGroupXYZ, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    #def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None, features: torch.Tensor = None, idx: torch.Tensor = None) -> torch.Tensor:
    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
               # idxs: (b, n)
        output: new_features: (b, c+3, m, nsample)
              #  grouped_idxs: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        #if idx is None:
        if self.radius is not None:
            idx = ballquery(self.radius, self.nsample, xyz, new_xyz)
        else:
            idx = knnquery(self.nsample, xyz, new_xyz)  # (b, m, nsample)
        xyz_trans = xyz.transpose(1, 2).contiguous()    # BxNx3 -> Bx3xN
        
        grouped_xyz = grouping(xyz_trans, idx)  # (b, 3, m, nsample)

        return grouped_xyz