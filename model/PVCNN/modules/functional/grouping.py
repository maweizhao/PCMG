from torch.autograd import Function

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)


from model.PVCNN.modules.functional.backend import _backend

__all__ = ['grouping']


class Grouping(Function):
    @staticmethod
    def forward(ctx, features, indices):
        """
        :param ctx:
        :param features: features of points, FloatTensor[B, C, N]
        :param indices: neighbor indices of centers, IntTensor[B, M, U], M is #centers, U is #neighbors
        :return:
            grouped_features: grouped features, FloatTensor[B, C, M, U]
        """
        features = features.contiguous()
        indices = indices.contiguous()
        ctx.save_for_backward(indices)
        ctx.num_points = features.size(-1)
        return _backend.grouping_forward(features, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_features = _backend.grouping_backward(grad_output.contiguous(), indices, ctx.num_points)
        return grad_features, None


grouping = Grouping.apply
