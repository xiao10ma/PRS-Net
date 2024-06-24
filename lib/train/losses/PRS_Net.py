import torch
import torch.nn as nn
from lib.utils import trans_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.sym_loss = SymmetryLoss(cfg.task_arg.gridBound, cfg.task_arg.gridSize)
        self.reg_loss = RegularizationLoss()

    def forward(self, batch):
        plane, quat = self.net(batch['voxel'])

        scalar_stats = {}
        loss = 0

        loss_sym_plane, loss_sym_quat = self.sym_loss(batch['sample'], batch['cp'], batch['voxel'], plane, quat)
        loss_reg_plane, loss_reg_quat = self.reg_loss(plane, quat, cfg.task_arg.reg_weight)
        
        scalar_stats.update({'loss_sym_plane': loss_sym_plane, 'loss_sym_quat': loss_sym_quat,
                             'loss_reg_plane': loss_reg_plane, 'loss_reg_quat': loss_reg_quat})
        loss += loss_sym_plane + loss_sym_quat + loss_reg_plane + loss_reg_quat

        scalar_stats.update({'loss': loss})

        return plane, quat, loss, scalar_stats

class RegularizationLoss(nn.Module):
    def __init__(self):
        super(RegularizationLoss, self).__init__()
        self.identity_matrix = torch.eye(3).cuda()

    def forward(self, plane=None, quat=None, weight=1):
        reg_rot_loss = torch.Tensor([0]).cuda()
        reg_plane_loss = torch.Tensor([0]).cuda()
        
        if plane is not None:
            normalized_planes = [trans_utils.normalize(p[:, 0:3]).unsqueeze(2) for p in plane]
            plane_matrix = torch.cat(normalized_planes, 2)
            plane_matrix_t = torch.transpose(plane_matrix, 1, 2)
            reg_plane_loss = (torch.matmul(plane_matrix, plane_matrix_t) - self.identity_matrix).pow(2).sum(2).sum(1).mean() * weight

        if quat is not None:
            normalized_quats = [q[:, 1:4].unsqueeze(2) for q in quat]
            quat_matrix = torch.cat(normalized_quats, 2)
            quat_matrix_t = torch.transpose(quat_matrix, 1, 2)
            reg_rot_loss = (torch.matmul(quat_matrix, quat_matrix_t) - self.identity_matrix).pow(2).sum(2).sum(1).mean() * weight

        return reg_plane_loss, reg_rot_loss

class SymmetryLoss(nn.Module):
    def __init__(self, grid_bound, grid_size):
        super(SymmetryLoss, self).__init__()
        self.grid_size = grid_size
        self.grid_bound = grid_bound
        self.calculate_distance = CalculateDistance.apply

    def forward(self, points, control_points, voxel, plane=None, quat=None, weight=1):
        reflective_loss = torch.Tensor([0]).cuda()
        rotational_loss = torch.Tensor([0]).cuda()
        
        for p in plane:
            reflected_points = trans_utils.planesymTransform(points, p)
            reflective_loss += self.calculate_distance(reflected_points, control_points, voxel, self.grid_size)
        
        for q in quat:
            rotated_points = trans_utils.rotsymTransform(points, q)
            rotational_loss += self.calculate_distance(rotated_points, control_points, voxel, self.grid_size)

        return reflective_loss, rotational_loss

def point_closest_cell_index(points, grid_bound=0.5, grid_size=32):
    grid_min = -grid_bound + grid_bound / grid_size
    grid_max = grid_bound - grid_bound / grid_size
    indices = (points - grid_min) * grid_size / (2 * grid_bound)
    indices = torch.round(torch.clamp(indices, min=0, max=grid_size-1))
    return indices    

class CalculateDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transformed_points, control_points, voxel, grid_size, weight=1):
        nearest_bin = point_closest_cell_index(transformed_points)
        index = torch.matmul(nearest_bin, torch.cuda.FloatTensor([grid_size**2, grid_size, 1])).long()
        mask = 1 - torch.gather(voxel.view(-1, grid_size**3), 1, index)
        index = index.unsqueeze(2).repeat(1, 1, 3)
        mask = mask.unsqueeze(2).repeat(1, 1, 3)
        closest_points = torch.gather(control_points, 1, index)
        ctx.constant = weight
        distance = (transformed_points - closest_points) * mask
        ctx.save_for_backward(distance)
        return torch.mean(torch.sum(torch.sum(torch.pow(distance, 2), 2), 1)) * weight

    @staticmethod
    def backward(ctx, grad_output):
        distance, = ctx.saved_tensors
        grad_transformed_points = 2 * distance * ctx.constant / distance.shape[0]
        return grad_transformed_points, None, None, None, None