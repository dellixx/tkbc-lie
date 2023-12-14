import torch

"""
Mitigating Heterogeneity among Factor Tensor via Lie Group Manifolds for Temporal Knowledge Graph Embedding

"""
class MatrixOperationsLie:
    def __init__(self, mat_n):
        """
        Initialize the MatrixOperations class.
        """
        self.mat_n = mat_n

    def map_to_lie(self, embeddings):
        """
        Maps the embeddings to a Lie group representation.
        """
        a, b, c, d = embeddings.chunk(4, dim=-1)
        
        a = torch.cos(a)
        b = -torch.sin(b)
        c = torch.sin(c)
        d = torch.cos(d)
        
        rotation_matrix = torch.cat((a, b, c, d), dim=-1)
        rotation_matrix = self.matrix_logarithm(rotation_matrix)
        return rotation_matrix

    def matrix_logarithm(self, R_batch):
        """
        Computes the matrix logarithm of a batch of rotation matrices.
        """
        R_batch = R_batch.contiguous().view(-1, self.mat_n, self.mat_n)
        R_batch = R_batch / R_batch.norm(dim=(1, 2), keepdim=True)
        traces = R_batch.diagonal(dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos((traces - 1) / 2)
        epsilon = 1e-8
        sin_theta = torch.sin(theta) + epsilon
        close_to_zero = theta < epsilon
        log_R_batch = torch.zeros_like(R_batch)
        log_R_batch[~close_to_zero] = theta[~close_to_zero].unsqueeze(-1).unsqueeze(-1) \
                                      / (2 * sin_theta[~close_to_zero].unsqueeze(-1).unsqueeze(-1)) \
                                      * (R_batch[~close_to_zero] - R_batch[~close_to_zero].transpose(-2, -1))

        assert not torch.isnan(log_R_batch).any(), "log_R_batch contains nan values."
        return log_R_batch.contiguous().view(-1, self.mat_n * self.mat_n)
