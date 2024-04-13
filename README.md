<h2 align="center">
Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds for Tensor Decomposition Based Temporal Knowledge Graph Embedding
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
  <img alt="Static Badge" src="https://img.shields.io/badge/License-MIT-green">
</p>



## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.x
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the tkbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are downloaded, add them to the package data folder by running :
```
python tkbc/process.py
```


## Reproducing results



```
python tkbc/learner.py --dataset ICEWS14 --model TeAST (TComplEx & TNTcomplEx & TeLM(rank=121)) --rank 128 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TeAST (TComplEx & TNTcomplEx & TeLM(rank=121)) --rank 128 --emb_reg 1e-3 --time_reg 1
```





## License
MIT licensed, as found in the LICENSE file.


## Our Method

```
# Copyright (c) Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds for Tensor Decomposition Based Temporal Knowledge Graph Embedding
import torch

"""
Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds for Tensor Decomposition Based Temporal Knowledge Graph Embedding

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

```
