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

**We provide the downloaded data at `tkbc/src_data`**

Once the datasets are [downloaded](https://github.com/facebookresearch/tkbc/blob/main/tkbc/scripts/download_data.sh), go to the tkbc/ folder and add them to the package data folder by running :



```
python tkbc/process.py
```
This will create the files required to compute the filtered metrics.

## Reproducing results



```
# ICEWS14
python tkbc/learner.py --dataset ICEWS14 --model TeAST --rank 128 --emb_reg 1e-2 --time_reg 1e-2
python tkbc/learner.py --dataset ICEWS14 --model TComplEx  --rank 128 --emb_reg 1e-2 --time_reg 1e-2
python tkbc/learner.py --dataset ICEWS14 --model TNTcomplEx  --rank 128 --emb_reg 1e-2 --time_reg 1e-2
python tkbc/learner.py --dataset ICEWS14 --model TeLM --rank 121 --emb_reg 1e-2 --time_reg 1e-2

# ICEWS05-15
python tkbc/learner.py --dataset ICEWS05-15 --model TeAST --rank 128 --emb_reg 1e-3 --time_reg 1
python tkbc/learner.py --dataset ICEWS05-15 --model TComplEx --rank 128 --emb_reg 1e-3 --time_reg 1
python tkbc/learner.py --dataset ICEWS05-15 --model TNTcomplEx --rank 128 --emb_reg 1e-3 --time_reg 1
python tkbc/learner.py --dataset ICEWS05-15 --model TeLM --rank 121 --emb_reg 1e-3 --time_reg 1
```





## License
MIT licensed, as found in the LICENSE file.


### Acknowledgement
We refer to the code of [TNTComplEx](https://github.com/facebookresearch/tkbc) and [TeLM](https://github.com/soledad921/TeLM). Thanks for their great contributions!
