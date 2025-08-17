# CAN
Replication package for the paper entitled: Automatic Identification of Extrinsic Bug Reports for Just-In-Time Bug Prediction


### Data Preparation


### Environment Settings
* GPU: Nvidia GeForce RTX 4090D 24G
* CPU: AMD EPYC 9754
* OS: Ubuntu 20.04
* RAM: 60 GB

```
git clone https://github.com/Hugo-Liang/CAN.git
cd CAN
conda create -n CAN python=3.10
conda activate CAN
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


### Train and Evaluate CAN

```python train_CNN_KAN.py```


### Get Involved
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.

