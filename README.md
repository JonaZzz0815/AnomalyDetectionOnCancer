# Outlier Detection On COVID-19
## dataset:
COVID-19 CT Images: https://github.com/UCSD-AI4H/COVID-CT  
  
## Resources:
https://github.com/yzhao062/anomaly-detection-resources  
ViT: https://github.com/lucidrains/vit-pytorch  
case with AE: http://sofasofa.io/tutorials/anomaly_detection/  
case with AE: https://zhuanlan.zhihu.com/p/260882741  
  
### Theme: Semi-supervised anomaly detection with Deep Clustering In AutoEncoder.  
Our Method:  
1. Adaptive sampling. 
2. Dropout / random ensemble. 
    1. dropout refer to: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
3. (M)AE+k-means. MAE :https://github.com/pengzhiliang/MAE-pytorch
4. pre-training. 
5. Downstream Task: anomaly detection by reconstruction or two-classes classification
6. loss=reconstruction loss+ cluster loss（k-means). 

### How to run
cd MAE  
python pretain.py  
python xxtrain.py  
### Models
pretrain on NonCOVID with classifier:https://drive.google.com/file/d/1jvxqTLnEE1ITJNSCrl1rBy38Zc1WWC43/view?usp=sharing
pretrain on NonCOVID with Reconstruction:https://drive.google.com/file/d/1_bMWvhYBSuety2z41PXDyhQNRYdXtStj/view?usp=sharing
pretrain on COVID with classifier:https://drive.google.com/file/d/1fzyclnPndkM-b_zJZfnnb5gAd3-e5nVL/view?usp=sharing
pretrain on COVID with Reconstruction:https://drive.google.com/file/d/1UI1JAO5jxAHx4Hv5GipIbBHEDJWJXZDZ/view?usp=sharing

### other methods' implementation:
see in SOTA file

### Result
| pretrain | downstream     | accuracy | ROC AUC  | PRC AUC |
|----------|----------------|----------|----------|---------|
| NonCOVID | Reconstruction | 0.848    | 0.525    | 0.923   |
| NonCOVID | Classifier     | 0.712    | 0.586    | 0.917   |
| COVID    | Reconstruction | 0.834    | 0.642    | 0.936   |
| COVID    | Classifier     | 0.640    | 0.543    | 0.903   |

