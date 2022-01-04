# Outlier Detection On COVID-19
## Possible dataset:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)  
corresponding code :https://github.com/AFAgarap/wisconsin-breast-cancer  
COVID-19 CT Images: https://github.com/UCSD-AI4H/COVID-CT  
https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets  
  
## Resources:
https://github.com/yzhao062/anomaly-detection-resources  
ViT: https://github.com/lucidrains/vit-pytorch  
case with AE: http://sofasofa.io/tutorials/anomaly_detection/  
case with AE: 自编码器AutoEncoder解决异常检测问题（手把手写代码） - 数据如琥珀的文章 - 知乎 https://zhuanlan.zhihu.com/p/260882741  

## Paper with code 
image :anomaly detextion on cifar：https://paperswithcode.com/sota/anomaly-detection-on-one-class-cifar-10
data : 
Deep Anomaly Detection with Deviation Networks:https://arxiv.org/pdf/1911.08623v1.pdf 
code:https://paperswithcode.com/paper/deep-anomaly-detection-with-deviation
## other resource（maybe usefull when writing report
cs229 course page: http://cs229.stanford.edu/proj2021spr/
  
### Theme: Semi-supervised anomaly detection with Deep Clustering In AutoEncoder.  
Our Method:  
1. Adaptive sampling. 
2. Dropout / random ensemble. 
    1. dropout refer to: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
3. (M)AE+k-means. MAE :https://github.com/pengzhiliang/MAE-pytorch
4. pre-training. 
    1. pre-training refer to: https://zhangkaifang.blog.csdn.net/article/details/89320108?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link. 
    2. MAE self-supervision pretraining： 别再无聊地吹捧了，一起来动手实现 MAE(Masked Autoencoders Are Scalable Vision Learners) 玩玩吧！ - CW不要無聊的風格的文章 - 知乎
https://zhuanlan.zhihu.com/p/439554945  
5. loss=重构loss+聚类loss（k-means). 
### How to run
cd MAE  
python pretain.py  
python xxtrain.py  
