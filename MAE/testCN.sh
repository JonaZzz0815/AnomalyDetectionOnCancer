###
 # @Author: your name
 # @Date: 2022-01-08 12:13:30
 # @LastEditTime: 2022-01-09 08:35:50
 # @LastEditors: Please set LastEditors
 # @Description: 
 # @FilePath: /AnomalyDetectionOnCancer/MAE/testCN.sh
### 
#python3 pretrain.py --model_path vit-t-mae.pt --pretrain COVID
python3 train.py --pretrained_model_path vit-t-mae.pt --output_model_path vit-t-AE-from_scratch.pt --pretrain COVID
python3 train_classify.py --pretrained_model_path vit-t-mae.pt --output_model_path vit-t-classifier-from_scratch.pt

