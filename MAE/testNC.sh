
###
 # @Author: your name
 # @Date: 2022-01-08 15:38:27
 # @LastEditTime: 2022-01-09 11:23:31
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /AnomalyDetectionOnCancer/MAE/testNC.sh
### 
#python3 pretrain.py --model_path vit-t-mae_2.pt --pretrain NonCOVID
python3 train.py --pretrained_model_path vit-t-mae_2.pt --output_model_path vit-t-AE-from_scratch_2.pt --pretrain NonCOVID
#python3 train_classify.py --pretrained_model_path vit-t-mae_2.pt --output_model_path vit-t-classifier-from_scratch_2.pt

