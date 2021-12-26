### 1. Only Encoder for pre-training, No Decoder.  
loss function = clustering loss. 
  
### 2. Only Encoder for pre-training, New Decoder created in the downstream task.  
loss function = reconstruction loss + clustering loss. 
  
### 3. Encoder & Decoder for pre-training.
loss function = reconstruction loss + clustering loss. 
  
  
## Datasets
### pre-training data: (the third dataset is enough.)
1. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 
2. head CT: https://www.kaggle.com/felipekitamura/head-ct-hemorrhage.  
3. COVID-19：https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset (all)
### downstream data: (the first dataset is enough.)
1. COVID-CT: https://github.com/UCSD-AI4H/COVID-CT. 
