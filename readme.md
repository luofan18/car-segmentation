# Code for Kaggle Carvana Image Masking Challenge.

The score is 0.996576 (dice coefficient) in private leader board. The best is 0.997332.

The neural network is trained using u-net-v5.ipynb.

The data_utils contains all codes to prepare data and perform data augmentation. 

The keras_custom implement custom metric and loss. (dice coefficient and boundary penalized loss)

Overall network architecture is as follow:
  
    
    
![Network architecture][architecture]

[architecture]: Architecture.png
