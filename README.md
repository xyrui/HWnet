# HWnet
Learning An Explicit Weighting Scheme for Adapting Complex HSI Noise (CVPR2021)

## Training DATASET
1. Please download CAVE DATAset from https://www.cs.columbia.edu/CAVE/databases/multispectral/ for training. The image size is of 512\*512\*31.

2. Crop your own training dataset. I randomly select 20 images for training and crop about 10000 patches of size 96\*96\*20. (This step is implemented by MATLAB code and I did not add the code here. You could freely choose your favourite way.) 

3. Save the training dataset in path "dataroot"(your own data path).

## Testing DATASET
Please refer to "data" file for creating your own test data. The noise generation methods in "data/utils" file are in consistent with those in "lib.py". 
