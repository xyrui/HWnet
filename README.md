# Learning An Explicit Weighting Scheme for Adapting Complex HSI Noise (CVPR2021)
<center>Xiangyu Rui<sup>1</sup>, Xiangyong Cao<sup>1</sup>, Qi Xie<sup>1</sup>, Zongsheng Yue<sup>1</sup>, Qian Zhao<sup>1</sup>, Deyu Meng<sup>1,2</sup></center>

<sup>1</sup>Xi’an Jiaotong University; <sup>2</sup>Pazhou Lab, Guangzhou

<center>a</center>

[Main paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Rui_Learning_an_Explicit_Weighting_Scheme_for_Adapting_Complex_HSI_Noise_CVPR_2021_paper.pdf)

[Supplement material](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Rui_Learning_an_Explicit_CVPR_2021_supplemental.pdf)


## Prepare data

### Training dataset
1. Please download CAVE DATAset from https://www.cs.columbia.edu/CAVE/databases/multispectral/ for training. The image size is of 512\*512\*31.

2. Crop your own training dataset. I randomly select 20 images for training and crop about 10000 patches of size 96\*96\*20. (This step is implemented by MATLAB code and I did not add the code here. You could freely choose your favourite way.) 

3. Save the training dataset in path "dataroot"(your own data path).

### Testing dataset
Please refer to "data" file for creating your own test data. The noise generation methods in "data/utils" file are in consistent with those in "lib.py".  

## Training and testing
Plean refer to "train_hwnet.py" and "test.py" for training and testing HWLRMF. More test codes for NAILRMA, NGmeet, LLRT and their weighted versions will be uploaded soon.

## Citation
If you are interested in our work, please cite  

```
@InProceedings{Rui_2021_CVPR, 
    author    = {Rui, Xiangyu and Cao, Xiangyong and Xie, Qi and Yue, Zongsheng and Zhao, Qian and Meng, Deyu},
    title     = {Learning an Explicit Weighting Scheme for Adapting Complex HSI Noise},    
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},    
    month     = {June},    
    year      = {2021},    
    pages     = {6739-6748}    
}
```
## Contacts
If you have any questions, please contract me by <xyrui@outlook.com> or <rxy14789653@stu.xjtu.edu.cn>.
