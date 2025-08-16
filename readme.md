# This repository contains the official implementation of the paper "Semi-supervised Multiscale Matching for SAR-Optical Image", submitted to AAAI 2026.

## 1. Data Preparation: 

Both datasets used in the paper are publicly available:

### SEN1-2 Dataset
Download from: https://mediatum.ub.tum.de/1436631
⚠️ Note: The direct download link on the website may be broken. However, you can still download the dataset via RSync, as detailed in the instructions on the site.

### QXS-SAROPT Dataset
Download from: https://github.com/yaoxu008/QXS-SAROPT
You may be required to complete a short survey to access the download link.


After downloading, unzip each dataset into the relative path './data/'.

For example, the SEN1-2 folder structure should look like this:
data/sen1-2/ROIs2017_winter
data/sen1-2/ROIs1970_fall
data/sen1-2/ROIs1868_summer
data/sen1-2/ROIs1158_spring

## 2. Environment Preparation

After creating a new conda environment, 
Run `pip install -r requirements.txt`

## 3. Semi-supervised Training

To train the model on the SEN1-2 dataset with only 1/16 labeled data (default setting), simply run: `'python ./train.py'`

You can switch to the QXSLAB dataset by commenting the code

```
    labeled_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(0, 16000))
    unlabeled_dataset = Sen12_dataset((256, 256), (192, 192), data_range=range(16000, 256000))
```

and uncommenting the code 

```
    #labeled_dataset = QXSLAB_dataset((256, 256), (192, 192), data_range=range(0, 20000))
    #unlabeled_dataset = QXSLAB_dataset((256, 256), (192, 192), data_range=range(1000, 16000))
```
in train.py

Alternatively, you can download the pretrained weights via the anonymous OSF link provided below:
https://osf.io/jrgca/?view_only=82a00ee861074effa3b2f65069f6c7cd

This model checkpoint was trained using the hyperparameter settings described in the paper.  
After downloading, place the file in the appropriate relative path: './model_checkpoints/'.

## 3. Evaluation

Run `'python ./eval.py'` to evaluate the RMSE and CMR of the model.