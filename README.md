# CycleGAN Study

## reference
Paper: https://arxiv.org/abs/1703.10593
github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Getting Started
### How to install
```    
git clone https://github.com/youngjun0627/cyclegan_study.git
cd cyclegan_study
```
### Change to data-path in config file.
```   
vi config.py
```
2. change the value of 'dataroot' key.

## Environment
- Ubuntu 20.04
- cuda version 11.2 (RTX2080)
- Python 3.8.8

## Requirements
- torch 1.10
- tqdm 4.59.0
- scikit-learn 0.22
- cv2
- albumentations 1.0.0

## Procedure
 TODO!

### Generate dataset
Using the following shell script.
```
#!/bin/bash                                                                                                        

FILE=$1
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebr    a" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" &&     $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_pho    tos" ]]; then
     echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, uk    iyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
     exit 1
 fi
 
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE
mkdir -p ./datasets
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE

# Adapt to project expected directory heriarchy
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"                                                   
```

### Train
 TODO!
 1. We must find metric method accurately to estimate the model.
 2. Show graph containing loss & metric.
 3. Improve model as adjusting learning parameters.
 
### Test
 TODO !
 1. Show generated image by our model.
 2. And, compare generated image and original image.

### Contribution
1. Create new branch.
2. In consideration of update, fetch main branch and merge.
3. If Have finished the working, request your branch to main branch.

