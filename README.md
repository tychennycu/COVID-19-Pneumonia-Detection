# COVID-19-Pneumonia-Detection
We trained ResNet50 and DenseNet201 to detect chest x-ray(CXR) images in MATLAB 2021a, they can be classified as negative, typical, atypical CXR images.<br>
If you want to use our trained model directly, you can start from step 3 below.

## Requirement
Image Processing Toolbox<br>
Deep Learning Toolbox

## Usage
### Step 1

Organize CXR images into files, and the name of the files can be recognized as labels when training the net.

### Step2

Use main_densenet201.m or main_resnet50_6labels.m to train the model.<br>
When you finished training, you can use the code below to save the net.
```
save netTransfer
```
### Step3
Put the CXR images into the folder named "valid", and use the code below to detect pneumonia.
```
%netTransfer = load('DenseNet201.mat');
%netTransfer = netTransfer.net;

imds_valid = imageDatastore('valid','FileExtensions','.dcm','ReadFcn',@dicompreprocess);
Pred = classify(netTransfer,imds_valid);
```
