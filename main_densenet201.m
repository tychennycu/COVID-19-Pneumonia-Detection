net = densenet201;

%% Read Image Dataclc
pwd = 'train';  %yourpath

%% check input size
inputSize = net.Layers(1).InputSize;

%% Create Image Datastore
imds = dcm2datastore(pwd,'.dcm',0);

%% Count Number of Images for Each Label
labelCount = countEachLabel(imds);
labelCount = labelCount.Count;
min_labelCount = min(labelCount);

%% Specify Training and Validation Sets
train_ratio = 0.7;
numTrainFiles = fix(min_labelCount*train_ratio);
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Replace Final Layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

layers = [
   layersTransfer
   fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) %可能需要調整參數
   softmaxLayer
   classificationLayer];

%% Specify Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Start Training Transfer Network
tic;
netTransfer = trainNetwork(imdsTrain,layers,options);
toc;
