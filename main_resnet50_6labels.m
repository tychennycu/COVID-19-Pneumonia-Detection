net = resnet50;
lgraph = layerGraph(net);

%% Read Image Dataclc
pwd = 'train';   %yourpath

%% check input size
inputSize = lgraph.Layers(1).InputSize;

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
Numberofclasses = 6;
lgraph = replaceLayer(lgraph,'fc1000',...
  fullyConnectedLayer(Numberofclasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fcNew'));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',...
  classificationLayer('Name','ClassificationNew'));

%% Specify Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Start Training Transfer Network
tic;
netTransfer = trainNetwork(imdsTrain,lgraph,options);
toc;
