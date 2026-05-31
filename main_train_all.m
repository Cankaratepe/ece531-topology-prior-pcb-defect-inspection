%% MAIN TRAINING SCRIPT
clear; clc; close all;

disp('--- Starting Corrected Training Pipeline ---');

rng(42); % Reproducibility

datasetPath = fullfile(pwd, 'transistor');
inputSizeStage1 = [224 224 3];
inputSizeSeg = [256 256];

% ------------------------------------------------------------
% 1. Create clean train/validation/test splits
% ------------------------------------------------------------
splits = createSplits_MVTec(datasetPath);
%splits = addSyntheticTrainingData(splits, datasetPath);

% ------------------------------------------------------------
% 2. Create topology / physics prior
% ------------------------------------------------------------
M_prior = makePhysicsPrior([1024 1024]);
M_prior_256 = imresize(M_prior, inputSizeSeg);

% ------------------------------------------------------------
% 3. Train Stage 1 ResNet-18 classifier
% ------------------------------------------------------------
disp('Training Stage 1 ResNet-18 classifier...');
stage1_Net = trainStage1_ResNet(splits, inputSizeStage1);

% ------------------------------------------------------------
% 4. Create segmentation datastores
% ------------------------------------------------------------
segDS = makeSegmentationDatastores(datasetPath, splits, inputSizeSeg, M_prior_256);

% ------------------------------------------------------------
% 5. Train Stage 2 baseline RGB U-Net
% ------------------------------------------------------------
disp('Training Stage 2 RGB U-Net baseline...');
unet_RGB = trainUNet_RGB(segDS);

% ------------------------------------------------------------
% 6. Train Stage 2 fixed-prior U-Net
% ------------------------------------------------------------
disp('Training Stage 2 Fixed-Prior U-Net...');
unet_FixedPrior = trainUNet_FixedPrior(segDS);

% ------------------------------------------------------------
% 7. Train Stage 2 learnable topology-conditioned attention U-Net
% ------------------------------------------------------------
disp('Training Stage 2 Learnable Topology Attention U-Net...');
unet_LearnableTopology = trainUNet_LearnableTopology(segDS);

% ------------------------------------------------------------
% 8. Save everything
% ------------------------------------------------------------
save('Trained_Models_Corrected.mat', ...
    'stage1_Net', ...
    'unet_RGB', ...
    'unet_FixedPrior', ...
    'unet_LearnableTopology', ...
    'M_prior_256', ...
    'datasetPath', ...
    'splits', ...
    'inputSizeSeg');

disp('Training complete. Run main_evaluate_all.m next.');