%% MAIN EVALUATION SCRIPT
clear;

if ~isfile('Trained_Models_Corrected.mat')
    error('Trained_Models_Corrected.mat not found. Run main_train_all.m first.');
end

load('Trained_Models_Corrected.mat');

disp('--- Evaluating corrected pipeline ---');

% ------------------------------------------------------------
% Stage 1 metrics
% ------------------------------------------------------------
stage1Metrics = evaluateStage1(stage1_Net, splits);

stage1ThresholdInfo = tuneStage1Threshold(stage1_Net, splits);

disp(' ');
disp('===================================================');
disp('STAGE 1 THRESHOLD-TUNED TEST METRICS');
disp('===================================================');
fprintf('Selected threshold: %.2f\n', stage1ThresholdInfo.BestThreshold);
fprintf('Validation best F1: %.4f\n', stage1ThresholdInfo.ValidationBestF1);
fprintf('Test accuracy:      %.4f\n', stage1ThresholdInfo.TestAccuracy);
fprintf('Test precision:     %.4f\n', stage1ThresholdInfo.TestPrecision);
fprintf('Test recall:        %.4f\n', stage1ThresholdInfo.TestRecall);
fprintf('Test F1:            %.4f\n', stage1ThresholdInfo.TestF1);
fprintf('Test AUROC:         %.4f\n', stage1ThresholdInfo.TestAUC);
fprintf('TP=%d, FP=%d, FN=%d, TN=%d\n', ...
    stage1ThresholdInfo.TP, ...
    stage1ThresholdInfo.FP, ...
    stage1ThresholdInfo.FN, ...
    stage1ThresholdInfo.TN);

disp(' ');
disp('===================================================');
disp('STAGE 1 CLASSIFICATION METRICS');
disp('===================================================');
disp(stage1Metrics);

% ------------------------------------------------------------
% Rebuild segmentation datastores
% ------------------------------------------------------------
segDS = makeSegmentationDatastores(datasetPath, splits, inputSizeSeg, M_prior_256);

% ------------------------------------------------------------
% Stage 2 threshold tuning on validation set
% ------------------------------------------------------------
thr_RGB = tuneSegmentationThreshold( ...
    unet_RGB, segDS.rgb.val, "RGB_UNet");

thr_Fixed = tuneSegmentationThreshold( ...
    unet_FixedPrior, segDS.physics.val, "FixedPrior_UNet");

thr_Learnable = tuneSegmentationThreshold( ...
    unet_LearnableTopology, segDS.physics.val, "LearnableTopology_UNet");

% ------------------------------------------------------------
% Stage 2 thresholded test evaluation
% ------------------------------------------------------------
metrics_RGB = evaluateSegmentationModel( ...
    unet_RGB, segDS.rgb.test, "RGB_UNet", thr_RGB.BestThreshold, false);

metrics_Fixed = evaluateSegmentationModel( ...
    unet_FixedPrior, segDS.physics.test, "FixedPrior_UNet", thr_Fixed.BestThreshold, false);

metrics_Learnable = evaluateSegmentationModel( ...
    unet_LearnableTopology, segDS.physics.test, "LearnableTopology_UNet", thr_Learnable.BestThreshold, false);

% ------------------------------------------------------------
% Ablation table
% ------------------------------------------------------------
modelNames = [
    "RGB U-Net Baseline"
    "Fixed-Prior U-Net"
    "Learnable Topology U-Net"
];

DefectIoU = [
    metrics_RGB.DefectIoU
    metrics_Fixed.DefectIoU
    metrics_Learnable.DefectIoU
];

MeanIoU = [
    metrics_RGB.MeanIoU
    metrics_Fixed.MeanIoU
    metrics_Learnable.MeanIoU
];

PixelAUROC = [
    metrics_RGB.PixelAUROC
    metrics_Fixed.PixelAUROC
    metrics_Learnable.PixelAUROC
];

Precision = [
    metrics_RGB.Precision
    metrics_Fixed.Precision
    metrics_Learnable.Precision
];

Recall = [
    metrics_RGB.Recall
    metrics_Fixed.Recall
    metrics_Learnable.Recall
];

F1 = [
    metrics_RGB.F1
    metrics_Fixed.F1
    metrics_Learnable.F1
];

Threshold = [
    metrics_RGB.Threshold
    metrics_Fixed.Threshold
    metrics_Learnable.Threshold
];

ablationTable = table(modelNames, Threshold, Precision, Recall, F1, DefectIoU, MeanIoU, PixelAUROC);

disp(' ');
disp('===================================================');
disp('STAGE 2 ABLATION RESULTS');
disp('===================================================');
disp(ablationTable);

writetable(ablationTable, 'Ablation_Table.csv');

% ------------------------------------------------------------
% Save metrics
% ------------------------------------------------------------
save('Evaluation_Results.mat', ...
    'stage1Metrics', ...
    'stage1ThresholdInfo', ...
    'thr_RGB', ...
    'thr_Fixed', ...
    'thr_Learnable', ...
    'metrics_RGB', ...
    'metrics_Fixed', ...
    'metrics_Learnable', ...
    'ablationTable');

disp('Evaluation complete. Results saved.');