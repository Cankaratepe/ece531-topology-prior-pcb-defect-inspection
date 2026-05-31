%% GENERATE SELECTED DEFECT-TYPE QUALITATIVE FIGURES
% Selects one example per defect type based on Fixed-Prior U-Net IoU.
%
% selectionMode:
%   "best"   -> visually strongest examples
%   "median" -> more representative examples

clear;

selectionMode = "best";   % Change to "median" for more conservative reporting

if ~isfile('Trained_Models_Corrected.mat')
    error('Trained_Models_Corrected.mat not found. Run main_train_all.m first.');
end

if ~isfile('Evaluation_Results.mat')
    error('Evaluation_Results.mat not found. Run main_evaluate_all.m first.');
end

load('Trained_Models_Corrected.mat');
load('Evaluation_Results.mat');

outputDir = fullfile(pwd, 'Paper_Figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

defectTypes = {'bent_lead', 'cut_lead', 'damaged_case', 'misplaced'};
numExamples = numel(defectTypes);

selectedImageFiles = cell(numExamples, 1);
selectedMaskFiles = cell(numExamples, 1);
selectedIoU = zeros(numExamples, 1);

fprintf('Selecting %s examples by Fixed-Prior U-Net IoU...\n', selectionMode);

for i = 1:numExamples
    defectType = defectTypes{i};

    [imgFile, maskFile, iouVal] = selectExampleByFixedIoU( ...
        datasetPath, ...
        splits, ...
        defectType, ...
        inputSizeSeg, ...
        M_prior_256, ...
        unet_FixedPrior, ...
        metrics_Fixed.Threshold, ...
        selectionMode);

    selectedImageFiles{i} = imgFile;
    selectedMaskFiles{i} = maskFile;
    selectedIoU(i) = iouVal;

    fprintf('%s selected, IoU = %.4f\n', defectType, iouVal);
end

%% Main qualitative figure
fig = figure('Name', 'Selected Defect-Type Qualitative Comparison', ...
    'Position', [100, 100, 1700, 900]);

% Keep the figure clean: do not include the gate in the main comparison
tiledlayout(5, numExamples, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

rowNames = {
    'Input'
    'Ground Truth'
    'RGB U-Net'
    'Fixed-Prior U-Net'
    'Learnable Topology U-Net'
};

for i = 1:numExamples

    defectType = defectTypes{i};

    imgOriginal = imread(selectedImageFiles{i});
    gtOriginal = imread(selectedMaskFiles{i});

    imgRGB = im2double(imresize(imgOriginal, inputSizeSeg));
    gtMask = imresize(gtOriginal, inputSizeSeg, 'nearest') > 0;

    img4 = cat(3, imgRGB, M_prior_256);

    [~, scoreRGB] = predictDefectScore(imgRGB, unet_RGB);
    [~, scoreFixed] = predictDefectScore(img4, unet_FixedPrior);
    [~, scoreLearnable] = predictDefectScore(img4, unet_LearnableTopology);

    predRGB = bwareaopen(scoreRGB >= metrics_RGB.Threshold, 10);
    predFixed = bwareaopen(scoreFixed >= metrics_Fixed.Threshold, 10);
    predLearnable = bwareaopen(scoreLearnable >= metrics_Learnable.Threshold, 10);

    nexttile(i);
    imshow(imgRGB);
    title(sprintf('%s\nFixed IoU = %.2f', strrep(defectType, '_', ' '), selectedIoU(i)), ...
        'FontWeight', 'bold', ...
        'Interpreter', 'none');

    if i == 1
        ylabel(rowNames{1}, 'FontWeight', 'bold');
    end

    nexttile(i + numExamples);
    imshow(gtMask);
    if i == 1
        ylabel(rowNames{2}, 'FontWeight', 'bold');
    end

    nexttile(i + 2*numExamples);
    imshow(makeOverlay(imgRGB, predRGB));
    if i == 1
        ylabel(rowNames{3}, 'FontWeight', 'bold');
    end

    nexttile(i + 3*numExamples);
    imshow(makeOverlay(imgRGB, predFixed));
    if i == 1
        ylabel(rowNames{4}, 'FontWeight', 'bold');
    end

    nexttile(i + 4*numExamples);
    imshow(makeOverlay(imgRGB, predLearnable));
    if i == 1
        ylabel(rowNames{5}, 'FontWeight', 'bold');
    end
end

sgtitle(sprintf('Qualitative Defect Localization by Defect Type (%s examples)', selectionMode), ...
    'FontSize', 16, ...
    'FontWeight', 'bold');

saveFigureCompat(fig, fullfile(outputDir, ['Fig_DefectType_Selected_' char(selectionMode) '.pdf']));
saveFigureCompat(fig, fullfile(outputDir, ['Fig_DefectType_Selected_' char(selectionMode) '.png']));

%% Separate topology-gate figure only for lead-critical visualization
figGate = figure('Name', 'Topology Gate Visualization', ...
    'Position', [100, 100, 1700, 520]);

tiledlayout(3, numExamples, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for i = 1:numExamples

    defectType = defectTypes{i};

    imgOriginal = imread(selectedImageFiles{i});
    gtOriginal = imread(selectedMaskFiles{i});

    imgRGB = im2double(imresize(imgOriginal, inputSizeSeg));
    gtMask = imresize(gtOriginal, inputSizeSeg, 'nearest') > 0;
    img4 = cat(3, imgRGB, M_prior_256);

    [~, scoreFixed] = predictDefectScore(img4, unet_FixedPrior);

    predFixed = bwareaopen(scoreFixed >= metrics_Fixed.Threshold, 10);

    predFixedGated = applyTopologyGate( ...
        scoreFixed, ...
        metrics_Fixed.Threshold, ...
        M_prior_256);

    nexttile(i);
    imshow(gtMask);
    title(strrep(defectType, '_', ' '), ...
        'FontWeight', 'bold', ...
        'Interpreter', 'none');

    nexttile(i + numExamples);
    imshow(makeOverlay(imgRGB, predFixed));
    if i == 1
        ylabel('Fixed-Prior', 'FontWeight', 'bold');
    end

    nexttile(i + 2*numExamples);
    imshow(makeOverlay(imgRGB, predFixedGated));
    if i == 1
        ylabel('Fixed + Gate', 'FontWeight', 'bold');
    end
end

sgtitle('Topology Gate Visualization: Useful for Lead-Critical Defects but May Suppress Body Defects', ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

saveFigureCompat(figGate, fullfile(outputDir, ['Fig_TopologyGate_' char(selectionMode) '.pdf']));
saveFigureCompat(figGate, fullfile(outputDir, ['Fig_TopologyGate_' char(selectionMode) '.png']));

disp('Selected qualitative figures generated.');

%% ============================================================
% Local helper functions
% ============================================================

function [selectedImg, selectedMask, selectedIoU] = selectExampleByFixedIoU( ...
    datasetPath, splits, defectType, inputSizeSeg, M_prior_256, net, threshold, selectionMode)

    testFiles = splits.stage2.testImageFiles;

    candidates = {};

    for k = 1:numel(testFiles)
        thisFile = testFiles{k};

        if contains(thisFile, [filesep defectType filesep])
            candidates{end+1,1} = thisFile; %#ok<AGROW>
        end
    end

    % If the clean split does not contain that defect type, use all images
    % from that folder only for visualization.
    if isempty(candidates)
        warning('No %s image found in Stage 2 test split. Falling back to dataset folder.', defectType);
        folder = fullfile(datasetPath, 'test', defectType);
        imds = imageDatastore(folder);
        candidates = imds.Files;
    end

    ious = zeros(numel(candidates), 1);
    maskFiles = cell(numel(candidates), 1);

    for k = 1:numel(candidates)
        imgFile = candidates{k};

        [~, name, ~] = fileparts(imgFile);
        maskFile = fullfile(datasetPath, 'ground_truth', defectType, [name '_mask.png']);
        maskFiles{k} = maskFile;

        imgOriginal = imread(imgFile);
        gtOriginal = imread(maskFile);

        imgRGB = im2double(imresize(imgOriginal, inputSizeSeg));
        gtMask = imresize(gtOriginal, inputSizeSeg, 'nearest') > 0;

        img4 = cat(3, imgRGB, M_prior_256);

        [~, scoreFixed] = predictDefectScore(img4, net);

        predFixed = scoreFixed >= threshold;
        predFixed = bwareaopen(predFixed, 10);

        TP = sum(predFixed(:) & gtMask(:));
        FP = sum(predFixed(:) & ~gtMask(:));
        FN = sum(~predFixed(:) & gtMask(:));

        ious(k) = TP / max(TP + FP + FN, eps);
    end

    switch string(selectionMode)
        case "best"
            [selectedIoU, idx] = max(ious);

        case "median"
            [sortedIoU, sortIdx] = sort(ious, 'ascend');
            midIdx = ceil(numel(sortedIoU)/2);
            selectedIoU = sortedIoU(midIdx);
            idx = sortIdx(midIdx);

        otherwise
            error('Unknown selectionMode. Use "best" or "median".');
    end

    selectedImg = candidates{idx};
    selectedMask = maskFiles{idx};
end

function overlay = makeOverlay(imgRGB, predBinary)

    predBinary = logical(predBinary);

    predCat = categorical(predBinary, ...
        [false true], ...
        ["Background", "Defect"]);

    overlay = labeloverlay(imgRGB, predCat, ...
        'IncludedLabels', "Defect", ...
        'Colormap', [1 1 0], ...
        'Transparency', 0.55);
end