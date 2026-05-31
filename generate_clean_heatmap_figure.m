%% GENERATE CLEAN DEFECT-TYPE HEATMAP FIGURE
clear;

if ~isfile('Trained_Models_Corrected.mat')
    error('Trained_Models_Corrected.mat not found.');
end

if ~isfile('Evaluation_Results.mat')
    error('Evaluation_Results.mat not found.');
end

load('Trained_Models_Corrected.mat');
load('Evaluation_Results.mat');

outputDir = fullfile(pwd, 'Paper_Figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

defectTypes = {'bent_lead', 'cut_lead', 'damaged_case', 'misplaced'};
defectTitles = {'Bent Lead', 'Cut Lead', 'Damaged Case', 'Misplaced'};

rowLabels = {
    'Input'
    'Ground Truth'
    'RGB U-Net Score'
    'Fixed-Prior U-Net Score'
    'Learnable Topology Score'
};

numRows = numel(rowLabels);
numCols = numel(defectTypes) + 1; % first column is labels

fig = figure('Name', 'Clean Defect-Type Heatmaps', ...
    'Position', [100, 100, 1800, 1000], ...
    'Color', 'w');

tiledlayout(numRows, numCols, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for r = 1:numRows
    nexttile((r-1)*numCols + 1);
    axis off;
    text(0.5, 0.5, rowLabels{r}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', ...
        'FontSize', 13);
end

for i = 1:numel(defectTypes)

    defectType = defectTypes{i};

    [imgFile, maskFile] = selectBestImageByFixedIoU( ...
        datasetPath, splits, defectType, inputSizeSeg, ...
        M_prior_256, unet_FixedPrior, metrics_Fixed.Threshold);

    imgOriginal = imread(imgFile);
    gtOriginal = imread(maskFile);

    imgRGB = im2double(imresize(imgOriginal, inputSizeSeg));
    gtMask = imresize(gtOriginal, inputSizeSeg, 'nearest') > 0;

    img4 = cat(3, imgRGB, M_prior_256);

    [~, scoreRGB] = predictDefectScore(imgRGB, unet_RGB);
    [~, scoreFixed] = predictDefectScore(img4, unet_FixedPrior);
    [~, scoreLearnable] = predictDefectScore(img4, unet_LearnableTopology);

    % Column index offset by 1 due to row-label column
    c = i + 1;

    % Row 1: input
    nexttile(c);
    imshow(imgRGB);
    title(defectTitles{i}, 'FontWeight', 'bold');

    % Row 2: ground truth
    nexttile(numCols + c);
    imshow(gtMask);

    % Row 3: RGB score
    nexttile(2*numCols + c);
    imagesc(scoreRGB);
    axis image off;
    colormap(gca, hot);
    colorbar;

    % Row 4: Fixed-Prior score
    nexttile(3*numCols + c);
    imagesc(scoreFixed);
    axis image off;
    colormap(gca, hot);
    colorbar;

    % Row 5: Learnable score
    nexttile(4*numCols + c);
    imagesc(scoreLearnable);
    axis image off;
    colormap(gca, hot);
    colorbar;
end

sgtitle('Defect Score Heatmaps by Defect Type', ...
    'FontSize', 18, ...
    'FontWeight', 'bold');

saveFigureCompat(fig, fullfile(outputDir, 'Fig_DefectType_Heatmaps_Clean.png'));
saveFigureCompat(fig, fullfile(outputDir, 'Fig_DefectType_Heatmaps_Clean.pdf'));

disp('Clean heatmap figure saved:');
disp(fullfile(outputDir, 'Fig_DefectType_Heatmaps_Clean.png'));

%% Local helper
function [selectedImg, selectedMask] = selectBestImageByFixedIoU( ...
    datasetPath, splits, defectType, inputSizeSeg, M_prior_256, net, threshold)

    testFiles = splits.stage2.testImageFiles;
    candidates = {};

    for k = 1:numel(testFiles)
        thisFile = testFiles{k};
        if contains(thisFile, [filesep defectType filesep])
            candidates{end+1,1} = thisFile; %#ok<AGROW>
        end
    end

    if isempty(candidates)
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
        predFixed = bwareaopen(scoreFixed >= threshold, 10);

        TP = sum(predFixed(:) & gtMask(:));
        FP = sum(predFixed(:) & ~gtMask(:));
        FN = sum(~predFixed(:) & gtMask(:));

        ious(k) = TP / max(TP + FP + FN, eps);
    end

    [~, idx] = max(ious);

    selectedImg = candidates{idx};
    selectedMask = maskFiles{idx};
end