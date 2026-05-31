%% GENERATE THRESHOLDED PAPER-STYLE QUALITATIVE FIGURES
clear;

if ~isfile('Trained_Models_Corrected.mat')
    error('Trained_Models_Corrected.mat not found. Run main_train_all.m first.');
end

if ~isfile('Evaluation_Results.mat')
    error('Evaluation_Results.mat not found. Run main_evaluate_all.m first.');
end

load('Trained_Models_Corrected.mat');
load('Evaluation_Results.mat');

segDS = makeSegmentationDatastores(datasetPath, splits, inputSizeSeg, M_prior_256);

outputDir = fullfile(pwd, 'Paper_Figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

disp('Generating thresholded paper-style figures...');

%% ------------------------------------------------------------
% FIGURE 1: Topology / physics prior
% ------------------------------------------------------------
fig1 = figure('Name', 'Topology Prior', ...
    'Position', [100, 100, 600, 500]);

imagesc(M_prior_256);
axis image off;
colormap hot;
colorbar;

title('Topology Prior for Electrically Critical Lead Regions', ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

saveFigureCompat(fig1, fullfile(outputDir, 'Fig1_Topology_Prior.pdf'));
saveFigureCompat(fig1, fullfile(outputDir, 'Fig1_Topology_Prior.png'));

%% ------------------------------------------------------------
% FIGURE 2: Thresholded qualitative comparison
%
% Rows:
%   1. Input
%   2. Ground truth
%   3. RGB U-Net
%   4. Fixed-Prior U-Net
%   5. Fixed-Prior + Topology Gate
%   6. Learnable Topology U-Net
% ------------------------------------------------------------
dsTest = segDS.physics.test;
reset(dsTest);

numExamples = 4;
examples = cell(numExamples, 1);

for i = 1:numExamples
    if hasdata(dsTest)
        examples{i} = read(dsTest);
    else
        warning('Only %d examples available in test datastore.', i-1);
        numExamples = i-1;
        examples = examples(1:numExamples);
        break;
    end
end

fig2 = figure('Name', 'Thresholded Qualitative Comparison', ...
    'Position', [100, 100, 1600, 1050]);

tiledlayout(6, numExamples, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

rowNames = {
    'Input'
    'Ground Truth'
    'RGB U-Net'
    'Fixed-Prior U-Net'
    'Fixed-Prior + Gate'
    'Learnable Topology U-Net'
};

for i = 1:numExamples

    data = examples{i};

    img4 = data{1};
    gtMask = data{2};

    imgRGB = img4(:,:,1:3);

    % Get defect scores
    [~, scoreRGB] = predictDefectScore(imgRGB, unet_RGB);
    [~, scoreFixed] = predictDefectScore(img4, unet_FixedPrior);
    [~, scoreLearnable] = predictDefectScore(img4, unet_LearnableTopology);

    % Thresholded binary predictions
    predRGB = scoreRGB >= metrics_RGB.Threshold;
    predFixed = scoreFixed >= metrics_Fixed.Threshold;
    predLearnable = scoreLearnable >= metrics_Learnable.Threshold;

    % Light cleanup
    predRGB = bwareaopen(predRGB, 10);
    predFixed = bwareaopen(predFixed, 10);
    predLearnable = bwareaopen(predLearnable, 10);

    % Topology-gated fixed-prior prediction
    predFixedGated = applyTopologyGate( ...
        scoreFixed, ...
        metrics_Fixed.Threshold, ...
        M_prior_256);

    % Row 1: input
    nexttile(i);
    imshow(imgRGB);
    title(sprintf('Example %d', i), 'FontWeight', 'bold');
    if i == 1
        ylabel(rowNames{1}, 'FontWeight', 'bold');
    end

    % Row 2: ground truth
    nexttile(i + numExamples);
    imshow(gtMask == "Defect");
    if i == 1
        ylabel(rowNames{2}, 'FontWeight', 'bold');
    end

    % Row 3: RGB U-Net
    nexttile(i + 2*numExamples);
    imshow(makeOverlay(imgRGB, predRGB));
    if i == 1
        ylabel(rowNames{3}, 'FontWeight', 'bold');
    end

    % Row 4: Fixed-Prior U-Net
    nexttile(i + 3*numExamples);
    imshow(makeOverlay(imgRGB, predFixed));
    if i == 1
        ylabel(rowNames{4}, 'FontWeight', 'bold');
    end

    % Row 5: Fixed-Prior + Topology Gate
    nexttile(i + 4*numExamples);
    imshow(makeOverlay(imgRGB, predFixedGated));
    if i == 1
        ylabel(rowNames{5}, 'FontWeight', 'bold');
    end

    % Row 6: Learnable Topology U-Net
    nexttile(i + 5*numExamples);
    imshow(makeOverlay(imgRGB, predLearnable));
    if i == 1
        ylabel(rowNames{6}, 'FontWeight', 'bold');
    end
end

sgtitle('Thresholded Qualitative Defect Localization Comparison', ...
    'FontSize', 16, ...
    'FontWeight', 'bold');

saveFigureCompat(fig2, fullfile(outputDir, 'Fig2_Thresholded_Qualitative_Comparison.pdf'));
saveFigureCompat(fig2, fullfile(outputDir, 'Fig2_Thresholded_Qualitative_Comparison.png'));

%% ------------------------------------------------------------
% FIGURE 3: Defect score heatmaps
% ------------------------------------------------------------
reset(dsTest);

if ~hasdata(dsTest)
    error('No test data available for heatmap figure.');
end

data = read(dsTest);

img4 = data{1};
gtMask = data{2};
imgRGB = img4(:,:,1:3);

[~, scoreRGB] = predictDefectScore(imgRGB, unet_RGB);
[~, scoreFixed] = predictDefectScore(img4, unet_FixedPrior);
[~, scoreLearnable] = predictDefectScore(img4, unet_LearnableTopology);

predFixedGated = applyTopologyGate( ...
    scoreFixed, ...
    metrics_Fixed.Threshold, ...
    M_prior_256);

fig3 = figure('Name', 'Defect Score Heatmaps', ...
    'Position', [100, 100, 1650, 520]);

tiledlayout(1, 6, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

nexttile;
imshow(imgRGB);
title('Input');

nexttile;
imshow(gtMask == "Defect");
title('Ground Truth');

nexttile;
imagesc(scoreRGB);
axis image off;
colormap(gca, hot);
colorbar;
title(sprintf('RGB Score, T=%.2f', metrics_RGB.Threshold));

nexttile;
imagesc(scoreFixed);
axis image off;
colormap(gca, hot);
colorbar;
title(sprintf('Fixed Score, T=%.2f', metrics_Fixed.Threshold));

nexttile;
imshow(predFixedGated);
title('Fixed + Gate');

nexttile;
imagesc(scoreLearnable);
axis image off;
colormap(gca, hot);
colorbar;
title(sprintf('Learnable Score, T=%.2f', metrics_Learnable.Threshold));

sgtitle('Pixel-Level Defect Score Heatmaps and Topology-Gated Output', ...
    'FontSize', 16, ...
    'FontWeight', 'bold');

saveFigureCompat(fig3, fullfile(outputDir, 'Fig3_Thresholded_Probability_Heatmaps.pdf'));
saveFigureCompat(fig3, fullfile(outputDir, 'Fig3_Thresholded_Probability_Heatmaps.png'));

%% ------------------------------------------------------------
% FIGURE 4: Ablation bar chart
% ------------------------------------------------------------
if isfile('Ablation_Table.csv')

    T = readtable('Ablation_Table.csv');

    fig4 = figure('Name', 'Ablation Bar Chart', ...
        'Position', [100, 100, 1000, 550]);

    metricsToPlot = [T.DefectIoU, T.MeanIoU, T.PixelAUROC];

    bar(metricsToPlot);
    grid on;

    xticklabels(T.modelNames);
    xtickangle(20);

    ylabel('Score');
    legend({'Defect IoU', 'Mean IoU', 'Pixel AUROC'}, ...
        'Location', 'northoutside', ...
        'Orientation', 'horizontal');

    title('Ablation Study of Topology Prior Integration', ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    saveFigureCompat(fig4, fullfile(outputDir, 'Fig4_Ablation_BarChart.pdf'));
    saveFigureCompat(fig4, fullfile(outputDir, 'Fig4_Ablation_BarChart.png'));
else
    warning('Ablation_Table.csv not found. Skipping ablation bar chart.');
end

disp('Thresholded paper figures generated in Paper_Figures folder.');

%% ------------------------------------------------------------
% Local helper function: overlay binary prediction on RGB image
% ------------------------------------------------------------
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