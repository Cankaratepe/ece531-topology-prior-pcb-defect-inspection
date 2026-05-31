function thresholdInfo = tuneStage1Threshold(net, splits)
% Tunes the Stage 1 faulty-class decision threshold on validation data,
% then evaluates the selected threshold on the held-out test data.

inputSize = [224 224];

% -----------------------------
% Validation set: choose threshold
% -----------------------------
imdsVal = imageDatastore(splits.stage1.valFiles);
imdsVal.Labels = splits.stage1.valLabels;

augVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

[~, valScores] = classify(net, augVal);

classNames = string(net.Layers(end).Classes);
faultyIdx = find(classNames == "Faulty");

if isempty(faultyIdx)
    faultyIdx = 2;
end

valFaultyScores = valScores(:, faultyIdx);
YVal = imdsVal.Labels == "Faulty";

thresholds = linspace(0, 1, 101);

valPrecision = zeros(size(thresholds));
valRecall = zeros(size(thresholds));
valF1 = zeros(size(thresholds));

for i = 1:numel(thresholds)

    YPred = valFaultyScores >= thresholds(i);

    TP = sum(YPred & YVal);
    FP = sum(YPred & ~YVal);
    FN = sum(~YPred & YVal);

    valPrecision(i) = TP / max(TP + FP, eps);
    valRecall(i) = TP / max(TP + FN, eps);
    valF1(i) = 2 * valPrecision(i) * valRecall(i) / max(valPrecision(i) + valRecall(i), eps);
end

[bestValF1, bestIdx] = max(valF1);
bestThreshold = thresholds(bestIdx);

% -----------------------------
% Test set: evaluate selected threshold
% -----------------------------
imdsTest = imageDatastore(splits.stage1.testFiles);
imdsTest.Labels = splits.stage1.testLabels;

augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

[~, testScores] = classify(net, augTest);

testFaultyScores = testScores(:, faultyIdx);
YTest = imdsTest.Labels == "Faulty";

YPredTest = testFaultyScores >= bestThreshold;

TP = sum(YPredTest & YTest);
FP = sum(YPredTest & ~YTest);
FN = sum(~YPredTest & YTest);
TN = sum(~YPredTest & ~YTest);

testAccuracy = (TP + TN) / max(TP + FP + FN + TN, eps);
testPrecision = TP / max(TP + FP, eps);
testRecall = TP / max(TP + FN, eps);
testF1 = 2 * testPrecision * testRecall / max(testPrecision + testRecall, eps);

[rocX, rocY, ~, testAUC] = perfcurve(YTest, testFaultyScores, true);

thresholdInfo = struct();
thresholdInfo.BestThreshold = bestThreshold;
thresholdInfo.ValidationBestF1 = bestValF1;
thresholdInfo.TestAccuracy = testAccuracy;
thresholdInfo.TestPrecision = testPrecision;
thresholdInfo.TestRecall = testRecall;
thresholdInfo.TestF1 = testF1;
thresholdInfo.TestAUC = testAUC;
thresholdInfo.TP = TP;
thresholdInfo.FP = FP;
thresholdInfo.FN = FN;
thresholdInfo.TN = TN;
thresholdInfo.Thresholds = thresholds;
thresholdInfo.ValidationPrecisionCurve = valPrecision;
thresholdInfo.ValidationRecallCurve = valRecall;
thresholdInfo.ValidationF1Curve = valF1;

% -----------------------------
% Figures
% -----------------------------
outputDir = fullfile(pwd, 'Paper_Figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fig1 = figure('Name', 'Stage 1 Validation Threshold Tuning', ...
    'Position', [100, 100, 850, 500]);

plot(thresholds, valPrecision, 'LineWidth', 2);
hold on;
plot(thresholds, valRecall, 'LineWidth', 2);
plot(thresholds, valF1, 'LineWidth', 2);

xline(bestThreshold, '--', ...
    sprintf('Best threshold = %.2f', bestThreshold), ...
    'LineWidth', 1.5);

grid on;
xlabel('Faulty-class decision threshold');
ylabel('Metric value');
legend({'Precision', 'Recall', 'F1-score'}, ...
    'Location', 'best');

title('Stage 1 Threshold Selection on Validation Set', ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig1, fullfile(outputDir, 'Fig_Stage1_Validation_Threshold_Tuning.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig1, fullfile(outputDir, 'Fig_Stage1_Validation_Threshold_Tuning.png'), ...
    'Resolution', 300);

fig2 = figure('Name', 'Stage 1 Test ROC', ...
    'Position', [100, 100, 650, 500]);

plot(rocX, rocY, 'LineWidth', 2);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Stage 1 Test ROC, AUROC = %.4f', testAUC), ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig2, fullfile(outputDir, 'Fig_Stage1_Test_ROC.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig2, fullfile(outputDir, 'Fig_Stage1_Test_ROC.png'), ...
    'Resolution', 300);

end