function thresholdInfo = tuneSegmentationThreshold(net, dsVal, modelName)

thresholds = linspace(0.05, 0.95, 91);

precision = zeros(size(thresholds));
recall = zeros(size(thresholds));
f1 = zeros(size(thresholds));
iou = zeros(size(thresholds));

allScores = [];
allGT = [];

reset(dsVal);

while hasdata(dsVal)

    data = read(dsVal);
    img = data{1};
    gtMask = data{2};

    [~, defectScore] = predictDefectScore(img, net);

    gtBinary = gtMask == "Defect";

    if ~isequal(size(defectScore), size(gtBinary))
        defectScore = imresize(defectScore, size(gtBinary), 'nearest');
    end

    allScores = [allScores; defectScore(:)];
    allGT = [allGT; gtBinary(:)];
end

for i = 1:numel(thresholds)

    predBinary = allScores >= thresholds(i);
    gtBinary = logical(allGT);

    TP = sum(predBinary & gtBinary);
    FP = sum(predBinary & ~gtBinary);
    FN = sum(~predBinary & gtBinary);

    precision(i) = TP / max(TP + FP, eps);
    recall(i) = TP / max(TP + FN, eps);
    f1(i) = 2 * precision(i) * recall(i) / max(precision(i) + recall(i), eps);
    iou(i) = TP / max(TP + FP + FN, eps);
end

% Use F1 as the threshold-selection criterion
[bestF1, bestIdx] = max(f1);

thresholdInfo = struct();
thresholdInfo.ModelName = modelName;
thresholdInfo.BestThreshold = thresholds(bestIdx);
thresholdInfo.BestValidationF1 = bestF1;
thresholdInfo.ValidationPrecision = precision(bestIdx);
thresholdInfo.ValidationRecall = recall(bestIdx);
thresholdInfo.ValidationIoU = iou(bestIdx);
thresholdInfo.Thresholds = thresholds;
thresholdInfo.PrecisionCurve = precision;
thresholdInfo.RecallCurve = recall;
thresholdInfo.F1Curve = f1;
thresholdInfo.IoUCurve = iou;

fprintf('\n%s threshold tuning:\n', modelName);
fprintf('Best threshold: %.2f\n', thresholdInfo.BestThreshold);
fprintf('Validation Precision: %.4f\n', thresholdInfo.ValidationPrecision);
fprintf('Validation Recall:    %.4f\n', thresholdInfo.ValidationRecall);
fprintf('Validation F1:        %.4f\n', thresholdInfo.BestValidationF1);
fprintf('Validation IoU:       %.4f\n', thresholdInfo.ValidationIoU);

% Save figure
outputDir = fullfile(pwd, 'Paper_Figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fig = figure('Name', ['Threshold Tuning - ' char(modelName)], ...
    'Position', [100, 100, 850, 500]);

plot(thresholds, precision, 'LineWidth', 2);
hold on;
plot(thresholds, recall, 'LineWidth', 2);
plot(thresholds, f1, 'LineWidth', 2);
plot(thresholds, iou, 'LineWidth', 2);

xline(thresholdInfo.BestThreshold, '--', ...
    sprintf('Best = %.2f', thresholdInfo.BestThreshold), ...
    'LineWidth', 1.5);

grid on;
xlabel('Defect probability threshold');
ylabel('Metric value');
legend({'Precision', 'Recall', 'F1', 'IoU'}, 'Location', 'best');

title(['Segmentation Threshold Tuning: ' char(modelName)], ...
    'FontSize', 14, 'FontWeight', 'bold');

safeName = char(matlab.lang.makeValidName(char(modelName)));

saveFigureCompat(fig, fullfile(outputDir, ['Fig_Threshold_' safeName '.pdf']));
saveFigureCompat(fig, fullfile(outputDir, ['Fig_Threshold_' safeName '.png']));

end