function metricsOut = evaluateSegmentationModel(net, dsTest, modelName, threshold, makeFigures)

if nargin < 5
    makeFigures = false;
end

predDir = fullfile(pwd, ['Predictions_' char(modelName)]);

if exist(predDir, 'dir')
    rmdir(predDir, 's');
end
mkdir(predDir);

reset(dsTest);

allScores = [];
allGT = [];

TP = 0;
FP = 0;
FN = 0;
TN = 0;

i = 1;

while hasdata(dsTest)

    data = read(dsTest);

    img = data{1};
    gtMask = data{2};

    [~, defectScore] = predictDefectScore(img, net);

    gtBinary = gtMask == "Defect";

    if ~isequal(size(defectScore), size(gtBinary))
        defectScore = imresize(defectScore, size(gtBinary), 'nearest');
    end

    predBinary = defectScore >= threshold;

    % Optional light cleanup: remove tiny isolated speckles
    predBinary = bwareaopen(predBinary, 10);

    outMask = uint8(predBinary) * 255;
    predFilename = fullfile(predDir, sprintf('prediction_%04d.png', i));
    imwrite(outMask, predFilename);

    TP = TP + sum(predBinary(:) & gtBinary(:));
    FP = FP + sum(predBinary(:) & ~gtBinary(:));
    FN = FN + sum(~predBinary(:) & gtBinary(:));
    TN = TN + sum(~predBinary(:) & ~gtBinary(:));

    allScores = [allScores; double(defectScore(:))];
    allGT = [allGT; logical(gtBinary(:))];

    if makeFigures && i <= 4

        figure;
        tiledlayout(1,4, 'TileSpacing', 'compact', 'Padding', 'compact');

        if size(img,3) == 4
            baseImg = img(:,:,1:3);
        else
            baseImg = img;
        end

        nexttile;
        imshow(baseImg);
        title('Input');

        nexttile;
        imshow(gtBinary);
        title('Ground Truth');

        nexttile;
        imagesc(defectScore);
        axis image off;
        colormap(gca, hot);
        colorbar;
        title('Defect Score');

        nexttile;
        overlay = labeloverlay(baseImg, categorical(predBinary, [0 1], ["Background", "Defect"]), ...
            'IncludedLabels', "Defect", ...
            'Colormap', [1 1 0], ...
            'Transparency', 0.55);
        imshow(overlay);
        title(sprintf('Prediction, T=%.2f', threshold));

        exportgraphics(gcf, sprintf('Fig_%s_Thresholded_Example_%02d.pdf', modelName, i));
    end

    i = i + 1;
end

Precision = TP / max(TP + FP, eps);
Recall = TP / max(TP + FN, eps);
F1 = 2 * Precision * Recall / max(Precision + Recall, eps);

DefectIoU = TP / max(TP + FP + FN, eps);
BackgroundIoU = TN / max(TN + FP + FN, eps);
MeanIoU = mean([BackgroundIoU, DefectIoU]);

try
    [~,~,~,PixelAUROC] = perfcurve(logical(allGT), allScores, true);
catch
    PixelAUROC = NaN;
end

metricsOut = struct();
metricsOut.ModelName = modelName;
metricsOut.Threshold = threshold;
metricsOut.Precision = Precision;
metricsOut.Recall = Recall;
metricsOut.F1 = F1;
metricsOut.DefectIoU = DefectIoU;
metricsOut.BackgroundIoU = BackgroundIoU;
metricsOut.MeanIoU = MeanIoU;
metricsOut.PixelAUROC = PixelAUROC;
metricsOut.TP = TP;
metricsOut.FP = FP;
metricsOut.FN = FN;
metricsOut.TN = TN;

fprintf('\n%s Metrics, threshold = %.2f:\n', modelName, threshold);
fprintf('Precision:   %.4f\n', Precision);
fprintf('Recall:      %.4f\n', Recall);
fprintf('F1:          %.4f\n', F1);
fprintf('Defect IoU:  %.4f\n', DefectIoU);
fprintf('Mean IoU:    %.4f\n', MeanIoU);
fprintf('Pixel AUROC: %.4f\n', PixelAUROC);

end