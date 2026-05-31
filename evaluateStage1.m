function metrics = evaluateStage1(net, splits)

inputSize = [224 224];

imdsTest = imageDatastore(splits.stage1.testFiles);
imdsTest.Labels = splits.stage1.testLabels;

augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

[YPred, scores] = classify(net, augTest);
YTrue = imdsTest.Labels;

classes = categories(YTrue);

positiveClass = "Faulty";

faultyIdx = find(string(net.Layers(end).Classes) == positiveClass);

if isempty(faultyIdx)
    faultyIdx = 2;
end

faultyScores = scores(:, faultyIdx);

[Xroc, Yroc, ~, AUC] = perfcurve(YTrue, faultyScores, positiveClass);

TP = sum((YPred == positiveClass) & (YTrue == positiveClass));
FP = sum((YPred == positiveClass) & (YTrue ~= positiveClass));
FN = sum((YPred ~= positiveClass) & (YTrue == positiveClass));
TN = sum((YPred ~= positiveClass) & (YTrue ~= positiveClass));

accuracy = mean(YPred == YTrue);
precision = TP / max(TP + FP, eps);
recall = TP / max(TP + FN, eps);
f1 = 2 * precision * recall / max(precision + recall, eps);

metrics = table(accuracy, precision, recall, f1, AUC, TP, FP, FN, TN);

figure('Name', 'Stage 1 ROC');
plot(Xroc, Yroc, 'LineWidth', 2);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Stage 1 ROC Curve, AUROC = %.4f', AUC));

exportgraphics(gcf, 'Fig_Stage1_ROC.pdf');

figure('Name', 'Stage 1 Confusion Matrix');
confusionchart(YTrue, YPred);
title('Stage 1 ResNet-18 Confusion Matrix');

exportgraphics(gcf, 'Fig_Stage1_ConfusionMatrix.pdf');

end