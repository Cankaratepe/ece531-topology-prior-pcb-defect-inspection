function splits = createSplits_MVTec(datasetPath)
% Creates clean train/validation/test splits for MVTec transistor.
% Also adds healthy images with blank masks to Stage 2 segmentation.

rng(42);

trainGoodPath = fullfile(datasetPath, 'train', 'good');
testPath = fullfile(datasetPath, 'test');
gtPath = fullfile(datasetPath, 'ground_truth');

% -------------------------
% Healthy images
% -------------------------
imdsTrainGood = imageDatastore(trainGoodPath);
healthyFiles_TrainOfficial = imdsTrainGood.Files;

imdsTestAll = imageDatastore(testPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labels = imdsTestAll.Labels;
labelStrings = string(labels);

healthyTestFiles = imdsTestAll.Files(labelStrings == "good");
faultyFiles = imdsTestAll.Files(labelStrings ~= "good");
faultyLabels = labelStrings(labelStrings ~= "good");

% -------------------------
% Split faulty images by defect type
% -------------------------
faultyTrain = {};
faultyVal = {};
faultyTest = {};

defectTypes = unique(faultyLabels);
defectTypes = defectTypes(defectTypes ~= "good");

for i = 1:numel(defectTypes)

    thisType = defectTypes(i);
    idx = find(faultyLabels == thisType);

    if isempty(idx)
        continue;
    end

    idx = idx(randperm(numel(idx)));
    n = numel(idx);

    if n == 1
        idxTrain = idx(1);
        idxVal = [];
        idxTest = [];
    elseif n == 2
        idxTrain = idx(1);
        idxVal = [];
        idxTest = idx(2);
    elseif n == 3
        idxTrain = idx(1);
        idxVal = idx(2);
        idxTest = idx(3);
    else
        nTrain = max(1, floor(0.50 * n));
        nVal = max(1, floor(0.25 * n));

        if nTrain + nVal >= n
            nVal = max(1, n - nTrain - 1);
        end

        idxTrain = idx(1:nTrain);
        idxVal = idx(nTrain+1:nTrain+nVal);
        idxTest = idx(nTrain+nVal+1:end);
    end

    faultyTrain = [faultyTrain; faultyFiles(idxTrain)];
    faultyVal = [faultyVal; faultyFiles(idxVal)];
    faultyTest = [faultyTest; faultyFiles(idxTest)];

    fprintf('%s: total=%d, train=%d, val=%d, test=%d\n', ...
        thisType, n, numel(idxTrain), numel(idxVal), numel(idxTest));
end

% -------------------------
% Split healthy images
% -------------------------
allHealthy = [healthyFiles_TrainOfficial; healthyTestFiles];
allHealthy = allHealthy(randperm(numel(allHealthy)));

nH = numel(allHealthy);
nHTrain = floor(0.60*nH);
nHVal = floor(0.20*nH);

healthyTrain = allHealthy(1:nHTrain);
healthyVal = allHealthy(nHTrain+1:nHTrain+nHVal);
healthyTest = allHealthy(nHTrain+nHVal+1:end);

% -------------------------
% Stage 1 classification splits
% -------------------------
splits.stage1.trainFiles = [healthyTrain; faultyTrain];
splits.stage1.trainLabels = categorical([ ...
    repmat("Healthy", numel(healthyTrain), 1); ...
    repmat("Faulty", numel(faultyTrain), 1)]);

splits.stage1.valFiles = [healthyVal; faultyVal];
splits.stage1.valLabels = categorical([ ...
    repmat("Healthy", numel(healthyVal), 1); ...
    repmat("Faulty", numel(faultyVal), 1)]);

splits.stage1.testFiles = [healthyTest; faultyTest];
splits.stage1.testLabels = categorical([ ...
    repmat("Healthy", numel(healthyTest), 1); ...
    repmat("Faulty", numel(faultyTest), 1)]);

% -------------------------
% Stage 2 segmentation splits
% Add healthy images with blank masks.
% Use limited healthy samples so defects are not overwhelmed.
% -------------------------
nHealthySegTrain = min(numel(healthyTrain), 2*numel(faultyTrain));
nHealthySegVal   = min(numel(healthyVal),   2*numel(faultyVal));
nHealthySegTest  = min(numel(healthyTest),  2*numel(faultyTest));

healthySegTrain = healthyTrain(1:nHealthySegTrain);
healthySegVal   = healthyVal(1:nHealthySegVal);
healthySegTest  = healthyTest(1:nHealthySegTest);

faultyTrainMasks = imageToMaskFiles(faultyTrain, datasetPath);
faultyValMasks   = imageToMaskFiles(faultyVal, datasetPath);
faultyTestMasks  = imageToMaskFiles(faultyTest, datasetPath);

blankTrainMasks = createBlankMaskFiles(healthySegTrain, datasetPath, 'stage2_train');
blankValMasks   = createBlankMaskFiles(healthySegVal, datasetPath, 'stage2_val');
blankTestMasks  = createBlankMaskFiles(healthySegTest, datasetPath, 'stage2_test');

splits.stage2.trainImageFiles = [faultyTrain; healthySegTrain];
splits.stage2.valImageFiles   = [faultyVal; healthySegVal];
splits.stage2.testImageFiles  = [faultyTest; healthySegTest];

splits.stage2.trainMaskFiles = [faultyTrainMasks; blankTrainMasks];
splits.stage2.valMaskFiles   = [faultyValMasks; blankValMasks];
splits.stage2.testMaskFiles  = [faultyTestMasks; blankTestMasks];

splits.stage2.defectOnlyTestImageFiles = faultyTest;
splits.stage2.defectOnlyTestMaskFiles = faultyTestMasks;

splits.gtPath = gtPath;

disp(' ');
disp('Split summary:');
fprintf('Stage 1 train: %d images\n', numel(splits.stage1.trainFiles));
fprintf('Stage 1 val:   %d images\n', numel(splits.stage1.valFiles));
fprintf('Stage 1 test:  %d images\n', numel(splits.stage1.testFiles));

fprintf('Stage 2 train: %d images (%d faulty + %d healthy blank)\n', ...
    numel(splits.stage2.trainImageFiles), numel(faultyTrain), numel(healthySegTrain));
fprintf('Stage 2 val:   %d images (%d faulty + %d healthy blank)\n', ...
    numel(splits.stage2.valImageFiles), numel(faultyVal), numel(healthySegVal));
fprintf('Stage 2 test:  %d images (%d faulty + %d healthy blank)\n', ...
    numel(splits.stage2.testImageFiles), numel(faultyTest), numel(healthySegTest));

end

function maskFiles = imageToMaskFiles(imageFiles, datasetPath)

maskFiles = cell(numel(imageFiles), 1);

for k = 1:numel(imageFiles)
    f = imageFiles{k};

    f = strrep(f, fullfile(datasetPath, 'test'), fullfile(datasetPath, 'ground_truth'));

    [folder, name, ~] = fileparts(f);
    maskFiles{k} = fullfile(folder, [name '_mask.png']);

    if ~isfile(maskFiles{k})
        warning('Mask file not found: %s', maskFiles{k});
    end
end

end

function blankMaskFiles = createBlankMaskFiles(imageFiles, datasetPath, splitName)

blankDir = fullfile(datasetPath, 'generated_blank_masks', splitName);

if ~exist(blankDir, 'dir')
    mkdir(blankDir);
end

blankMaskFiles = cell(numel(imageFiles), 1);

for k = 1:numel(imageFiles)

    imgInfo = imfinfo(imageFiles{k});
    H = imgInfo.Height;
    W = imgInfo.Width;

    blankMask = zeros(H, W, 'uint8');

    maskFile = fullfile(blankDir, sprintf('blank_%04d.png', k));
    imwrite(blankMask, maskFile);

    blankMaskFiles{k} = maskFile;
end

end