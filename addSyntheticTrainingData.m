function splits = addSyntheticTrainingData(splits, datasetPath)
% Adds synthetic defect images to TRAINING ONLY.
%
% If masks are not available, synthetic images are used only for
% Stage 1 binary classification.
%
% Final validation/test sets remain real MVTec images.

syntheticClassificationRoot = fullfile(datasetPath, 'synthetic', 'classification_train');

if ~exist(syntheticClassificationRoot, 'dir')
    disp('No synthetic classification folder found. Continuing without synthetic data.');
    return;
end

defectTypes = {'bent_lead', 'cut_lead', 'damaged_case', 'misplaced'};

syntheticImageFiles = {};

for i = 1:numel(defectTypes)

    defectType = defectTypes{i};
    imgFolder = fullfile(syntheticClassificationRoot, defectType);

    if ~exist(imgFolder, 'dir')
        fprintf('Synthetic folder not found for %s. Skipping.\n', defectType);
        continue;
    end

    imdsSynth = imageDatastore(imgFolder);

    for k = 1:numel(imdsSynth.Files)
        syntheticImageFiles{end+1,1} = imdsSynth.Files{k}; %#ok<AGROW>
    end
end

if isempty(syntheticImageFiles)
    disp('No synthetic images found.');
    return;
end

% Add synthetic images to Stage 1 classifier training only
splits.stage1.trainFiles = [splits.stage1.trainFiles; syntheticImageFiles];

splits.stage1.trainLabels = [splits.stage1.trainLabels; ...
    categorical(repmat("Faulty", numel(syntheticImageFiles), 1))];

fprintf('Added %d synthetic images to Stage 1 training only.\n', ...
    numel(syntheticImageFiles));

disp('Synthetic images were NOT added to Stage 2 segmentation because no pixel masks were provided.');

end