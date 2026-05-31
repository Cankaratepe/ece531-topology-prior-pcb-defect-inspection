function segDS = makeSegmentationDatastores(datasetPath, splits, inputSizeSeg, M_prior_256)

classes = ["NormalBackground", "Defect"];
labelIDs = [0, 255];

% -----------------------------
% RGB datastores
% -----------------------------
imdsTrain = imageDatastore(splits.stage2.trainImageFiles);
imdsVal   = imageDatastore(splits.stage2.valImageFiles);
imdsTest  = imageDatastore(splits.stage2.testImageFiles);

pxdsTrain = pixelLabelDatastore(splits.stage2.trainMaskFiles, classes, labelIDs);
pxdsVal   = pixelLabelDatastore(splits.stage2.valMaskFiles, classes, labelIDs);
pxdsTest  = pixelLabelDatastore(splits.stage2.testMaskFiles, classes, labelIDs);

dsTrainRGB = transform(combine(imdsTrain, pxdsTrain), ...
    @(data) preprocessRGBSeg(data, inputSizeSeg));

dsValRGB = transform(combine(imdsVal, pxdsVal), ...
    @(data) preprocessRGBSeg(data, inputSizeSeg));

dsTestRGB = transform(combine(imdsTest, pxdsTest), ...
    @(data) preprocessRGBSeg(data, inputSizeSeg));

% -----------------------------
% Physics/topology channel datastores
% -----------------------------
dsTrainPhysics = transform(combine(imdsTrain, pxdsTrain), ...
    @(data) addPhysicsChannel(data, M_prior_256, inputSizeSeg));

dsValPhysics = transform(combine(imdsVal, pxdsVal), ...
    @(data) addPhysicsChannel(data, M_prior_256, inputSizeSeg));

dsTestPhysics = transform(combine(imdsTest, pxdsTest), ...
    @(data) addPhysicsChannel(data, M_prior_256, inputSizeSeg));

% -----------------------------
% Store
% -----------------------------
segDS.classes = classes;
segDS.labelIDs = labelIDs;

segDS.rgb.train = dsTrainRGB;
segDS.rgb.val = dsValRGB;
segDS.rgb.test = dsTestRGB;

segDS.physics.train = dsTrainPhysics;
segDS.physics.val = dsValPhysics;
segDS.physics.test = dsTestPhysics;

segDS.raw.testImages = imdsTest;
segDS.raw.testMasks = pxdsTest;

segDS.inputSizeSeg = inputSizeSeg;
segDS.M_prior_256 = M_prior_256;

end

function dataOut = preprocessRGBSeg(dataIn, targetSize)

img = imresize(dataIn{1}, targetSize);
mask = imresize(dataIn{2}, targetSize, 'nearest');

img = im2double(img);

dataOut = {img, mask};

end