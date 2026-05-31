function dataOut = addPhysicsChannel(dataIn, M_prior, targetSize)

img = imresize(dataIn{1}, targetSize);
mask = imresize(dataIn{2}, targetSize, 'nearest');

img = im2double(img);

if size(M_prior,1) ~= targetSize(1) || size(M_prior,2) ~= targetSize(2)
    M_prior = imresize(M_prior, targetSize);
end

img_4Channel = cat(3, img, M_prior);

dataOut = {img_4Channel, mask};

end