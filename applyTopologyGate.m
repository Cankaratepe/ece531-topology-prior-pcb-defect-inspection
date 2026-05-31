function predBinary = applyTopologyGate(defectScore, threshold, M_prior)
% Physics/topology-guided post-processing.
% Suppresses predictions far away from electrically critical lead regions.

if ~isequal(size(defectScore), size(M_prior))
    M_prior = imresize(M_prior, size(defectScore));
end

candidate = defectScore >= threshold;

% Lead-region ROI from topology prior
roi = M_prior > 0.35;

% Expand ROI slightly so nearby lead defects are preserved
roi = imdilate(roi, strel('disk', 12));

% Apply topology gate
predBinary = candidate & roi;

% Remove small speckles
predBinary = bwareaopen(predBinary, 10);

% Light smoothing
predBinary = imclose(predBinary, strel('disk', 3));

end