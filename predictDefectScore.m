function [predMaskDefault, defectScore] = predictDefectScore(img, net)
% Returns:
%   predMaskDefault : MATLAB default semantic segmentation labels
%   defectScore     : estimated probability/score for Defect class

try
    [predMaskDefault, scoreMap, allScores] = semanticseg(img, net);
catch
    [predMaskDefault, scoreMap] = semanticseg(img, net);
    allScores = [];
end

% Try to get the Defect class index
defectIdx = 2;

try
    classNames = string(net.Layers(end).Classes);
    tempIdx = find(classNames == "Defect", 1);
    if ~isempty(tempIdx)
        defectIdx = tempIdx;
    end
catch
    defectIdx = 2;
end

% Best case: all class probability maps are available
if ~isempty(allScores) && ndims(allScores) == 3 && size(allScores,3) >= defectIdx
    defectScore = double(allScores(:,:,defectIdx));
    return;
end

% Fallback case: scoreMap is only the confidence of predicted class
if ismatrix(scoreMap)
    winningScore = double(scoreMap);
elseif ndims(scoreMap) == 3 && size(scoreMap,3) == 1
    winningScore = double(scoreMap(:,:,1));
elseif ndims(scoreMap) == 3 && size(scoreMap,3) >= defectIdx
    defectScore = double(scoreMap(:,:,defectIdx));
    return;
else
    warning('Unexpected score map format. Using binary predicted mask as score.');
    defectScore = double(predMaskDefault == "Defect");
    return;
end

% Convert winning-class confidence into approximate defect confidence
predIsDefect = predMaskDefault == "Defect";

defectScore = zeros(size(winningScore));
defectScore(predIsDefect) = winningScore(predIsDefect);
defectScore(~predIsDefect) = 1 - winningScore(~predIsDefect);

defectScore = min(max(defectScore, 0), 1);

end