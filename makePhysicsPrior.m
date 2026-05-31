function M = makePhysicsPrior(targetSize)
% Creates a simple transistor topology prior.
% High values correspond to expected current-carrying metallic leads.
%
% This is not the final decision mask. It is used as a topology cue.

H = targetSize(1);
W = targetSize(2);

M = ones(H, W) * 0.1;

% Approximate normalized lead boxes.
% These are based on the typical MVTec transistor alignment.
leadBoxes = [
    0.31 0.49 0.39 0.88
    0.48 0.49 0.56 0.88
    0.65 0.49 0.73 0.88
];

for i = 1:size(leadBoxes,1)
    x1 = round(leadBoxes(i,1) * W);
    y1 = round(leadBoxes(i,2) * H);
    x2 = round(leadBoxes(i,3) * W);
    y2 = round(leadBoxes(i,4) * H);

    x1 = max(1,x1); y1 = max(1,y1);
    x2 = min(W,x2); y2 = min(H,y2);

    M(y1:y2, x1:x2) = 1.0;
end

M = imgaussfilt(M, 0.04 * min(H,W));
M = mat2gray(M);

end