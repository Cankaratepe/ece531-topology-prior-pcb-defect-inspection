classdef PhysicsInputAttentionLayer < nnet.layer.Layer
    % Learnable topology-conditioned spatial attention layer.
    %
    % Input:
    %   H x W x 4 x N
    %
    % Channels:
    %   1:3 = RGB image
    %   4   = topology / physics prior
    %
    % Output:
    %   H x W x 3 x N
    %
    % Operation:
    %   Y = RGB .* (1 + sigmoid(alpha) * topology)

    properties (Learnable)
        Alpha
    end

    methods
        function layer = PhysicsInputAttentionLayer(name)
            layer.Name = name;
            layer.Description = "Learnable topology-conditioned input attention";

            % Mild initial topology influence
            layer.Alpha = single(0.25);
        end

        function Z = predict(layer, X)
            RGB = X(:,:,1:3,:);
            M = X(:,:,4,:);

            alpha = 1 ./ (1 + exp(-layer.Alpha));

            attention = 1 + alpha .* M;
            attention = repmat(attention, 1, 1, 3, 1);

            Z = RGB .* attention;
        end

        function Z = forward(layer, X)
            Z = predict(layer, X);
        end
    end
end