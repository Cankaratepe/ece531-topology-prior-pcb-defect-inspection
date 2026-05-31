function net = trainUNet_LearnableTopology(segDS)

numClasses = 2;

% Base RGB U-Net
baseGraph = unetLayers([256 256 3], numClasses, 'EncoderDepth', 4);

% Replace original image input with a 4-channel input + learnable attention
input4 = imageInputLayer([256 256 4], ...
    'Name', 'Input_RGB_Topology', ...
    'Normalization', 'none');

attentionLayer = PhysicsInputAttentionLayer('Learnable_Topology_Attention');

lgraph = layerGraph();

lgraph = addLayers(lgraph, input4);
lgraph = addLayers(lgraph, attentionLayer);

% Remove original image input layer from base graph
baseLayers = baseGraph.Layers;
baseConnections = baseGraph.Connections;

originalInputName = baseLayers(1).Name;

baseGraph = removeLayers(baseGraph, originalInputName);

lgraph = addLayers(lgraph, baseGraph.Layers);
lgraph = connectLayers(lgraph, 'Input_RGB_Topology', 'Learnable_Topology_Attention');
lgraph = connectLayers(lgraph, 'Learnable_Topology_Attention', baseLayers(2).Name);

% Restore internal U-Net connections
for i = 1:size(baseGraph.Connections,1)
    src = baseGraph.Connections.Source{i};
    dst = baseGraph.Connections.Destination{i};

    try
        lgraph = connectLayers(lgraph, src, dst);
    catch
        % Some connections may already exist depending on MATLAB version
    end
end

classWeights = [1.0, 2.0];

pxLayer = pixelClassificationLayer( ...
    'Name', 'Segmentation-Layer', ...
    'Classes', segDS.classes, ...
    'ClassWeights', classWeights);

lgraph = replaceLayer(lgraph, 'Segmentation-Layer', pxLayer);

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', segDS.physics.val, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(segDS.physics.train, lgraph, options);

end