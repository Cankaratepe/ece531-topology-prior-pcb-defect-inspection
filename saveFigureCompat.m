function saveFigureCompat(figHandle, filename)
% Saves figures in a MATLAB-version-compatible way.
% Converts string inputs to char for older MATLAB versions.

% Convert string to character vector if needed
if isstring(filename)
    filename = char(filename);
end

[folder, ~, ext] = fileparts(filename);

if isstring(folder)
    folder = char(folder);
end

if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end

try
    % Newer MATLAB versions
    exportgraphics(figHandle, filename);
catch
    % Older MATLAB fallback
    set(figHandle, 'PaperPositionMode', 'auto');

    switch lower(ext)
        case '.pdf'
            print(figHandle, filename, '-dpdf', '-bestfit');

        case '.png'
            print(figHandle, filename, '-dpng', '-r300');

        case '.jpg'
            print(figHandle, filename, '-djpeg', '-r300');

        case '.jpeg'
            print(figHandle, filename, '-djpeg', '-r300');

        otherwise
            warning('Unknown file extension. Saving as PNG instead.');
            print(figHandle, [filename '.png'], '-dpng', '-r300');
    end
end

end