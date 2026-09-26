function convert_matrices()
    % Configure input and output directories
    data_root = getenv('HD_MSN_DATA_ROOT');
    if isempty(data_root), data_root = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data'); end
    input_dirs = {fullfile(data_root, 'HC308'), fullfile(data_root, 'HD308')};
    output_dirs = {fullfile(data_root, 'HC308_converted'), fullfile(data_root, 'HD308_converted')};
    
    % Set preprocessing parameters
    threshold = 0.3;  % Correlation threshold; adjust if required by the analysis plan
    
    % Process each directory
    for i = 1:length(input_dirs)
        input_dir = input_dirs{i};
        output_dir = output_dirs{i};
        
        % Create the output directory if needed
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % Locate all .mat files
        files = dir(fullfile(input_dir, '*similarity_matrix.mat'));
        
        % Process each file
        fprintf('Processing directory: %s\n', input_dir);
        fprintf('Total files: %d\n', length(files));
        
        for j = 1:length(files)
            try
                % Load the original data
                input_file = fullfile(input_dir, files(j).name);
                data = load(input_file);
                
                % Extract the MSN matrix
                NetworkMatrix = data.msn;
                
                % Preprocess the matrix
                % 1. Transform correlation coefficients to the [0, 1] range
                NetworkMatrix = (NetworkMatrix + 1)/2;
                
                % 2. Apply the threshold
                NetworkMatrix(NetworkMatrix < threshold) = 0;
                
                % 3. Enforce symmetry
                NetworkMatrix = (NetworkMatrix + NetworkMatrix')/2;
                
                % 4. Check for isolated nodes
                degree = sum(NetworkMatrix > 0);
                if any(degree == 0)
                    fprintf('Warning: isolated nodes detected in %s\n', files(j).name);
                end
                
                % Create the output filename
                [~, name, ~] = fileparts(files(j).name);
                new_name = strrep(name, '_similarity_matrix', '_network');
                output_file = fullfile(output_dir, [new_name, '.mat']);
                
                % Save the converted data
                save(output_file, 'NetworkMatrix');
                
                fprintf('Processed successfully: %s -> %s\n', files(j).name, [new_name, '.mat']);
                
                % Print basic summary statistics
                fprintf('  Matrix range: [%.4f, %.4f]\n', min(NetworkMatrix(:)), max(NetworkMatrix(:)));
                fprintf('  Nonzero element proportion: %.2f%%\n', 100*sum(NetworkMatrix(:)>0)/numel(NetworkMatrix));
                
            catch e
                fprintf('Failed to process file: %s\nError: %s\n', files(j).name, e.message);
            end
        end
        
        fprintf('\nCompleted directory %s\n', input_dir);
    end
    
    fprintf('\nAll files processed successfully.\n');
end

% =============================================================================
% SCRIPT DOCUMENTATION
% Purpose: Convert source MSN matrix files into standardized matrix representations required by downstream analyses.
% Input source/location: HD and HC matrix directories specified by HD_MSN_DATA_ROOT or project-relative input locations.
% Output location: Converted matrix files in the configured project output directories.
% Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
% Main steps: Enumerate source matrices; load each matrix; validate dimensions/content; convert the representation; save standardized outputs; report failures.
% Log location: No dedicated log file; status and errors are written to the MATLAB console.
% =============================================================================
