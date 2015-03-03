%% SOM Setup: Load the data files
load('../SOM_tags.mat');
load('../small-labeled-vectors.mat');
% Create the generic SOM
sD = som_data_struct(vectors, 'comp_names', tags);
sD.data = double(sD.data);
sD.labels = num2cell(labels');
sD = som_normalize(sD, 'var');
sM = som_make(sD);

%% Get frequency counts for (auto)-labelling of all grid points
sM = som_autolabel(sM, sD, 'add');
grid_labels = sM.labels;
grid_size = size(grid_labels, 1);

%% Convolution
convsize = 5;
convmat = 1/(convsize^2)*ones(convsize, convsize);
full_grid_labels = grid_labels;
mat_grid_labels = zeros(size(full_grid_labels,1), size(full_grid_labels,2));
% Replace with 0 
for i = 1:size(full_grid_labels, 1)
    for j = 1:size(full_grid_labels, 2)
        if isempty(full_grid_labels{i,j})
            mat_grid_labels(i,j) = 0;
        else 
            mat_grid_labels(i,j) = full_grid_labels{i,j};
        end
    end
end
H = conv2(mat_grid_labels, convmat, 'same'); % where is the matrix

%% Continue with Frequency
grid_labels = num2cell(H);
grid_label_counts = cell(size(grid_size, 1), 1); % continuous spectrum of classes
for i = 1:size(grid_labels, 1)
    count = 0;       
    for j = 1:size(grid_labels,2)        
        next = grid_labels{i,j};        
        if ~isempty(next)            
            count = count + next;
        end
    end
    grid_label_counts{i, 1} = count;   
end

