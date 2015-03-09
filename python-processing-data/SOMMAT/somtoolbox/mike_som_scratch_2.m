
%% SOM Setup: Load the data files
load('../SOM_tags.mat');
load('../small-labeled-vectors.mat');
% Create the generic SOM
sD = som_data_struct(vectors, 'comp_names', tags);
sD.data = double(sD.data);
sD.labels = num2cell(labels');
sD = som_normalize(sD, 'var');
sM = som_make(sD, 'msize', [50 50]);

%% Get frequency counts for (auto)-labelling of all grid points
sM = som_autolabel(sM, sD, 'add');
grid_labels = sM.labels;
grid_size = size(grid_labels, 1);

%% Continue with Frequency
grid_label_counts = cell(size(grid_size, 1), 2); % 2 classes
for i = 1:size(grid_labels,1)    
    count_pos = 0;    
    count_neg = 0;    
    for j = 1:size(grid_labels,2)        
        next = grid_labels{i,j};        
        if ~isempty(next)            
            if next == 1                
                count_pos = count_pos + 1;            
            elseif next == 0                
                count_neg = count_neg + 1;            
            else
                display('ERROR!!!!');            
            end
        end
    end
    grid_label_counts{i, 1} = count_neg;    
    grid_label_counts{i, 2} = count_pos;
end

%% Create a matrix to store the fractions
grid_weighted_fractions = zeros(grid_size, 1);
for i = 1:grid_size
    good_votes = grid_label_counts{i, 2};
    bad_votes = grid_label_counts{i, 1};
    if good_votes == 0 && bad_votes == 0 
        grid_weighted_fractions(i) = 0;
    else
        grid_weighted_fractions(i) = (good_votes / (good_votes + bad_votes)) * good_votes; 
    end
end

% Reshape the weighted matrix to be the same as the map
msize = sM.topol.msize;
grid_weighted_fractions = reshape(grid_weighted_fractions, msize);

% Generate the convolution filter (using gaussian kernel)
m = 5; n = 5; % sizes of the filters
sigma = 1;
[h1, h2] = meshgrid(-(m-1)/2:(m-1)/2, -(n-1)/2:(n-1)/2);
hg = exp(- (h1.^2+h2.^2) / (2*sigma^2));
h = hg ./ sum(hg(:));

% Convolve the matrix while preserving size
convolved_weighted_fractions = conv2(grid_weighted_fractions, h, 'same');

% Optional visualization
figure(1); HeatMap(convolved_weighted_fractions)
figure(2); surf(convolved_weighted_fractions)
figure(3); imagesc(convolved_weighted_fractions); colormap jet; 
% Done. 

%% Given some piece of new data 
% Let's call this new data newData --> array of vectors
% Normalize the vector first 
newData = som_normalize(newData, sM);
% Calculate BMU and map
[bmus, qerrs] = som_bmus(sM, newData, 'best');
bmus_size = size(bmus,  1);
som_responses = zeros(bmus_size, 1);
row_responses = zeros(bmus_size, 1);
col_responses = zeros(bmus_size, 1);

for i = 1:bmus_size 
    tmp = zeros(msize(1)*msize(2), 1);
    tmp(bmus(i, 1)) = 1;
    tmp = reshape(tmp, [msize(1), msize(2)]);
    % Find the index with the one
    [row, col, nothing] = find(tmp == 1);
    som_responses(i) = convolved_weighted_fractions(bmus(i));
    row_responses(i) = row;
    col_responses(i) = col;
end
% Return the som_unit

%% How to get from the sorted result back to data?
[sorted_responses, sorted_indexes] = sort(som_responses);
sorted_responses = flipud(sorted_responses);
sorted_indexes   = flipud(sorted_indexes);
raw_vectors = [];
for i = 1:10
    raw_vectors = cat(1, raw_vectors, som_denormalize(newData(sorted_indexes(i), :), sM));
end
top_ten = raw_vectors;

% plotting with top ten
figure; hold on; % create new figure
axis([0 msize(2) 0 msize(1)])
imagesc(convolved_weighted_fractions); colormap jet; 
scatter(col_responses, row_responses, 'o', 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');
for i=1:10
    scatter(col_responses(sorted_indexes(i)), row_responses(sorted_indexes(i)), 'o', 'MarkerEdgeColor', 'white', 'MarkerFaceColor', 'white');
end
hold off;

% Now we can move back to python and search for these! --> check out their
% dates and stuff..!