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
% grid_label_counts{i,0} = number of positively labelled data points for which this is the BMU
% grid_label_counts{i,1} = number of negatively labelled data points for which this is the BMU
grid_labels = num2cell(H);
% grid_label_counts = cell(size(grid_size, 1), 2); % 2 classes
grid_label_counts = cell(size(grid_size, 1), 1); % 2 classes
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
% for i = 1:size(grid_labels,1)    
%     count_pos = 0;    
%     count_neg = 0;    
%     for j = 1:size(grid_labels,2)        
%         next = grid_labels{i,j};        
%         if ~isempty(next)            
%             if next == 1                
%                 count_pos = count_pos + 1;            
%             elseif next == 0                
%                 count_neg = count_neg + 1;            
%             else
%                 display('ERROR!!!!');            
%             end
%         end
%     end
%     grid_label_counts{i, 1} = count_neg;    
%     grid_label_counts{i, 2} = count_pos;
% end

%% Label the data based on the frequency!
som_generated_labels = [];
each_positive = grid_label_counts(:,2);
each_negative = grid_label_counts(:,1);
for i = 1:size(grid_label_counts, 1)
    if each_positive{i} > each_negative{i}
        s = 1;
    else
        s = 0;
    end 
    som_generated_labels = [som_generated_labels; s];
end

%% Calculate the BMU
idx = som_bmus(sM, vector);

