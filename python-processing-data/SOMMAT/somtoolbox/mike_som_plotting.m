disp('Loading the feature vectors and labels');
load('../SOM_tags.mat');
load('../small-labeled-vectors.mat');
disp('Setup the SOM data structure');
sData = som_data_struct(vectors, 'comp_names', tags);
% Make sure the inputted vectors are double
sData.data = double(sData.data);
% Make sure the labels are strings
sData.labels = num2cell(labels');

% Normalize the data
sData = som_normalize(sData,'var');
% Actually train the map
sMap = som_make(sData, 'msize', [100 100]);

% Use the labels on the map
disp('Automatically label the SOM through voting');
sMap = som_autolabel(sMap,sData,'vote');
som_show(sMap,'umat','all', 'empty', 'labels', 'norm', 'd');
disp('Plot the U-Matrix and the Component Labels');
som_show_add('label',sMap.labels,'textsize',8,'textcolor','r','subplot',2);
