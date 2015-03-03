%    The basic usage of the SOM Toolbox proceeds like this: 
%      1. construct data set
%      2. normalize it
%      3. train the map
%      4. visualize map
%      5. analyse results


% Construct the data in the proper manner
sData = som_data_struct(vectors, 'name', 'financial data w/ small CUSUM labels', 'comp_names', tags);

% Make sure the inputted vectors are double
sData.data = double(sData.data);

% Make sure the labels are strings
sData.labels = num2cell(int2str(labels'));

% Normalize the data
sData = som_normalize(sData,'var');

% Actually train the map
sMap = som_make(sData);

% Use the labels on the map
sMap = som_autolabel(sMap,sData,'vote');

% Plot the UMatrix
som_show(sMap,'norm','d')
som_show(sMap,'umat','all','comp',[26], 'norm', 'd');

% But the problem is that the UMatrix is very unclear, aka we must now do a log of it. 
U = som_umat(sMap);
logU = log(U);
% Visualize this
h=som_cplane([sMap.topol.lattice,'U'],sMap.topol.msize, logU(:));

