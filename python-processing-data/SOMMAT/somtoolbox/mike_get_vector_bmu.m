function label = mike_get_vector_bmu(vector, sMap)
    %% Calculate the BMU
    idx = som_bmus(sMap, vector);
    label = idx;
end