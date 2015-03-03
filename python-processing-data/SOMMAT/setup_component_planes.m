individual_parts = [];
for i=1:25
    places = zeros(25, 1); places(i) = 1;
    tmp = som_umat(sMap, 'mask', places);
    tmp = tmp(1:2:size(tmp,1),1:2:size(tmp,2));
    individual_parts = [individual_parts, tmp(:)];
end