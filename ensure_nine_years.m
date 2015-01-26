function [ truth ] = ensure_nine_years( sdate )
% This ensures that at least 9 years data are present. I assume here that
% data is consecutive. 

%   sdate - char in format 'Q1 2007'
    Qdate = strsplit(sdate, '/');
    Qyear = Qdate(1);
    
    c = clock();
    thisYear = c(1);
    
    if (thisYear - str2num(Qyear{:}) >= 9)
        truth = true;
    else
        truth = false;
    end
end