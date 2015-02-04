function [ start_date, end_date ] = wu_get_quarter_dates( sdate )
%get_quarter_dates Creates variables containing the dates corresponding to
%the beginning and the end of the quarter specified in sdate
%   sdate - char in format 'Q1 2007'

%   start_date, end_date = char in format '01032007'

    Qdate=sdate(2);
    Qyear=sdate((end-3):end);

    if Qdate=='4'
        start_date = strcat(Qyear, '-01-01'); 
        end_date = strcat(Qyear, '-03-31'); 
    elseif Qdate=='1'
        start_date = strcat(Qyear, '-04-01'); 
        end_date = strcat(Qyear, '-06-30');      
    elseif Qdate=='2'
        start_date = strcat(Qyear, '-07-01'); 
        end_date = strcat(Qyear, '-09-30');  
    elseif Qdate=='3'
        start_date = strcat(Qyear, '-10-01'); 
        end_date = strcat(Qyear, '-12-31');     
    end

end

