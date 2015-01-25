function [ start_date, end_date ] = get_quarter_dates( sdate )
%get_quarter_dates Creates variables containing the dates corresponding to
%the beginning and the end of the quarter specified in sdate
%   sdate - char in format 'Q1 2007'

%   start_date, end_date = char in format '01032007'

    Qdate=sdate(2);
    Qyear=sdate((end-3):end);

    if Qdate=='4'
        start_date = strcat('0101', Qyear); 
        end_date = strcat('3103', Qyear); 
    elseif Qdate=='1'
        start_date = strcat('0104', Qyear); 
        end_date = strcat('3006', Qyear);      
    elseif Qdate=='2'
        start_date = strcat('0107', Qyear); 
        end_date = strcat('3009', Qyear);  
    elseif Qdate=='3'
        start_date = strcat('0110', Qyear); 
        end_date = strcat('3112', Qyear);     
    end

end

