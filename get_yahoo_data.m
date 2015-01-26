% Let's start with this one. We will try to shift get_data.m
% https://uk.finance.yahoo.com/q/cp?s=%5EFTAS

% Rules:
% 1. Select stocks with at least 9 years of data available between January, 1 1990 and today
% 2. Price data should be collected weekly on Fridays after market close. Data for stocks that 
%    were not traded during a particular week should be excluded.

% Download historical download from EFTA finance.
buy_data = [];
page = 0;
tokens = 1;

while isempty(tokens) == 0
    % Get current page url from yahoo finance
    url_string = strcat('https://uk.finance.yahoo.com/q/cp?s=%5EFTAS&c=',int2str(page));
    str = urlread(url_string);
    expr = '((?<=(<t(d|r) class="(yfnc_tabledata1)">))(.*?)(?=(</td>)))|((?<=(<td>))(\d+\.\d+)(?=(</td>)))';
    tokens = regexp(str,expr,'tokens');
    tokens = tokens(1:2:end);
    tokens = vertcat(tokens{:});

    % Replace all the html formatting with spaces
    pat = '(<[^>]*>)|(&nbsp)';
    for i = 1:size(tokens, 1)
        tokens(i, :) = regexprep(tokens(i, :), pat, '');
    end

    % Concat the data to the growing buffer
    buy_data = [buy_data; tokens];
    % Do this for all pages
    page = page+1 % end page
end

% A ticker is GOOG. This is an array of company signs
% We can skip all that other processing because I'm just
% taking everything for now. Later I need make sure at least
% 9 years history are there for these.
tickers = buy_data;
for i = 1:size(tickers, 1)
    tickers{i} = tickers{i}(1:end-2);
end
tickers = tickers';

%For each stock within the historical trading data download all historical
%fundamental data. % Remove duplicates
tickers = unique(tickers); 

% Create two empty structs
hist_fund_data = struct();
stocks = struct();

% The exchanges I'm interested in.
exchanges = [{'NYSE'}, {'NASDAQ'}, {'LSE'}, {'USOTC'}];

%%%%%%%%%%%%%%%%%%%%%%%%% COPIED PART %%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:size(tickers, 2)
    % Convert to string
    fund_name=char(tickers(i));
    
    % Initialize empty variables
    storage_size = 307;
    page=0;
    fund_data=cell(storage_size,1); % Why 307? Probably knows the number of columns coming in.
    
    nr_table = 6;
    safety_check = 2;
    out_table=ones(1, 3);
    passer_var=0; % Indicates if we were in the loop or not
    
    while size(out_table, 2)>2 && page<2 % While we are less than 50 pages
        for j=1:size(exchanges, 2) % For each exchange 
            try
                exchange_name=char(exchanges(1, j)); % Convert name to char
                % Construct string. Start date = 0 since page = 0
                url_string = strcat('http://uk.advfn.com/p.php?pid=financials&btn=istart_date&mode=quarterly_reports&symbol=', exchange_name ,'%3A', fund_name, '&istart_date=', int2str(page));
                
                % Check that this table contains info and not just the url
                % is functional
                test = getTableFromWeb_mod(url_string, safety_check);
                if strcmp(test{1}, 'Unknown symbol')
                    error('functional url but empty table.')
                end
                
                % Somehow picks out all the data from table 6!
                out_table = getTableFromWeb_mod(url_string, nr_table);
                passer_var=1;
                break;
            catch
                sprintf('NO');
            end
        end
        
        if passer_var==0
            break;
        end
   
        size_ot=size(out_table, 1); % This should be 307.
        % This is the second column padded with zeros. If everything worked, there should be no padding. 
        new_data=[out_table(:, 2); num2cell(zeros(storage_size-size_ot, 1))];
        
        % Add new_data to growing stack.
        fund_data=[fund_data, new_data];
        
        page = page+1
    end
end