% This script is used to scrape data from the london stock exchange for
% FTSE Share components.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Download historical insider trading data
% Take out bad noise
tickers={};

% Select all the links we want to pull from
link_sp500 = 'http://www.londonstockexchange.com/exchange/prices-and-markets/international-markets/indices/home/sp-500.html?page=';
link_nasdaq = 'http://www.londonstockexchange.com/exchange/prices-and-markets/international-markets/indices/home/nasdaq-100.html?page=';
link_array = {link_sp500, link_nasdaq};

for l=1:length(link_array)
    buy_data=[];
    page=0;
    tokens=1;
    prevtokens=1;
    disp(strcat('STEP 1 (from source:', num2str(l), ').', 'Downloading historical trading data from: ', link_array{l}));
    while isempty(tokens)==0
        % Get current page url
        url_string = strcat(link_array{l},int2str(page));
        % Read current page url
        str = urlread(url_string);
        % Regex to pull out the right classes 
        expr = '((?<=(<t(d|r) scope="row" class="name">))(.*?)(?=(</td>)))|((?<=(<td>))(\d+\.\d+)(?=(</td>)))';
        tokens = regexp(str,expr,'tokens');
        % Squish it into a column
        tokens = vertcat(tokens{:});

        % Replace all the html formatting with spaces
        pat = '(<[^>]*>)|(&nbsp)';
        for i=1:size(tokens, 1)
            tokens(i, :)=regexprep(tokens(i, :), pat, '');
        end

        % Know when to stop -- since the page numbers go on forever
        if strcmp(class(prevtokens), 'double') == 0
            if isempty(setdiff(tokens, prevtokens))
                break;
            end
        end

        % Concat the data to the growing buffer
        buy_data=[buy_data; tokens];
        page=page+1; % end page
        disp(strcat('Grabbing data from page:',num2str(page))); 
        prevtokens = tokens;
    end

    skip='\d{2,3}\.\d{2}';
    for i=1:size(buy_data, 1)
        if (isempty(regexp(buy_data{i}, skip, 'ONCE')))
            tickers=[tickers; buy_data{i}];
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data from the yahoo stuff.
link_yahoo = 'https://uk.finance.yahoo.com/q/cp?s=%5EFTAS&c=';
buy_data=[];
page=0;
tokens=1;
disp(strcat('STEP 1 (from source:3). Downloading data from: ', link_yahoo));

while isempty(tokens)==0
    % Get current page url
    url_string = strcat(link_yahoo,int2str(page));
    % Read current page url
    str = urlread(url_string);
    % Regex to pull out the right classes 
    expr = '((?<=(<t(d|r) class="yfnc_tabledata1">))(.*?)(?=(</td>)))|((?<=(<td>))(\d+\.\d+)(?=(</td>)))';
    tokens = regexp(str,expr,'tokens');
    tokens = tokens(1:2:end);
    tokens = vertcat(tokens{:});
    
    % Replace all the html formatting with spaces
    pat = '(<[^>]*>)|(&nbsp)';
    for i=1:size(tokens, 1)
        tmp=regexprep(tokens(i, :), pat, '');
        tokens(i) = cellstr(tmp{:}(1:end-2));
    end
    
    buy_data=[buy_data; tokens];
    page=page+1; % end page
    disp(strcat('Grabbing data from page:',num2str(page))); 
end

tickers=[tickers; buy_data];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read from csv file
csv_file = 'companylist.csv';
disp(strcat('STEP 1 (from source:4). Downloading data from csvfile:', csv_file));
csv_data = csvimport(csv_file);
tmpticks = (csv_data(:, 1));
% make them readable
for i=1:length(tmpticks)
    tmpticks{i} = tmpticks{i}(2:end-1);
end
% Adding tickers
disp('Joining all ticker data. Creating ticker object.');
tickers=tickers(1:2:end);
tickers=[tickers; tmpticks];

% Read from other csv file
nyse_file = 'nyselist.csv';
disp(strcat('STEP 1 (from source:5). Downloading data from csvfile:', nyse_file));
nyse_data = csvimport(nyse_file);
nyse_tickers = nyse_data(2:end);
% Adding tickers
tickers=[tickers; nyse_tickers];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tickers=unique(tickers);
tickers=tickers';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create two empty structs
hist_fund_data=struct(); 
stocks=struct();

% The exchanges I'm interested in.
exchanges=[{'NYSE'}, {'NASDAQ'}];

disp('DONE STEP 1. Finished grabbing tickers.');
disp('-----------------------------------------------------');
disp('STEP 2. Start grabbing data for each ticker.');

% This will store that information in the hist hash
for i=1:size(tickers, 2)
    % Convert to string -- delete white space
    fund_name=strtrim(char(tickers(i)));
    disp(strcat('Scrape for ticker:', fund_name));
    % Initialize empty variables
    page=0;
    fund_data=cell(307,1); % Why 307? Probably knows the number of columns coming in.

    nr_table = 6;
    out_table=ones(1, 3);
    passer_var=0; % Indicates if we were in the loop or not
    not_enough_data=0; % Indicates if there is enough data based on first date

    while size(out_table, 2)>2 && page<5 % While we are less than 200 pages
        for j=1:size(exchanges, 2) % For each exchange 
            try
                exchange_name=char(exchanges(1, j)); % Convert name to char
                % Construct string. Start date = 0 since page = 0
                url_string = strcat('http://uk.advfn.com/p.php?pid=financials&btn=istart_date&mode=quarterly_reports&symbol=', exchange_name ,'%3A', fund_name, '&istart_date=', int2str(page));
                % Somehow picks out all the data! What is nr_table used for? 
                out_table  = getTableFromWeb_mod(url_string, nr_table);
                if size(out_table, 1) == 3 % Hack against improper Exchange testing returns
                    error('Improper Exchange Channel');
                end
                
                if page == 0
                    earliest_date = out_table{2,2};
                    if (ensure_nine_years(earliest_date) == false)
                        disp(strcat('Quitting due to < 9 years of data record.'));
                        not_enough_data = 1;
                    else 
                        disp(strcat('Continuing due to sufficient years of data record.'));
                    end 
                end
                
                passer_var=1;
                break;
            catch
                sprintf('NO');
            end
        end

        if passer_var==0
            break;
        end
        
        if not_enough_data==1
            passer_var=0;
            break
        end

        size_ot=size(out_table, 1); % This should be 307.
        % This is the second column padded with zeros. If everything worked, there should be no padding. 
        new_data=[out_table(:, 2); num2cell(zeros(307-size_ot, 1))];
        % Add new_data to growing stack.
        fund_data=[fund_data, new_data];
        page=page+1; 
        disp(strcat(fund_name, ': page', num2str(page)));
    end

    % Still in the loop for each ticker
    if passer_var==1 % If we managed to get data from uk website
        % Set the first column as the variable list
        fund_data(1:size_ot, 1)=out_table(:, 1);

        % If for some reason we've missed some attributes, delete them.
        if size_ot<307
            fund_data((size_ot+1):end, :)=[];
        end

        % Flip the matrix.
        fund_data=fund_data';

        expr='^[0-9]{4}';
        % The size of fund_data's 1st dimension should be 1. 
        for i=2:size(fund_data, 1)
            % Get's the year and applies a regex to pull it out.
            year=regexp(fund_data(i, 2), expr, 'match'); % year{1} pulls it out
            % Convert the year to Q# #### format
            fund_data(i, 1)=cellstr(sprintf('Q%s %s', fund_data{i, 5}, char(year{1})));
        end

        % I removed the 307 constraint. Some pieces are not 307 length.
        % It hashes the ticker to the giant piece of data we pulled.
        disp('Size is correct; adding to the Hash');
        hist_fund_data.(fund_name)=fund_data;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Download historical fund data -- What is this?
        % 6 year total span
        % Track stocks here
        start_date = get_quarter_dates(hist_fund_data.(fund_name){2,1});
        end_date   = get_quarter_dates(hist_fund_data.(fund_name){end,1});
        try
            disp(strcat('PART B: Continuing to grabbing stock data for: ', fund_name));
            stocks.(fund_name) = hist_stock_data(start_date, end_date, fund_name);
            disp('Finished pulling stock data.');
        catch
            continue
        end
    end
end
   