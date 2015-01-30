% This script is intended to be used to extract S&P500 Prices. 
% to be an indicator of the general market movements.

function market_stocks = hist_SP500_stock_data(start_date, end_date, freq)
    
    % Default to week if not given
    if nargin == 2
        freq = 'd';
    end
    
    % Select the ticker ^GSPC.
    fund_name = 'GSPC';
    market_stocks = struct();

    % Split this up for start_date.
    bd = start_date(1:2);       % beginning day
    bm = sprintf('%02d',str2double(start_date(3:4))-1); % beginning month
    by = start_date(5:8);       % beginning year

    % Split this up for end_date.
    ed = end_date(1:2);         % ending day
    em = sprintf('%02d',str2double(end_date(3:4))-1);   % ending month
    ey = end_date(5:8);         % ending year

    % Using the data broken down from above, create a url to read from
    link_market = strcat('http://real-chart.finance.yahoo.com/table.csv?s=%5E'...
            ,fund_name,'&a=',bm,'&b=',bd,'&c=',by,'&d=',em,'&e=',ed,'&f=',...
            ey,'&g=',freq,'&ignore=.csv');
    disp(link_market);
        
    % Read the URL
    [data, status] = urlread(link_market);

    if status % Only do this on success
        % Organize the data using the comma delimiter
        [date, op, high, low, cl, volume, adj_close] = ...
            strread(data(43:end),'%s%s%s%s%s%s%s','delimiter',',');
        % We should only do Fridays to be consistent!
        dayofweek = weekday(date) == 6;
        market_stocks(1).Ticker = fund_name; % Ticker Symbol
        market_stocks(1).Date = date(dayofweek); % Data date
        market_stocks(1).AdjClose = str2double(adj_close(dayofweek)); % Adjusted Closing Price
    end

    % Don't really need this but nice garbage collection
    clear date op high low cl volume adj_close dayofweek data status
end
    
