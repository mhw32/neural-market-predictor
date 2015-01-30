function returns = relative_returns(market_data, stock_data)
    % Then you need to calculate log relative returns for each stock. 
    % For each weekly time t the log relative return of the stock compared.
    % to the index is calculated according to the following:

    % log(price_stock_t/price_stock_(t-1))-log(price_index_t/price_index_(t-1))

    %%%%%%%%%% LOAD DATA THAT WE NEED %%%%%%%%%%

    % Download stock data for SP500.
    if market_data == false % Optional loading of data .
        start_date = '03011950'; % Jan 1st, 2007
        end_date = '30012015'; % Dec 31st, 2012
        market_stock_prices = hist_SP500_stock_data(start_date, end_date);
    else
        market_stock_prices = load(market_data);
    end
    
    % Load in the individual stock data.
    indiv_stock_prices = load(stock_data);
    returns = struct();
    
    %%%%%%%%%% CREATE THE LOG RELATIVE RETURN DATA STRUCTURE %%%%%%%%%%
    
    calculations = struct([]);
    tickers = fieldnames(indiv_stock_prices.stocks);
    
    % Initial variable setting
    idx = 2; counter = 1;
    fund_name = tickers(idx);
    test = indiv_stock_prices.stocks.(fund_name{:});
    test_date = test.Date;
    test_stock = test.AdjClose;
    
    % Format index data into nice things.
    indice       = market_stock_prices;
    indice_date  = indice.market.Date;
    indice_stock = indice.market.AdjClose;
    
    % Make sure the test_stock is not empty.
    if ~isempty(test_stock) 
        % Create askewed data.
        unshift = test_stock(1:end-1);
        shift   = test_stock(2:end);
        % Get the askewed date.
        begin_date    = test_date(1);
        finish_date   = test_date(end);
        
        % Get the same region of data from index stock prices.
        I_start = find(ismember(indice_date, begin_date));
        I_end   = find(ismember(indice_date, finish_date));
        indice_unshift = indice_stock(I_start:I_end-1);
        indice_shift = indice_stock(I_start+1:I_end);
        
        % Now we can use the data we made. Fill out the struct.
        calculations(counter).Ticker = fund_name;
        calculations(counter).Date = test_date(1:end-1);
        % Use the formula for log relative return.
        calculations(counter).Price = log(unshift ./ shift) - log(indice_unshift ./ indice_shift);   
        % Increase index.
        counter = counter+1;
    end
    % Create outer hash.
    returns.(fund_name{:}) = calculations;
end





