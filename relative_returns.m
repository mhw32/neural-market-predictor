function returns = relative_returns(market_data, stock_data)
    % Then you need to calculate log relative returns for each stock. 
    % For each weekly time t the log relative return of the stock compared 
    % to the index is calculated according to the following:

    % log(price_stock_t/price_stock_(t-1))-log(price_index_t/price_index_(t-1))

    %%%%%%%%%% LOAD DATA THAT WE NEED %%%%%%%%%%

    % Download stock data for SP500
    if market_data == false % Optional loading of data 
        start_date = '03011950'; % Jan 1st, 2007
        end_date = '30012015'; % Dec 31st, 2012
        market_stock_prices = hist_SP500_stock_data(start_date, end_date);
    else
        market_stock_prices = load(market_data);
    end
    % Load in the individual stock data
    indiv_stock_prices = load(stock_data);
    returns = struct();
    
    %%%%%%%%%% CREATE THE LOG RELATIVE RETURN DATA STRUCTURE %%%%%%%%%%
    calculations = struct([]);
    tickers = fieldnames(indiv_stock_prices.stocks);
    
    idx = 2; counter = 1;
    fund_name = tickers(idx);
    test_object = indiv_stock_prices.stocks.(fund_name{:});
    test_stock = test_object.AdjClose;
    
    % Make sure the test_stock is not empty
    if ~isempty(test_stock) 
        unshift = test_stock(1:length(test_stock)-1);
        shift   = test_stock(2:length(test_stock));
        % Fill out the struct
        calculations(counter).Ticker = fund_name;
        calculations(counter).Date = test_object.Date;
        calculations(counter).Price = log(unshift ./ shift);        
        counter = counter+1;
    end
    
    returns.(fund_name{:}) = calculations;
end





