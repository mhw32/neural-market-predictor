function returns = relative_returns(market_stock_prices, indiv_stock_prices)
    % Then you need to calculate log relative returns for each stock. 
    % For each weekly time t the log relative return of the stock compared.
    % to the index is calculated according to the following:

    % log(price_stock_t/price_stock_(t-1))-log(price_index_t/price_index_(t-1))
    returns = struct();
    
    %%%%%%%%%% CREATE THE LOG RELATIVE RETURN DATA STRUCTURE %%%%%%%%%%
    
    calculations = struct([]);
    counter = 1;
    
    % Initial variable setting
    test       = indiv_stock_prices;
    test_date  = test.Date;
    test_stock = test.AdjClose;
    
    % Format index data into nice things.
    indice       = market_stock_prices;
    indice_date  = indice.Date;
    indice_stock = indice.AdjClose;
    
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
        
        % The problem is that we can't assume the stock data 
        % will be continous, so we have to fill in random NaN
        if length(test_stock) ~= length(indice_stock(I_start:I_end))
            % oops.. this is a big error! They should be the same size
            % I don't know why this is happening, so for now let's just
            % ignore it 
            disp('!!!!!!!!!!!!! FATALITY: Bug in scrapping stock data -- incomplete');
            returns = NaN;
            return 
        end
        
        indice_unshift = indice_stock(I_start:I_end-1);
        indice_shift = indice_stock(I_start+1:I_end);       

        % Now we can use the data we made. Fill out the struct.
        calculations(counter).Date = test_date(1:end-1);
        % Use the formula for log relative return.
        calculations(counter).Price = log(unshift ./ shift) - log(indice_unshift ./ indice_shift); 
    end
    % Create outer hash.
    returns = calculations;
end





