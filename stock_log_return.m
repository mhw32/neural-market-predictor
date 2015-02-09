%{ This function calculates the log return for stock prices %}

function returns = stock_log_return(stock_data)
    % log(price_stock_t/price_stock_(t-1))
    calculations = struct([]);
    % Create storage 
    dates  = stock_data.Date;
    prices = stock_data.AdjClose;
    % Create skewed data
    leftshift  = prices(1:end-1);
    rightshift = prices(2:end);
    % Store the calculations
    calculations(1).Date = dates(2:end);
    calculations(1).LogReturn = log(leftshift ./ rightshift);
    % Return the object
    returns = calculations;
end