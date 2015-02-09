function funcs = calculate_specs
    % Calculate the generic features
    funcs.get_shares_outstanding               = @get_shares_outstanding;
    funcs.get_current_ratio                    = @get_current_ratio;
    funcs.get_curr_over_tot_assets             = @get_curr_over_tot_assets;
    funcs.get_curr_minus_stock_over_tot_assets = @get_curr_minus_stock_over_tot_asset;
    funcs.get_curr_liab_over_tot_assets        = @get_curr_liab_over_tot_assets;
    funcs.get_cash_over_curr_liab              = @get_cash_over_curr_liab;
    funcs.get_cash_over_tot_assets             = @get_cash_over_tot_assets;
    funcs.get_cash_over_tot_debt               = @get_cash_over_tot_debt;
    funcs.get_capital_over_tot_assets          = @get_capital_over_tot_assets;
    funcs.get_long_debt_over_tot_assets        = @get_long_debt_over_tot_assets;
    funcs.get_tot_debt_over_tot_assets         = @get_tot_debt_over_tot_assets;
    funcs.get_ebidta_over_tot_assets           = @get_ebidta_over_tot_assets;
    funcs.get_income_over_tot_assets           = @get_income_over_tot_assets;
    funcs.get_tot_sales_over_tot_assets        = @get_tot_sales_over_tot_assets;
    funcs.get_cash_flow_over_tot_assets        = @get_cash_flow_over_tot_assets;
    funcs.get_cash_flow_over_tot_sales         = @get_cash_flow_over_tot_sales;
    funcs.get_curr_assets_over_tot_sales       = @get_curr_assets_over_tot_sales;
    funcs.get_capital_over_tot_sales           = @get_capital_over_tot_sales;
    funcs.get_acc_pay_over_tot_sales           = @get_acc_pay_over_tot_sales;
    funcs.get_acc_rec_over_tot_sales           = @get_acc_rec_over_tot_sales;
    funcs.get_invent_over_tot_sales            = @get_invent_over_tot_sales;
    funcs.get_cash_over_tot_sales              = @get_cash_over_tot_sales;
    funcs.get_oper_profit                      = @get_oper_profit;
    funcs.get_return_avg_equity                = @get_return_avg_equity;
    funcs.get_tot_liab                         = @get_tot_liab;
    funcs.get_oper_funds                       = @get_oper_funds;
    funcs.get_oper_margin                      = @get_oper_margin;
    funcs.get_interest_coverage                = @get_interest_coverage;
    funcs.get_fix_asset_turnover               = @get_fix_asset_turnover;
    funcs.get_tot_asset_turnover               = @get_tot_asset_turnover;
    funcs.get_cash_ratio                       = @get_cash_ratio;
    funcs.get_asset_value                      = @get_asset_value;
    funcs.get_earnings_per_share               = @get_earnings_per_share;
    funcs.get_ebitda_per_share                 = @get_ebitda_per_share;
    funcs.get_divid_per_share                  = @get_divid_per_share;
    funcs.get_tot_assets_over_nav              = @get_tot_assets_over_nav;
    funcs.get_tot_assets_over_nav_trend        = @get_tot_assets_over_nav_trend;
    % funcs.get_beta                             = @get_beta;
    % Get year over year features
    % funcs.get_earnings_yoy                     = @get_earnings_yoy;
    % funcs.get_earnings_yoy_trend               = @get_earnings_yoy_trend;
    % funcs.get_nav_yoy                          = @get_nav_yoy;
    % funcs.get_nav_yoy_trend                    = @get_nav_yoy_trend;
    % funcs.get_revenue_yoy                      = @get_revenue_yoy;
    % funcs.get_revenue_yoy_trend                = @get_revenue_yoy_trend;
    % Get stock price features
    funcs.get_stock_adj_close_price            = @get_stock_adj_close_price;
    % funcs.get_log_revenue_return               = @get_log_revenue_return;
    funcs.slice_by_date                        = @slice_by_date;
    funcs.find_closest_date                    = @find_closest_date;
    funcs.get_date                             = @get_date;
    funcs.do_size_check                        = @do_size_check;
end

function idx=find_me(cells, tag)
    idx=find(ismember(cells, tag));
end

function val=get_shares_outstanding(k, v)
    idx = find_me(k, 'shares out (common class only)');
    val = v(:, idx);
end

function val=get_current_ratio(k, v)
    idx = find_me(k, 'current ratio');
    val = v(:, idx);
end

function val=get_curr_over_tot_assets(k, v)
    num_idx = find_me(k, 'total current assets');
    den_idx = find_me(k, 'total assets');
    val = v(:,num_idx) ./ v(:,den_idx);
end

function val=get_curr_minus_stock_over_tot_asset(k, v)
    idx_one = find_me(k, 'total current assets');
    idx_two = find_me(k, 'inventories');
    idx_three = find_me(k, 'total assets');
    val = (v(:,idx_one) - v(:,idx_two)) ./ v(:,idx_three);
end

function val=get_curr_liab_over_tot_assets(k, v)
    idx_top = find_me(k, 'total current liabilities');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_cash_over_curr_liab(k, v)
    idx_top = find_me(k, 'cash & equivalents');
    idx_bottom = find_me(k, 'total current liabilities');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_cash_over_tot_assets(k, v)
    idx_top = find_me(k, 'cash & equivalents');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_cash_over_tot_debt(k, v)
    idx_one = find_me(k, 'cash & equivalents');
    idx_two = find_me(k, 'long-term debt');
    idx_three = find_me(k, 'short-term debt');
    val = v(:,idx_one) ./ (v(:,idx_two)+v(:,idx_three));
end

function val=get_capital_over_tot_assets(k, v)
    idx_top = find_me(k, 'working capital');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_long_debt_over_tot_assets(k, v)
    idx_top = find_me(k, 'long-term debt');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_tot_debt_over_tot_assets(k, v)
    idx_one = find_me(k, 'long-term debt');
    idx_two = find_me(k, 'short-term debt');
    idx_three = find_me(k, 'total assets');
    val = (v(:,idx_one) + v(:,idx_two)) ./ v(:,idx_three);
end

function val=get_ebidta_over_tot_assets(k, v)
    idx_top = find_me(k, 'EBITDA');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_income_over_tot_assets(k, v)
    idx_top = find_me(k, 'total net income');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_tot_sales_over_tot_assets(k, v)
    idx_one = find_me(k, 'foreign sales');
    idx_two = find_me(k, 'domestic sales');
    idx_three = find_me(k, 'total assets');
    val = (v(:,idx_one) + v(:,idx_two)) ./ v(:,idx_three);
end

function val=get_cash_flow_over_tot_assets(k, v)
    idx_top = find_me(k, 'net cash from total operating activities');
    idx_bottom = find_me(k, 'total assets');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_cash_flow_over_tot_sales(k, v)
    idx_one = find_me(k, 'net cash from total operating activities');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_curr_assets_over_tot_sales(k, v)
    idx_one = find_me(k, 'total current assets');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_capital_over_tot_sales(k, v)
    idx_one = find_me(k, 'working capital');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_acc_pay_over_tot_sales(k, v)
    idx_one = find_me(k, 'accounts payable');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_acc_rec_over_tot_sales(k, v)
    idx_one = find_me(k, 'accounts receivable');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_invent_over_tot_sales(k, v)
    idx_one = find_me(k, 'inventories');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_cash_over_tot_sales(k, v)
    idx_one = find_me(k, 'net cash from total operating activities');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_oper_profit(k, v)
    idx = find_me(k, 'gross operating profit');
    val = v(:,idx);
end

function val=get_return_avg_equity(k, v)
    idx = find_me(k, 'Return on Stock Equity (ROE)');
    val = v(:,idx);
end

function val=get_tot_liab(k, v)
    idx = find_me(k, 'total liabilities');
    val = v(:,idx);
end

function val=get_oper_funds(k, v)
    idx = find_me(k, 'net income (total operations)');
    val = v(:,idx);
end

function val=get_oper_margin(k, v)
    idx_one = find_me(k, 'net income (total operations)');
    idx_two = find_me(k, 'foreign sales');
    idx_three = find_me(k, 'domestic sales');
    val = v(:,idx_one) ./ (v(:,idx_two) + v(:,idx_three));
end

function val=get_interest_coverage(k, v)
    idx = find_me(k, 'interest coverage (cont. operations)');
    val = v(:,idx);
end

function val=get_fix_asset_turnover(k, v)
    idx_one = find_me(k, 'foreign sales');
    idx_two = find_me(k, 'domestic sales');
    idx_three = find_me(k, 'purchase of property, plant & equipment');
    val = (v(:,idx_one) + v(:,idx_two)) ./ v(:,idx_three);
end

function val=get_tot_asset_turnover(k, v)
    idx_one = find_me(k, 'foreign sales');
    idx_two = find_me(k, 'domestic sales');
    idx_three = find_me(k, 'total assets');
    val = (v(:,idx_one) + v(:,idx_two)) ./ v(:,idx_three);
end

function val=get_cash_ratio(k, v)
    idx_top = find_me(k, 'net cash from total operating activities');
    idx_bottom = find_me(k, 'total current liabilities');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_asset_value(k, v)
    idx_one = find_me(k, 'total assets');
    idx_two = find_me(k, 'total liabilities');
    val = v(:,idx_one) - v(:,idx_two);
end

function val=get_earnings_per_share(k, v)
    idx = find_me(k, 'Basic EPS - Total');
    val = v(:,idx);
    % val = f('Basic EPS - Normalized');
end

function val=get_ebitda_per_share(k, v)
    idx_top = find_me(k, 'EBITDA');
    idx_bottom = find_me(k, 'shares out (common class only)');
    val = v(:,idx_top) ./ v(:,idx_bottom);
end

function val=get_divid_per_share(k, v)
    idx = find_me(k, 'Dividends Paid Per Share (DPS)');
    val = v(:,idx);
end

function val=get_tot_assets_over_nav(k, v)
    idx_one = find_me(k, 'total assets');
    idx_two = find_me(k, 'total liabilities');
    val = v(:,idx_one) ./ (v(:,idx_one) + v(:,idx_two));
end

% Get over time features
%{

function val=get_tot_assets_over_nav_trend(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function val=get_beta(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function val=get_earnings_yoy(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function get_earnings_yoy_trend(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function get_nav_yoy(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function get_nav_yoy_trend(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function get_revenue_yoy(f)
    % TO-DO: B/C THIS IS OVER TIME
end

function get_revenue_yoy_trend(f)
    % TO-DO: B/C THIS IS OVER TIME
end
%}

% Get stock price features
% DEPRECATED ---------------------------------------
%{ 
function val=get_stock_adj_close_price(s, dates)
    begun = false;
    unitlength = size(dates, 1);
    dates = get_date(dates);
    if isempty(s)
        val = emptyMatrix(unitlength);
    else
        filterP = {};
        full_stock_dates = datetime(s.Date(1:end));
        full_stock_prices = s.AdjClose(1:end);

        start_dates = datetime(dates{1});
        end_dates   = datetime(dates{2});
        for i=1:length(start_dates)
            curr_start = start_dates(i);
            curr_end   = end_dates(i);

            selection = full_stock_dates(full_stock_dates > curr_start);
            Pselect = full_stock_prices(full_stock_dates > curr_start);
            Pselect = Pselect(selection < curr_end);
            
            filterP = [filterP; {Pselect}];
        end
        val = filterP;
    end
end
%}

function val = find_closest_date(find_time, array_time)
    % array_time in our case = all of the daily S&P500 data;
    % find_time = the particular time we want to find. 
    [~,idx] = min(abs(datetime(array_time)-datetime(find_time)));
    val = idx; % This is the right idx -- we need the price here.
end

% Given all of the stock data, this allows you to pick out a piece of it
function [d, p]=slice_by_date(start_date, end_date, returns)
    full_stock_dates = datetime(returns.Date);
    full_stock_prices = returns.LogReturn;
    start_date = datetime(start_date);
    end_date   = datetime(end_date);
    
    selection = full_stock_dates > start_date;
    date_slice = full_stock_dates(selection);
    price_slice = full_stock_prices(selection);
    p = price_slice(date_slice < end_date);
    d  = date_slice(date_slice < end_date);
end

% DEPRECATED ---------------------------------------
%{
function val=get_log_revenue_return(m, s, dates)
    unitlength = size(dates, 1);
    dates = get_date(dates);
    if isempty(s)
        val = emptyMatrix(unitlength);
    elseif length(s.AdjClose) <= 2
        val = emptyMatrix(unitlength);
    else
        r = relative_returns(m, s);
        % hacky soln rite here
        if isa(r, 'double') && isnan(r)
            val = emptyMatrix(unitlength);
        else
            filterP = {};
            full_stock_dates = datetime(r.Date);
            full_stock_prices = r.Price;

            start_dates = datetime(dates{1});
            end_dates   = datetime(dates{2});
            for i=1:length(start_dates)
                curr_start = start_dates(i);
                curr_end   = end_dates(i);

                selection = full_stock_dates(full_stock_dates > curr_start);
                Pselect = full_stock_prices(full_stock_dates > curr_start);
                Pselect = Pselect(selection < curr_end);
                filterP = [filterP; {Pselect}];
            end
            val = filterP;
        end
    end
end
%}

function val=get_date(dates)
    fake_dates = dates;
    start_date = [];
    end_date   = [];
    for i=1:length(fake_dates)
        quarter = fake_dates(i);
        [s, e] = wu_get_quarter_dates(quarter{:});
        start_date = [start_date; s];
        end_date   = [end_date; e];
    end
    val = {start_date end_date};
end

function val=emptyMatrix(number)
    val = [];
    b = zeros(0,1);
    c = [];
    for i=1:number
        val = [val; {c*b}];
    end
end

%{ 
function [p, r]=do_size_check(p, r)
    for i=1:size(p, 1)
        if size(p{i}, 1) > 0 && size(p{i}, 1) == size(r{i}, 1) + 1
            % Remove one from p
            p{i} = p{i}(2:end);
        end
    end
end
%} 
