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
    funcs.get_log_revenue_return               = @get_log_revenue_return;
end

function val=get_shares_outstanding(f)
    val = f('shares out (common class only)');
end

function val=get_current_ratio(f)
    val = f('current ratio');
end

function val=get_curr_over_tot_assets(f)
    val = f('total current assets') ./ f('total assets');
end

function val=get_curr_minus_stock_over_tot_asset(f)
    val = (f('total current assets') - f('inventories')) ./ f('total assets');
end

function val=get_curr_liab_over_tot_assets(f)
    val = f('total current liabilities') ./ f('total assets');
end

function val=get_cash_over_curr_liab(f)
    val = f('cash & equivalents') ./ f('total current liabilities');
end

function val=get_cash_over_tot_assets(f)
    val = f('cash & equivalents') ./ f('total assets');
end

function val=get_cash_over_tot_debt(f)
    val = f('cash & equivalents') ./ (f('long-term debt')+f('short-term debt'));
end

function val=get_capital_over_tot_assets(f)
    val = f('working capital') ./ f('total assets');
end

function val=get_long_debt_over_tot_assets(f)
    val = f('long-term debt') ./ f('total assets');
end

function val=get_tot_debt_over_tot_assets(f)
    val = (f('long-term debt') + f('short-term debt')) ./ f('total assets');
end

function val=get_ebidta_over_tot_assets(f)
    val = f('EBITDA') ./ f('total assets');
end

function val=get_income_over_tot_assets(f)
    val = f('total net income') ./ f('total assets');
end

function val=get_tot_sales_over_tot_assets(f)
    val = (f('foreign sales') + f('domestic sales')) ./ f('total assets');
end

function val=get_cash_flow_over_tot_assets(f)
    val = f('net cash from total operating activities') ./ f('total assets');
end

function val=get_cash_flow_over_tot_sales(f)
    val = f('net cash from total operating activities') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_curr_assets_over_tot_sales(f)
    val = f('total current assets') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_capital_over_tot_sales(f)
    val = f('working capital') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_acc_pay_over_tot_sales(f)
    val = f('accounts payable') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_acc_rec_over_tot_sales(f)
    val = f('accounts receivable') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_invent_over_tot_sales(f)
    val = f('inventories') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_cash_over_tot_sales(f)
    val = f('net cash from total operating activities') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_oper_profit(f)
    val = f('gross operating profit');
end

function val=get_return_avg_equity(f)
    val = f('Return on Stock Equity (ROE)');
end

function val=get_tot_liab(f)
    val = f('total liabilities');
end

function val=get_oper_funds(f)
    val = f('net income (total operations)');
end

function val=get_oper_margin(f)
    val = f('net income (total operations)') ./ (f('foreign sales') + f('domestic sales'));
end

function val=get_interest_coverage(f)
    val = f('interest coverage (cont. operations)');
end

function val=get_fix_asset_turnover(f)
    val = (f('foreign sales') + f('domestic sales')) ./ f('purchase of property, plant & equipment');
end

function val=get_tot_asset_turnover(f)
    val = (f('foreign sales') + f('domestic sales')) ./ f('total assets');
end

function val=get_cash_ratio(f)
    val = f('net cash from total operating activities') ./ f('total current liabilities');
end

function val=get_asset_value(f)
    val = f('total assets') - f('total liabilities');
end

function val=get_earnings_per_share(f)
    val = f('Basic EPS - Total');
    % val = f('Basic EPS - Normalized');
end

function val=get_ebitda_per_share(f)
    val = f('EBITDA') ./ f('shares out (common class only)');
end

function val=get_divid_per_share(f)
    val = f('Dividends Paid Per Share (DPS)');
end

function val=get_tot_assets_over_nav(f)
    val = f('total assets') ./ (f('total assets') + f('total liabilities'));
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
function val=get_stock_adj_close_price(s)
    if isempty(s); val = NaN; else val = s.AdjClose; end;
end

function val=get_log_revenue_return(r)
    val = r.AdjClose;
end