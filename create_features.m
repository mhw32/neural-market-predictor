function [full_indexes, full_features] = create_features(hist_fund_data, stock_data, market_data)   
   disp('BEGIN FEATURES FUNCTION: Reformatting data into hashed feature vector for learning');
   %%%%%%%%%% LET'S DEFINE A GLOBAL STRUCTURE %%%%%%%%%% 
   full_indexes  = [];
   full_features = [];
   tickers = fieldnames(hist_fund_data);
   
   % Loop through all the tickers
   for k=1:length(tickers)
       %%%%%%%%%% FIRST LET'S WORK ON THE EXISTING FEATURES %%%%%%%%%%
       % Use this just to get a list of the variables
       dummy_var = tickers{k};
       disp(strcat('Ticker #', num2str(k), ': Beginning calculations for-', dummy_var));

       % Let's make use of all the ratios existing from the scrape
       existing_features_tag  = hist_fund_data.(dummy_var)(1, :)';
       existing_features_data = hist_fund_data.(dummy_var)(2:end, :); 
       
       % Find the ratios calculations section
       ratio_start = find(strcmp(existing_features_tag, 'RATIOS CALCULATIONS')); 
       % If it isn't there, we don't want this particular stock, so skip it
       if isempty(ratio_start)
           disp(strcat('ERROR: Unable to find RATIOS CALCULATIONS section. Skipping stock:', dummy_var));
           continue; 
       end
       
       % Copy all of the existing before subsection to be for later use
       allk = existing_features_tag';
       alldates = existing_features_data(:, 1);
       allv = existing_features_data;
       allv = cellfun(@(x)str2double(x), allv, 'UniformOutput', false);
       allv = cell2dubs(allv);
       
       % Back to shaving existing features
       existing_features_tag = existing_features_tag(ratio_start:end);
       existing_features_data = existing_features_data(:, ratio_start:end);

       % These are the features that don't work
       bad_features_tag = {'RATIOS CALCULATIONS', 'PROFIT MARGINS', 'NORMALIZED RATIOS', ...
            'SOLVENCY RATIOS', 'EFFICIENCY RATIOS', 'ACTIVITY RATIOS', 'LIQUIDITY RATIOS', ...
            'CAPITAL STRUCTURE RATIOS', 'PROFITABILITY', 'AGAINST THE INDUSTRY RATIOS'};
       % Find these indices and remove them
       bad_features_index = [];
       A = existing_features_tag; B = bad_features_tag';
       % Loop through the save the index 
       for i=1:length(A)
           for j=1:length(B)
               tmp1 = A(i); tmp2 = B(j);
               if strcmp(tmp1{:}, tmp2{:}) == true
                   bad_features_index = [bad_features_index i];
               end
           end
       end
       existing_features_tag(bad_features_index)  = [];
       existing_features_data(:, bad_features_index) = [];

       % Fill out the part1_features structure
       part_one_key   = existing_features_tag';
       part_one_value = existing_features_data;
       part_one_value = cellfun(@(x)str2double(x), part_one_value, 'UniformOutput', false);
       part_one_value = cell2dubs(part_one_value);
       
       % This is the structure for the individual Ticker feature
       disp(strcat('Ticker #', num2str(k), ': PART 1 complete for-', dummy_var)); 
       
       %%%%%%%%%% NOW LET'S WORK ON THE NEW FEATURES %%%%%%%%%%    
       % All this stuff is really repetitive... I don't really know how to do
       % it better
       new_features_tags = {'Shares Outstanding', 'Current Ratio', 'Current Assets / Total Assets', ...
           '(Current Assets - Stock) / Total Assets', 'Current Liabilities / Total Assets', ...
           'Cash / Current Liabilities', 'Cash / Total Assets', 'Cash / Total Debt', ...
           'Net Op. Work. Capital / Total Assets', 'Long Term Debt / Total Assets', ...
           'Total Debt / Total Assets', 'EBIDTA / Total Assets', 'Net Income / Total Assets', ...
           'Total Sales / Total Assets', 'Operating Cash Flow / Total Assets', ...
           'Operating Cash Flow / Total Sales', 'Current Assets / Total Sales', ...
           'Net Op. Work. Capital / Total Sales', 'Accounts Payable / Total Sales', ...
           'Accounts Receivable / Total Sales', 'Inventory / Total Sales', ...
           'Cash / Total Sales', 'Operating Profit', 'Return on Average Equity', ...
           'Total Liabilities', 'Funds from Operations', 'Operating Margin', ...
           'Interest Coverage', 'Fixed Asset Turnover', 'Total Asset Turnover', ...
           'Cash Ratio', 'Net Asset Value ', 'Earnings Per Share', 'EBITDA Per Share', ...
           'Dividend Per Share', 'Total Assets / NAV'};

       % Find all the functions 
       fs = calculate_specs;
       new_features_data = [fs.get_shares_outstanding(allk, allv), fs.get_current_ratio(allk, allv), fs.get_curr_over_tot_assets(allk, allv), ... 
           fs.get_curr_minus_stock_over_tot_assets(allk, allv), fs.get_curr_liab_over_tot_assets(allk, allv), ...
           fs.get_cash_over_curr_liab(allk, allv), fs.get_cash_over_tot_assets(allk, allv), fs.get_cash_over_tot_debt(allk, allv), ...
           fs.get_capital_over_tot_assets(allk, allv), fs.get_long_debt_over_tot_assets(allk, allv), ...
           fs.get_tot_debt_over_tot_assets(allk, allv), fs.get_ebidta_over_tot_assets(allk, allv), ...
           fs.get_income_over_tot_assets(allk, allv), fs.get_tot_sales_over_tot_assets(allk, allv), ...
           fs.get_cash_flow_over_tot_assets(allk, allv), fs.get_cash_flow_over_tot_sales(allk, allv), ...
           fs.get_curr_assets_over_tot_sales(allk, allv), fs.get_capital_over_tot_sales(allk, allv), ...
           fs.get_acc_pay_over_tot_sales(allk, allv), fs.get_acc_rec_over_tot_sales(allk, allv), ...
           fs.get_invent_over_tot_sales(allk, allv), fs.get_cash_over_tot_sales(allk, allv), fs.get_oper_profit(allk, allv), ...
           fs.get_return_avg_equity(allk, allv), fs.get_tot_liab(allk, allv), fs.get_oper_funds(allk, allv), fs.get_oper_margin(allk, allv), ...
           fs.get_interest_coverage(allk, allv), fs.get_fix_asset_turnover(allk, allv), fs.get_tot_asset_turnover(allk, allv), ...
           fs.get_cash_ratio(allk, allv), fs.get_asset_value(allk, allv), fs.get_earnings_per_share(allk, allv), ...
           fs.get_ebitda_per_share(allk, allv), fs.get_divid_per_share(allk, allv), fs.get_tot_assets_over_nav(allk, allv)]; 

       % Let's do part 2 of the features
       part_two_key   = new_features_tags;
       part_two_value = new_features_data;
       
       disp(strcat('Ticker #', num2str(k), ': PART 2 complete for-', dummy_var)); 
       % Let's merge the two hash tables -- not very readable
       
       merged_keys  = [part_one_key part_two_key];
       merged_value = [part_one_value part_two_value];
       
       % Add three descriptive measures: ticker, start-date, end-date
       %descriptive_tags = {'Ticker', 'Start Date', 'End Date'};
       %merged_keys = [descriptive_tags merged_keys];
       %tmp = fs.get_date(alldates);
       %dummy_value = val2vec(dummy_var, size(merged_value, 1));
       %merged_value = [dummy_value tmp{1} tmp{2} merged_value];
       %disp(merged_value);
       
       % Start the information for stock + returns
       stock_feature_tags = {'Stock Adjusted Closing Price','Log Revenue Return'}; 
       price_data = fs.get_stock_adj_close_price(stock_data.(dummy_var), alldates);
       return_data = fs.get_log_revenue_return(market_data, stock_data.(dummy_var), alldates);
       % Function to ensure price and returns are the same size
       [price_data, return_data] = fs.do_size_check(price_data, return_data);
       
       % For each row in merged_value, we know that there are a group of
       % stocks. Duplicate each row for each stock val.
       merged_keys = [merged_keys stock_feature_tags];
       tmp = [];
       for i=1:size(merged_value, 1)
           if size(price_data{i}, 1) > 0 && size(return_data{i}, 1) > 0
               for j=1:size(price_data{i}, 1)
                   tmp = [tmp; merged_value(i, :) price_data{i}(j) return_data{i}(j)];
               end
           else
               tmp = [tmp; merged_value(i, :) NaN NaN]; 
           end
       end
       merged_value = tmp;
       
       if k == 1
           full_indexes = merged_keys;
       end
       % Fill out the full_features structure
       full_features = [full_features; merged_value];
   end
   % Compile it to the outer hash
   disp('END FEATURES FUNCTION: successful termination'); 
end

function tmp=val2vec(value, number)
    tmp = [];
    for i=1:number
        tmp = [tmp; value];
    end
end

function val=cell2dubs(v)
    val = [];
    for i=1:size(v, 2)
       tmp = [v{:, i}]';
       val = [val tmp];
    end
end



