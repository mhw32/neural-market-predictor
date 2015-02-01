function features_data = create_features(hist_fund_data, stock_data, market_data)   
   %%%%%%%%%% LET'S DEFINE A GLOBAL STRUCTURE %%%%%%%%%% 
   full_features = struct();

   %%%%%%%%%% FIRST LET'S WORK ON THE EXISTING FEATURES %%%%%%%%%%
   % Use this just to get a list of the variables
   dummy_var = 'AA';
   ratio_start = 212;
   
   % Let's make use of all the ratios existing from the scrape
   existing_features_tag  = hist_fund_data.(dummy_var)(1, :)';
   existing_features_data = hist_fund_data.(dummy_var)(2:end, :);   
   
   % Copy all of the existing before subsection to be for later use
   all_existing_features_tag = existing_features_tag;
   all_existing_features_data = existing_features_data;
   all_keys = all_existing_features_tag';
   all_values = [];
   for i=1:length(all_existing_features_data)
       all_values = [all_values cellfun(@(x)str2double(x), {all_existing_features_data(:, i, 1)}, 'UniformOutput', false)];
   end
   all_f = containers.Map(all_keys, all_values);
   
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
   keyz = existing_features_tag';
   valuez = [];
   for i=1:length(existing_features_data)
       valuez = [valuez {existing_features_data(:, i, 1)}];
   end
   % This is the structure for the individual Ticker feature
   part1_features = containers.Map(keyz, valuez);
   
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
       'Dividend Per Share', 'Total Assets / NAV', 'Stock Adjusted Closing Price', ...
       'Log Revenue Return'}; 
   
   % Find all the functions 
   fs = calculate_specs;
   new_features_data = {fs.get_shares_outstanding(all_f), fs.get_current_ratio(all_f), fs.get_curr_over_tot_assets(all_f), ... 
       fs.get_curr_minus_stock_over_tot_assets(all_f), fs.get_curr_liab_over_tot_assets(all_f), ...
       fs.get_cash_over_curr_liab(all_f), fs.get_cash_over_tot_assets(all_f), fs.get_cash_over_tot_debt(all_f), ...
       fs.get_capital_over_tot_assets(all_f), fs.get_long_debt_over_tot_assets(all_f), ...
       fs.get_tot_debt_over_tot_assets(all_f), fs.get_ebidta_over_tot_assets(all_f), ...
       fs.get_income_over_tot_assets(all_f), fs.get_tot_sales_over_tot_assets(all_f), ...
       fs.get_cash_flow_over_tot_assets(all_f), fs.get_cash_flow_over_tot_sales(all_f), ...
       fs.get_curr_assets_over_tot_sales(all_f), fs.get_capital_over_tot_sales(all_f), ...
       fs.get_acc_pay_over_tot_sales(all_f), fs.get_acc_rec_over_tot_sales(all_f), ...
       fs.get_invent_over_tot_sales(all_f), fs.get_cash_over_tot_sales(all_f), fs.get_oper_profit(all_f), ...
       fs.get_return_avg_equity(all_f), fs.get_tot_liab(all_f), fs.get_oper_funds(all_f), fs.get_oper_margin(all_f), ...
       fs.get_interest_coverage(all_f), fs.get_fix_asset_turnover(all_f), fs.get_tot_asset_turnover(all_f), ...
       fs.get_cash_ratio(all_f), fs.get_asset_value(all_f), fs.get_earnings_per_share(all_f), ...
       fs.get_ebitda_per_share(all_f), fs.get_divid_per_share(all_f), fs.get_tot_assets_over_nav(all_f), ...
       fs.get_stock_adj_close_price(stock_data.(dummy_var).AdjClose), fs.get_log_revenue_return(market_data.AdjClose)}; 
 
   % Let's do part 2 of the features
   part2_features = containers.Map(new_features_tags, new_features_data);
   
   % Let's merge the two hash tables -- not very readable
   merged_features = [part1_features; part2_features];
  
   % Fill out the full_features structure
   full_features.(dummy_var) = merged_features;
   features_data = full_features;
end