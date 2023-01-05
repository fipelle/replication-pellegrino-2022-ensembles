# Libraries
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, MessyTimeSeries, MessyTimeSeriesOptim, DecisionTree, Statistics;
include("./macro_functions.jl");
include("./finance_functions.jl");

function retrieve_rmse(output_folder_name::String, equity_index_id::Int64, compute_ep_cycle::Bool, include_factor_augmentation::Bool, include_factor_transformations::Bool; exclude_2020::Bool=true)
    
    # Models
    bagging       = read(jldopen("$(output_folder_name)/1/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["forecast_array"]);
    random_forest = read(jldopen("$(output_folder_name)/2/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["forecast_array"]);

    # Outturn
    outturn = read(jldopen("$(output_folder_name)/1/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["outturn_array"])
    
    if exclude_2020
        bagging = bagging[1:end-55];
        random_forest = random_forest[1:end-55];
        outturn = outturn[1:end-55];
    end

    # Squared errors
    se_bagging = (bagging-outturn).^2;
    se_random_forest = (random_forest-outturn).^2;
    se_random_walk = (outturn).^2;

    # Return forecast error
    return mean([se_bagging se_random_forest], dims = 1) ./ mean(se_random_walk), se_bagging, se_random_forest;
end

# WARNING: manual input required
compute_ep_cycle = false;
output_folder_name = ifelse(compute_ep_cycle, "./BC_and_EP_output", "./BC_output");

# Retrieve forecast evaluation
baseline       = [retrieve_rmse(output_folder_name, i, compute_ep_cycle, false, false, exclude_2020=false) for i = 11:20];
factor_refined = [retrieve_rmse(output_folder_name, i, compute_ep_cycle, true, true, exclude_2020=false) for i = 11:20];

# Construct tables
baseline_table = vcat([baseline[i][1] for i in axes(baseline, 1)]...);
factor_refined_table = vcat([factor_refined[i][1] for i in axes(factor_refined, 1)]...);
