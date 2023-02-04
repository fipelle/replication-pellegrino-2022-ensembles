# Libraries
using CSV, DataFrames, Dates, FileIO, JLD, Latexify, Logging;
using LinearAlgebra, MessyTimeSeries, MessyTimeSeriesOptim, DecisionTree, Statistics;
include("./macro_functions.jl");
include("./finance_functions.jl");

function retrieve_rmse(output_folder_name::String, equity_index_id::Int64, compute_ep_cycle::Bool, include_factor_augmentation::Bool, include_factor_transformations::Bool; pre_covid::Bool=true)
    
    # Models
    bagging       = read(jldopen("$(output_folder_name)/1/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["forecast_array"]);
    random_forest = read(jldopen("$(output_folder_name)/2/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["forecast_array"]);

    # Outturn
    outturn = read(jldopen("$(output_folder_name)/1/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld")["outturn_array"])
    
    if pre_covid
        bagging = bagging[1:812];
        random_forest = random_forest[1:812];
        outturn = outturn[1:812];
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

# Initialise `merged_table`
merged_table_bagging = zeros(10, 4);
merged_table_rf = zeros(10, 4);

for (index, pre_covid) in enumerate([true; false])
    
    # Retrieve forecast evaluation
    baseline       = [retrieve_rmse(output_folder_name, i, compute_ep_cycle, false, false, pre_covid=pre_covid) for i = 11:20];
    factor_refined = [retrieve_rmse(output_folder_name, i, compute_ep_cycle, true, true, pre_covid=pre_covid) for i = 11:20];

    # Construct tables
    baseline_table = vcat([baseline[i][1] for i in axes(baseline, 1)]...);
    factor_refined_table = vcat([factor_refined[i][1] for i in axes(factor_refined, 1)]...);

    # Update `merged_table`
    merged_table_bagging[:, (1:2) .+ (index==2)*2] = [baseline_table[:,1] factor_refined_table[:,1]];
    merged_table_rf[:, (1:2) .+ (index==2)*2] = [baseline_table[:,2] factor_refined_table[:,2]];
end

println(latexify(round.(merged_table_bagging, digits=3)))