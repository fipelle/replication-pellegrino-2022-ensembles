using Dates, DataFrames, DecisionTree, StableRNGs, Statistics, JLD, MessyTimeSeries;

function retrieve_rel_errors(output_folder_name::String, index_id::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool; exclude_2020::Bool=true)
    
    # Models
    pair_bootstrap = read(jldopen("./$(output_folder_name)/pair_bootstrap/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_default.jld")["forecast_array"])
    artificial_jackknife = read(jldopen("./$(output_folder_name)/artificial_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_default.jld")["forecast_array"])
    
    # Outturn
    outturn = read(jldopen("./$(output_folder_name)/artificial_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_default.jld")["outturn_array"])

    # Squared errors
    se_pair_bootstrap = (pair_bootstrap-outturn).^2;
    se_artificial_jackknife = (artificial_jackknife-outturn).^2;
    se_random_walk = (outturn).^2;

    # Return forecast error
    if exclude_2020
        return mean([se_pair_bootstrap se_artificial_jackknife][1:end-55, :], dims = 1) ./ mean(se_random_walk[1:end-55]), se_pair_bootstrap[1:end-55], se_artificial_jackknife[1:end-55];
    else
        return mean([se_pair_bootstrap se_artificial_jackknife], dims = 1) ./ mean(se_random_walk), se_pair_bootstrap, se_artificial_jackknife;
    end
end

output_folder_name = "BC_output";

baseline = [retrieve_rel_errors(output_folder_name, i, false, false, exclude_2020=false) for i = 11:20];
#factor_base = [retrieve_rel_errors(output_folder_name, i, true, false, exclude_2020=false) for i = 11:20];
factor_refined = [retrieve_rel_errors(output_folder_name, i, true, true, exclude_2020=false) for i = 11:20];

baseline_table = vcat([baseline[i][1] for i=1:length(baseline)]...);
#factor_base_table = vcat([factor_base[i][1] for i=1:length(factor_base)]...);
factor_refined_table = vcat([factor_refined[i][1] for i=1:length(factor_refined)]...);
