using Dates, DataFrames, DecisionTree, StableRNGs, Statistics, JLD, TSAnalysis;

function retrieve_rel_errors(output_folder_name::String, index_id::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)
    block_bootstrap = read(jldopen("./$(output_folder_name)/block_bootstrap/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_0_8.jld")["errors_oos"])
    block_jackknife = read(jldopen("./$(output_folder_name)/block_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_0_2.jld")["errors_oos"])
    pair_bootstrap = read(jldopen("./$(output_folder_name)/pair_bootstrap/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_default.jld")["errors_oos"])
    artificial_jackknife_opt = read(jldopen("./$(output_folder_name)/artificial_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_default.jld")["errors_oos"])
    artificial_jackknife_0_5 = read(jldopen("./$(output_folder_name)/artificial_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_0_5.jld")["errors_oos"])
    random_walk_benchmark = read(jldopen("./$(output_folder_name)/artificial_jackknife/equity_index_$(index_id)_$(include_factor_augmentation)_$(use_refined_BC)_0_5.jld")["errors_oos_rw"])

    return mean([block_bootstrap block_jackknife pair_bootstrap artificial_jackknife_opt artificial_jackknife_0_5][1:end-24, :], dims = 1) ./ mean(random_walk_benchmark[1:end-24])
end

output_folder_name = "BC_output";

baseline_table = vcat([retrieve_rel_errors(output_folder_name, i, false, false) for i = 1:10]...);
intermediate_factor_augmented_table = vcat([retrieve_rel_errors(output_folder_name, i, true, false) for i = 1:10]...);
factor_augmented_table = vcat([retrieve_rel_errors(output_folder_name, i, true, true) for i = 1:10]...);
