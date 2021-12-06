# Libraries
using Distributed;
@everywhere include("./TimeSeriesEnsembles.jl");
@everywhere using MessyTimeSeriesOptim;
@everywhere using Main.TimeSeriesEnsembles;
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, Statistics, MessyTimeSeries;
include("./macro_functions.jl");
include("./finance_functions.jl");

#=
Load arguments passed through the command line
=#

# EP or not
compute_ep_cycle = parse(Bool, ARGS[1]);

# Equity index id
equity_index_id = parse(Int64, ARGS[2]);

# Use lags of equity index as predictors
include_factor_augmentation = parse(Bool, ARGS[3]);

# Use BC breakdown, rather than the raw estimate
use_refined_BC = parse(Bool, ARGS[4]);

# Subsample size (percentage of obs. taken/removed)
subsample = parse(Float64, ARGS[5]); # if NaN and AJK, the code computes d̂

# Subsampling function
subsampling_function_id = parse(Int64, ARGS[6]);

if subsampling_function_id == 0
    subsampling_function = nothing;
    subsampling_mnemonic = "pair_bootstrap";
    tree_subsampling_method = 1;

elseif subsampling_function_id == 1
    subsampling_function = moving_block_bootstrap;
    subsampling_mnemonic = "block_bootstrap";
    tree_subsampling_method = 2;
    @warn("Untested subsampling_function_id!")

elseif subsampling_function_id == 2
    subsampling_function = block_jackknife;
    subsampling_mnemonic = "block_jackknife";
    tree_subsampling_method = 2;
    @warn("Untested subsampling_function_id!")

elseif subsampling_function_id == 3
    subsampling_function = artificial_jackknife;
    subsampling_mnemonic = "artificial_jackknife";
    tree_subsampling_method = 2;
end

# Output folder
log_folder_path = ARGS[7];

# Fixed number of max_samples for artificial jackknife
max_samples = 1000;

#=
Setup logger
=#

subsample_str = replace("$(ifelse(isnan(subsample), "default", subsample))", "."=>"_");

io = open("$(log_folder_path)/$(subsampling_mnemonic)/status_equity_index_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str).txt", "w+");
global_logger(ConsoleLogger(io));

#=
First log entries
=#

@info("------------------------------------------------------------")
@info("compute_ep_cycle: $(compute_ep_cycle)");
@info("equity_index_id: $(equity_index_id)");
@info("include_factor_augmentation: $(include_factor_augmentation)");
@info("use_refined_BC: $(use_refined_BC)");
@info("subsample: $(subsample)");
@info("subsampling_function_id: $(subsampling_function_id)");
@info("subsampling_mnemonic: $(subsampling_mnemonic)");
@info("tree_subsampling_method: $(tree_subsampling_method)")
@info("log_folder_path: $(log_folder_path)");
@info("max_samples: $(max_samples)");
flush(io);

#=
Load key macro output and selected equity index
=#

# Macro
macro_output = jldopen("$(log_folder_path)/err_type_4.jld");
data_vintages = read(macro_output["data_vintages"]);
release_dates = read(macro_output["release_dates"]);
candidates = read(macro_output["candidates"]);
errors = read(macro_output["errors"]);
optimal_hyperparams = candidates[:, argmin(errors)];

# Equity index
df_equity_index = CSV.read("./data/wilshire_selection.csv", DataFrame);
equity_index = df_equity_index[:, 1+equity_index_id];

#=
DFM settings
=#

# Series classification (WARNING: manual input required)
n_series = 9;
n_cycles = 7; # shortcut to denote the variable that identifies the energy cycle
n_cons_prices = 2;

# Get settings
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices);

#=
Hyperparameter selection for aggregator

This operation is performed with the same data used for the macro selection. Indeed, the model is:
    - estimated on the first half of the selection sample (from 1984-01-31 to 1994-06-30) and
    - validated on the remaining observations (from 1994-07-31 to 2005-01-31).

The selection sample is the first vintage in `data_vintages`.
=#

# First data vintage
first_data_vintage = data_vintages[1]; # default: data up to 2005-01-31

# Selection sample dimensions
estimation_sample_length = fld(size(first_data_vintage,1), 2);
validation_sample_length = size(first_data_vintage, 1) - estimation_sample_length;

@info("------------------------------------------------------------")
@info("Generate subsamples");

# Pair bootstrap samples - post ssm
if tree_subsampling_method == 1
    ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, training_samples_target, training_samples_predictors, selection_samples_target, selection_samples_predictors, validation_samples_target, validation_samples_predictors = get_selection_samples_bootstrap(first_data_vintage, equity_index, estimation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, coordinates_params_rescaling);

# Custom subsampling methods - pre ssm (i.e., block bootstrap, block jackknife, artificial delete-d jackknife)
elseif tree_subsampling_method == 2
    ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, training_samples_target, training_samples_predictors, selection_samples_target, selection_samples_predictors, validation_samples_target, validation_samples_predictors = get_selection_samples_custom(io, subsampling_function, subsample, first_data_vintage, equity_index, estimation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, coordinates_params_rescaling);
end

# Compute validation error
grid_min_samples_leaf = collect(range(5, stop=50, length=10)) |> Vector{Int64};
errors_validation = zeros(length(grid_min_samples_leaf));

@info("------------------------------------------------------------")
@info("Select hyperparameters for aggregator");
flush(io);

# Loop over the grid of candidate min_samples_leaf hyperparameters
for (i, min_samples_leaf) in enumerate(grid_min_samples_leaf)

    @info("Candidate hyperparameter = $(min_samples_leaf)");
    flush(io);

    # Estimate aggregator with the current parametrisation
    trees = estimate_tree_aggregator(training_samples_target, training_samples_predictors, min_samples_leaf=min_samples_leaf, n_bootstrap_samples=max_samples);
    
    # Loop over the validation sample periods
    for t=1:validation_sample_length

        # Forecast
        fc = forecast_tree_aggregator(validation_samples_predictors[t], trees);

        # Compute validation squared error
        errors_validation[i] += (validation_samples_target[t] - fc)^2
    end

    # Compute the validation mean squared error
    errors_validation[i] /= validation_sample_length;
end

# Optimal hyperparameter
optimal_min_samples_leaf = grid_min_samples_leaf[argmin(errors_validation)];
@info("Optimal min_samples_leaf=$(optimal_min_samples_leaf)");

#=
Out-of-sample forecasts

The out-of-sample exercise stores the one-step ahead squared error for the target equity index. 
This operation produces forecasts referring to every month from 2005-02-28 to 2021-01-31.
=#

offset_vintages = 4; # the equity index value for 2005-01-31 is used in the estimation. This offset allows to start the next calculations from the next reference point and to be a truly out-of-sample exercise.
trees = estimate_tree_aggregator(selection_samples_target, selection_samples_predictors, min_samples_leaf=optimal_min_samples_leaf, n_bootstrap_samples=max_samples);
outturn_array = zeros(length(data_vintages)-offset_vintages);
forecast_array = zeros(length(data_vintages)-offset_vintages);

@info("------------------------------------------------------------")
for v in axes(forecast_array, 1)
    @info ("OOS iteration $(v) out of $(length(forecast_array))");
    flush(io);

    # Retrieve outturn and construct up-to-date predictor matrix
    outturn, predictors = get_oos_samples(ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, data_vintages[v+offset_vintages], equity_index, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);

    # Forecast
    fc_outturn = forecast_tree_aggregator(predictors, trees);

    # Store outturn and forecast
    outturn_array[v] = outturn;
    forecast_array[v] = fc_outturn;
end

#=
Store output in JLD
=#

@info("------------------------------------------------------------")
@info("Saving output to JLD");
flush(io);

# Reference periods
release_dates_oos = release_dates[1+offset_vintages:end];
equity_index_ref = [Dates.lastdayofmonth(Dates.firstdayofmonth(data_vintages[v][end,1])+Month(1)) for v in axes(data_vintages,1)];
equity_index_ref_oos = equity_index_ref[1+offset_vintages:end];
distance_from_reference_month = Dates.value.(equity_index_ref_oos-release_dates_oos);

# Store output
save("$(log_folder_path)/$(subsampling_mnemonic)/equity_index_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str).jld", 
    Dict("equity_index_id" => equity_index_id,
         "subsampling_mnemonic" => subsampling_mnemonic,
         "tree_subsampling_method" => tree_subsampling_method, 
         "include_factor_augmentation" => include_factor_augmentation,
         "use_refined_BC" => use_refined_BC,
         "ecm_kalman_settings" => ecm_kalman_settings,
         "ecm_std_diff_data" => ecm_std_diff_data,
         "business_cycle_position" => business_cycle_position,
         #"trees" => trees, # comment out if you want to store the trees as well -> very large jld output files
         "optimal_hyperparams" => optimal_hyperparams, 
         "errors_validation" => errors_validation, 
         "grid_min_samples_leaf" => grid_min_samples_leaf, 
         "optimal_min_samples_leaf" => optimal_min_samples_leaf, 
         "outturn_array" => outturn_array, 
         "forecast_array" => forecast_array, 
         "release_dates_oos" => release_dates_oos,
         "equity_index_ref_oos" => equity_index_ref_oos,
         "distance_from_reference_month" => distance_from_reference_month));

@info("------------------------------------------------------------")
@info("Done!");
@info("------------------------------------------------------------")
close(io);