# Libraries
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, MessyTimeSeries, MessyTimeSeriesOptim, ScikitLearn, Statistics;
@sk_import ensemble: RandomForestRegressor;
using Infiltrator; # TEMP
include("./macro_functions.jl");
include("./finance_functions.jl");

#=
Load arguments passed through the command line
=#

compute_ep_cycle=true; equity_index_id=1; include_factor_augmentation=true; use_refined_BC=true; regression_model=1; log_folder_path="./BC_and_EP_output";

# Fixed number of trees per ensemble
n_estimators = 1000;

#=
Setup logger
=#

io = open("$(log_folder_path)/$(regression_model)/status_equity_index_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(compute_ep_cycle).txt", "w+");
global_logger(ConsoleLogger(io));

#=
First log entries
=#

@info("------------------------------------------------------------")
@info("compute_ep_cycle: $(compute_ep_cycle)");
@info("equity_index_id: $(equity_index_id)");
@info("include_factor_augmentation: $(include_factor_augmentation)");
@info("use_refined_BC: $(use_refined_BC)");
@info("regression_model: $(regression_model)")
@info("log_folder_path: $(log_folder_path)");
@info("n_estimators: $(n_estimators)");
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

# `df_equity_index` must have one time period more than `data_vintages[end]`
if size(df_equity_index, 1) != 1+size(data_vintages[end], 1)
    error("`df_equity_index` must have one time period more than `data_vintages[end]`!");
end

#=
DFM settings
=#

# Series classification (WARNING: manual input required)
n_series = size(data_vintages[end], 2)-1; # `-1` excludes the reference dates column
n_cycles = 7;                             # shortcut to denote the variable that identifies the energy cycle (i.e., `WTISPLC` after having included it prior to `PCEPI`)
n_cons_prices = 2;                        # number of consumer price indices in the dataset

# Get settings
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices, false);

#=
Hyperparameter selection for aggregator

This operation is performed with the same data used for the macro selection. Indeed, the model is:
    - estimated on the first half of the selection sample (from 1984-01-31 to 1994-06-30) and
    - validated on the remaining observations (from 1994-07-31 to 2005-01-31).

The selection sample is the first vintage in `data_vintages`.
=#

@info("------------------------------------------------------------")
@info("Partition data into training and validation samples");
flush(io);

# First data vintage
first_data_vintage = data_vintages[1]; # default: data up to 2005-01-31

# Selection sample dimensions
estimation_sample_length = fld(size(first_data_vintage, 1), 2);
validation_sample_length = size(first_data_vintage, 1) - estimation_sample_length;

# Get training and validation samples
_, _, estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors = get_macro_data_partitions(first_data_vintage, equity_index[1:size(first_data_vintage, 1) + 1], estimation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, compute_ep_cycle, n_cycles, coordinates_params_rescaling);

@info("------------------------------------------------------------")
@info("Construct hyperparameters grid");
flush(io);

# Compute validation error
grid_max_samples      = [1.0];                                                  # the number of samples to draw from X to train each base estimator
grid_max_features     = [1.0];                                                  # the default of 1.0 is equivalent to bagged trees and more randomness can be achieved by setting smaller values
grid_min_samples_leaf = collect(range(5, stop=50, length=10)) |> Vector{Int64}; # the minimum number of samples required to be at a leaf node

grid_hyperparameters = Vector{NamedTuple}();
for max_samples in grid_max_samples
    for max_features in grid_max_features
        for min_samples_leaf in grid_min_samples_leaf
            push!(grid_hyperparameters, (random_state=1, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, min_samples_leaf=min_samples_leaf));
        end
    end
end

@info("------------------------------------------------------------")
@info("Select hyperparameters for tree ensemble");
flush(io);

# Initialise
validation_errors = zeros(length(grid_hyperparameters));

# Compute the validation mean squared error looping over each candidate hyperparameter
for (i, model_settings) in enumerate(grid_hyperparameters)
    _, _, validation_errors[i] = estimate_and_validate_skl_model(estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors, RandomForestRegressor, model_settings);
end

# Optimal hyperparameter
optimal_rf_settings = grid_hyperparameters[argmin(validation_errors)];
@info("Optimal min_samples_leaf=$(optimal_rf_settings)");
flush(io);

#=
Out-of-sample forecasts

The out-of-sample exercise stores the one-step ahead squared error for the target equity index. 
This operation produces forecasts referring to every month from 2005-02-28 to 2021-01-31.
=#

# Estimate on full selection sample
sspace, std_diff_data, selection_samples_target, selection_samples_predictors, _ = get_macro_data_partitions(first_data_vintage, equity_index[1:size(first_data_vintage, 1) + 1], estimation_sample_length+validation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, compute_ep_cycle, n_cycles, coordinates_params_rescaling);

@infiltrate
optimal_rf_instance = estimate_skl_model(selection_samples_target, selection_samples_predictors, RandomForestRegressor, optimal_rf_settings);

# The equity index value for 2005-01-31 is used in the estimation. This offset allows to start the next calculations from the next reference point and to be a truly out-of-sample exercise
offset_vintages = 4;

# Memory pre-allocation for output
outturn_array = zeros(length(data_vintages)-offset_vintages);
forecast_array = zeros(length(data_vintages)-offset_vintages);

@info("------------------------------------------------------------")
for v in axes(forecast_array, 1)
    @info ("OOS iteration $(v) out of $(length(forecast_array))");
    flush(io);

    # Select current vintage
    current_data_vintage = data_vintages[v+offset_vintages];
    current_data_vintage_length = size(current_data_vintage, 1);

    # Compute predictor matrix and get outturn for the target
    _, _, _, _, current_validation_samples_target, current_validation_samples_predictors = get_macro_data_partitions(current_data_vintage, equity_index[1:current_data_vintage_length + 1], current_data_vintage_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, compute_ep_cycle, n_cycles, coordinates_params_rescaling, sspace, std_diff_data);

    # Store new forecast
    @infiltrate
    forecast_array[v] = ScikitLearn.predict(optimal_rf_instance, permutedims(current_validation_samples_predictors[:, end])); # in ScikitLearn all input predictors matrices are vertical - i.e., of shape (n_sample, n_feature)

    # Store current outturn
    outturn_array[v] = current_validation_samples_target[end];
end

# STORE OUTPUT TO JLD
