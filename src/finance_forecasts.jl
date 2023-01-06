# Libraries
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, MessyTimeSeries, MessyTimeSeriesOptim, DecisionTree, StableRNGs, Statistics;
include("./macro_functions.jl");
include("./finance_functions.jl");

#=
Load arguments passed through the command line
=#

# Equity index id
equity_index_id = parse(Int64, ARGS[1]);

# Regression model
regression_model = parse(Int64, ARGS[2])

# EP or not
compute_ep_cycle = parse(Bool, ARGS[3]);

# Use factors as well as autoregressive data on the target
include_factor_augmentation = parse(Bool, ARGS[4]);

# Use factors transformations as well as the level
include_factor_transformations = parse(Bool, ARGS[5]);

# Output folder
log_folder_path = ARGS[6];

# Fixed number of trees per ensemble
n_trees = 1000;

#=
Setup logger
=#

io = open("$(log_folder_path)/$(regression_model)/status_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).txt", "w+");
global_logger(ConsoleLogger(io));

#=
First log entries
=#

@info("------------------------------------------------------------")
@info("equity_index_id: $(equity_index_id)");
@info("regression_model: $(regression_model)")
@info("compute_ep_cycle: $(compute_ep_cycle)");
@info("include_factor_augmentation: $(include_factor_augmentation)");
@info("include_factor_transformations: $(include_factor_transformations)");
@info("log_folder_path: $(log_folder_path)");
@info("n_trees: $(n_trees)");
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

# Set seed
rng = StableRNG(1);

# First data vintage
first_data_vintage = data_vintages[1]; # default: data up to 2005-01-31

# Selection sample dimensions
estimation_sample_length = fld(size(first_data_vintage, 1), 2);

# Get training and validation samples
_, _, estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors = get_macro_data_partitions(first_data_vintage[1:end-1, :], equity_index[1:size(first_data_vintage, 1)], estimation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, include_factor_transformations, compute_ep_cycle, n_cycles, coordinates_params_rescaling);

# Predictors dimensions in estimation sample
n_predictors, estimation_sample_adj_length = size(estimation_samples_predictors);

@info("n_predictors: $(n_predictors)");
@info("estimation_sample_adj_length: $(estimation_sample_adj_length)");
@info("------------------------------------------------------------")
@info("Construct hyperparameters grid");
flush(io);

# Compute validation error
grid_partial_sampling  = [1.0];                                                           # the number of samples to draw from X to train each base estimator
range_min_samples_leaf = range(0.01, stop=0.50, length=25);
grid_min_samples_leaf  = collect(range_min_samples_leaf .* estimation_sample_adj_length); # the minimum number of samples required to be at a leaf node
grid_min_samples_leaf  = unique(ceil.(grid_min_samples_leaf)) |> Vector{Int64};           # round to nearest integers

# Bagging
if regression_model == 1
    grid_n_subfeatures = [n_predictors];                                                  # the default of `n_predictors` is equivalent to bagged trees

# Random forest
elseif regression_model == 2
    grid_n_subfeatures = collect(range(0.05, stop=0.95, length=25) .* n_predictors);      # more randomness can be achieved by setting smaller values than `n_predictors`
    grid_n_subfeatures = unique(ceil.(grid_n_subfeatures)) |> Vector{Int64};              # round to nearest integers

else
    error("Unsupported `regression_model!`")
end

grid_hyperparameters = Vector{Dict}();
for partial_sampling in grid_partial_sampling
    for n_subfeatures in grid_n_subfeatures
        for min_samples_leaf in grid_min_samples_leaf
            push!(grid_hyperparameters, Dict(:rng => rng, :n_trees => n_trees, :partial_sampling => partial_sampling, :n_subfeatures => n_subfeatures, :min_samples_leaf => min_samples_leaf));
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
    _, _, validation_errors[i] = estimate_and_validate_dt_model(estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors, RandomForestRegressor, model_settings);
end

# Optimal hyperparameter
optimal_rf_settings = grid_hyperparameters[argmin(validation_errors)];

# Convert optimal `min_samples_leaf` in percentage terms
optimal_rf_settings[:min_samples_leaf] = range_min_samples_leaf[findfirst(grid_min_samples_leaf .== optimal_rf_settings[:min_samples_leaf])];

# Update logs
@info("Optimal hyperparameters: $(optimal_rf_settings)");
flush(io);

#=
Out-of-sample forecasts

The out-of-sample exercise stores the one-step ahead squared error for the target equity index. 
This operation produces forecasts referring to every month from 2005-02-28 to 2021-01-31.
=#

#=
Memory pre-allocation for output
Note that I am using Vectors{...} for the first three variables to avoid scope errors
=#

sspace = Vector{KalmanSettings}(undef, 1);
std_diff_data = Vector{FloatMatrix}(undef, 1);
optimal_rf_instance = Vector{RandomForestRegressor}(undef, 1);
outturn_array = zeros(length(data_vintages));
forecast_array = zeros(length(data_vintages));

@info("------------------------------------------------------------")
for v in axes(forecast_array, 1)
    @info ("OOS iteration $(v) out of $(length(forecast_array))");
    flush(io);

    # Select current vintage
    current_data_vintage = data_vintages[v];
    current_data_vintage_length = size(current_data_vintage, 1);

    if v==1#year(current_data_vintage[end, :reference_dates]) != year(current_data_vintage[end-1, :reference_dates])

        # Recover hyperparameters
        current_optimal_rf_settings = copy(optimal_rf_settings);
        current_optimal_rf_settings[:min_samples_leaf] = ceil(optimal_rf_settings[:min_samples_leaf]*(current_data_vintage_length-1)) |> Int64; # current_data_vintage_length-1 is fine
        
        # Re-estimate random forest
        sspace[1], std_diff_data[1], selection_samples_target, selection_samples_predictors, _ = get_macro_data_partitions(current_data_vintage[1:end-1, :], equity_index[1:size(current_data_vintage, 1)], current_data_vintage_length - 1, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, include_factor_transformations, compute_ep_cycle, n_cycles, coordinates_params_rescaling);
        optimal_rf_instance[1] = estimate_dt_model(selection_samples_target, selection_samples_predictors, RandomForestRegressor, current_optimal_rf_settings);        
    end

    #=
    Compute predictor matrix and get outturn for the target
    Note that the following function is using an estimated `sspace` and `std_diff_data` setting `t0=current_data_vintage_length-1` does not have an impact on the conditioning set -> it is a convenient trick to obtain the right arrays to forecast in oos
    =#
    
    _, _, _, _, current_target, current_predictors = get_macro_data_partitions(current_data_vintage, equity_index[1:current_data_vintage_length + 1], current_data_vintage_length-1, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, include_factor_transformations, compute_ep_cycle, n_cycles, coordinates_params_rescaling, sspace[1], std_diff_data[1]);

    # Store new forecast
    forecast_array[v] = DecisionTree.predict(optimal_rf_instance[1], permutedims(current_predictors))[end]; # the function returns a 1-dimensional vector

    # Store current outturn
    outturn_array[v] = current_target[end]; # current_target is a 1-dimensional vector
end

@info("------------------------------------------------------------")
@info("optimal_rf_instance: $(optimal_rf_instance[1])");

#=
Store output in JLD
=#

@info("------------------------------------------------------------")
@info("Saving output to JLD");
flush(io);

# Reference periods for the target (i.e., equity index)
equity_index_ref = [Dates.lastdayofmonth(Dates.firstdayofmonth(data_vintages[v][end,1])+Month(1)) for v in axes(data_vintages, 1)];

# Distance from reference period
distance_from_reference_month = Dates.value.(equity_index_ref-release_dates);

# Store output
save("$(log_folder_path)/$(regression_model)/output_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).jld", 
    Dict("equity_index_id" => equity_index_id,
         "regression_model" => regression_model,
         "compute_ep_cycle" => compute_ep_cycle,
         "include_factor_augmentation" => include_factor_augmentation,
         "include_factor_transformations" => include_factor_transformations,
         "sspace" => sspace[1],
         "std_diff_data" => std_diff_data[1],
         "grid_hyperparameters" => grid_hyperparameters, 
         "validation_errors" => validation_errors, 
         "optimal_rf_settings" => optimal_rf_settings, 
         "optimal_rf_instance" => optimal_rf_instance[1], 
         "outturn_array" => outturn_array, 
         "forecast_array" => forecast_array, 
         "release_dates" => release_dates,
         "equity_index_ref" => equity_index_ref,
         "distance_from_reference_month" => distance_from_reference_month));

@info("------------------------------------------------------------")
@info("Done!");
@info("------------------------------------------------------------")
close(io);
