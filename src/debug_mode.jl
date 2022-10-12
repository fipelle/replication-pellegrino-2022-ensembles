# Libraries
using Distributed;
@everywhere using MessyTimeSeriesOptim;
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, Statistics, MessyTimeSeries;
include("./macro_functions.jl");
using Infiltrator;
include("./finance_functions.jl");

#=
Load arguments passed through the command line
=#

compute_ep_cycle=false; equity_index_id=1; include_factor_augmentation=false; use_refined_BC=true; regression_model=1; log_folder_path="./BC_output";

# Fixed number of max_samples for artificial jackknife
max_samples = 1000;

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

# First data vintage
first_data_vintage = data_vintages[1]; # default: data up to 2005-01-31

# Selection sample dimensions
estimation_sample_length = fld(size(first_data_vintage, 1), 2);
validation_sample_length = size(first_data_vintage, 1) - estimation_sample_length;

@info("------------------------------------------------------------")
@info("Partition data into training, selection and validation samples");

# Get selection and validation samples
ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, training_samples_target, training_samples_predictors, selection_samples_target, selection_samples_predictors, validation_samples_target, validation_samples_predictors = get_macro_data_partitions(first_data_vintage, equity_index, estimation_sample_length, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC, coordinates_params_rescaling);
