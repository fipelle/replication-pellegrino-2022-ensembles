# Libraries
using Distributed;
@everywhere using MessyTimeSeriesOptim;
@everywhere include("./get_real_time_datasets.jl");
using CSV, FileIO, JLD;
using Random, LinearAlgebra, MessyTimeSeries;
include("./macro_functions.jl");

#=
Load arguments passed through the command line
=#

# EP or not
compute_ep_cycle = parse(Bool, ARGS[1]);

# Error estimator id
err_type = parse(Int64, ARGS[2]);

# Output folder
log_folder_path = ARGS[3];

#=
Generate data vintages
=#

# Macroeconomic indicators
tickers = ["TCU", "INDPRO", "PCE", "PAYEMS", "EMRATIO", "UNRATE", "PCEPI", "CPIAUCNS", "CPILFENS"];
tickers_to_deflate = ["PCE"];
fred_options = Dict(:realtime_start => "2005-01-31", :realtime_end => "2020-12-31", :observation_start => "1983-01-01"); # 1983 is one year prior to the actual observation_start

# Series classification (WARNING: manual input required)
n_cycles = 7;      # shortcut to denote the variable that identifies the energy cycle (i.e., `WTISPLC` after having included it prior to `PCEPI`)
n_cons_prices = 2; # number of consumer price indices in the dataset

# Download options for ALFRED
frequencies = ["m" for i=1:length(tickers)];
rm_base_year_effect = zeros(length(tickers));
rm_base_year_effect[findfirst(tickers .== "INDPRO")] = 1; # PCEPI adjusted below with an ad-hoc mechanism, when deflating PCE - no other series requires similar adjustments
rm_base_year_effect = rm_base_year_effect .== 1;

# Download data from ALFRED
df = get_fred_vintages(tickers, frequencies, fred_options, rm_base_year_effect);

# Energy commodities
tickers_energy = ["WTISPLC"];
fred_options_energy = Dict(:observation_start => "1983-01-01", :observation_end => "2020-12-31", :aggregation_method => "avg"); # here, :observation_end is aligned with the macro :realtime_end since data is released at the end of each reference month
df_energy = get_financial_vintages(tickers_energy, fred_options_energy, Date(fred_options[:realtime_start]));

# Join `df` and `df_energy`
insert!(tickers, length(tickers)-n_cons_prices+1, tickers_energy...); # place before inflation indices (excluding `PCEPI` given that is later removed from the sample)
df = outerjoin(df, df_energy, on=[:reference_dates, :release_dates]);
df = df[!, vcat(:release_dates, :reference_dates, Symbol.(tickers))];
sort!(df, :release_dates);

# Build data vintages
data_vintages, release_dates = get_vintages_array(df, "m");

# Remove `:PCEPI` from the data vintages, after having used it for deflating the series indicated in tickers_to_deflate
tickers = deflate_vintages_array!(data_vintages, release_dates, tickers, tickers_to_deflate);

#=
Setup validation problem
=#

# Selection sample
selection_sample = data_vintages[1][!,2:end] |> JMatrix{Float64};
selection_sample = permutedims(selection_sample);

# Number of series
n_series = length(tickers);

# Validation inputs: common for all err_types
gamma_bounds = ([12, 12], [0.01, 2.50], [0.0, 1.0], [1.0, 1.2]);
grid_prerun = HyperGrid(gamma_bounds..., 1);
grid = HyperGrid(gamma_bounds..., 1000);
weights = zeros(n_series);
weights[findfirst(tickers .== "RPCE")] = 1.0;

# Validation inputs: common for all oos err_types
t0 = 126;
n, T = size(selection_sample);

# Subsample
if err_type < 3 # iis and oos
    subsample = 1.0; # not used in validation for these cases

elseif err_type == 3 # block jackknife
    subsample = parse(Float64, ARGS[4]);

elseif err_type == 4 # artificial jackknife
    d = optimal_d(n, T);
    subsample = d/(n*T);
end

# Validation inputs: specific for the artificial jackknife (not used in validation for the other cases)
max_samples = 1000;

# Validation settings
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices);
vs_prerun = ValidationSettings(err_type, selection_sample, false, DFMSettings, model_args=model_args, model_kwargs=model_kwargs, coordinates_params_rescaling=coordinates_params_rescaling, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path, verb=false);
vs = ValidationSettings(err_type, selection_sample, false, DFMSettings, model_args=model_args, model_kwargs=model_kwargs, coordinates_params_rescaling=coordinates_params_rescaling, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path);

# Test run to speed up compilation
_ = select_hyperparameters(vs_prerun, grid_prerun);

# Compute and print ETA
ETA = @elapsed select_hyperparameters(vs_prerun, grid_prerun);
ETA *= grid.draws;
ETA /= 3600;
println("ETA: $(round(ETA, digits=2)) hours");

# Actual run
candidates, errors = select_hyperparameters(vs, grid);

# Save output to JLD
save("$(log_folder_path)/err_type_$(err_type).jld", Dict("df" => df, "data_vintages" => data_vintages, "release_dates" => release_dates, "selection_sample" => selection_sample, "vs" => vs, "grid" => grid, "candidates" => candidates, "errors" => errors));