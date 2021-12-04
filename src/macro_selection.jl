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
fred_options = Dict(:realtime_start => "2005-01-31", :realtime_end => "2020-12-31", :observation_start => "1983-01-01"); # 1983 is one year prior to the actual observation_start

# Series classification (WARNING: manual input required)
n_cycles = 7; # shortcut to denote the variable that identifies the energy cycle
n_cons_prices = 2;

# Download data from ALFRED
frequencies = ["m" for i=1:length(tickers)];
rm_base_year_effect = zeros(length(tickers));
rm_base_year_effect[findfirst(tickers .== "INDPRO")] = 1; # PCEPI adjusted below with an ad-hoc mechanism, when deflating PCE
rm_base_year_effect = rm_base_year_effect .== 1;
df = get_fred_vintages(tickers, frequencies, fred_options, rm_base_year_effect);

# Energy commodities
tickers_energy = ["WTISPLC"];
fred_options_energy = Dict(:observation_start => "1983-01-01", :observation_end => "2020-12-31", :aggregation_method => "avg"); # here, :observation_end is aligned with the macro :realtime_end since data is released at the end of each reference month
df_energy = get_financial_vintages(tickers_energy, fred_options_energy, Date(fred_options[:realtime_start]));

# Join `df` and `df_energy`
df = outerjoin(df, df_energy, on=[:reference_dates, :release_dates]);
df = df[!, vcat(:release_dates, :reference_dates, Symbol.(tickers[1:end-n_cons_prices]), Symbol.(tickers_energy), Symbol.(tickers[end-n_cons_prices+1:end]))];
sort!(df, :release_dates);

# No. of series
tickers = names(df)[3:end];
n_series = length(tickers);

# Build data vintages
data_vintages, release_dates = get_vintages_array(df, "m");

# Manual data transformations

# Compute reference value to the first obs. of the last PCEPI vintage
first_obs_last_vintage_PCEPI = data_vintages[end][1, :PCEPI]; # this is used for rebasing PCEPI

# Final columns to keep in each vintage
new_vintage_cols = vcat(:reference_dates, Symbol.(tickers[tickers .!= "PCEPI"]));

# Update `tickers` and `n_series` to account for the removal of PCEPI
tickers = tickers[tickers .!= "PCEPI"];
n_series -= 1;

# Loop over every data vintage
for i=1:length(data_vintages)

    # Pointer
    vintage = data_vintages[i];

    # Rescale PCE deflator
    if ismissing(vintage[1, :PCEPI])
        error("Base year effect cannot be removed from PCEPI");
    end
    vintage[!, :PCEPI] ./= vintage[1, :PCEPI];
    vintage[!, :PCEPI] .*= first_obs_last_vintage_PCEPI;

    # Custom real variables
    for ticker in [:PCE]
        vintage[!, ticker] ./= vintage[!, :PCEPI];
        vintage[!, ticker] .*= 100;
    end

    # Remove PCEPI
    vintage = vintage[:, new_vintage_cols];

    # Compute YoY%
    for ticker in Symbol.(tickers[n_cycles:end])
        vintage[13:end, ticker] = 100*(vintage[13:end, ticker] ./ vintage[1:end-12, ticker] .- 1);
    end

    # Overwrite vintage by removing the first 12 months and PCEPI
    vintage = vintage[13:end, :];

    # Rename PCE to RPCE
    rename!(vintage, (:PCE => :RPCE));

    # Update data_vintages
    data_vintages[i] = vintage;
end

# Update tickers accordingly
tickers[findfirst(tickers .== "PCE")] = "RPCE";

# Remove problematic ALFRED data vintages for PCEPI
ind_problematic_release = findfirst(release_dates .== Date("2009-08-04")); # PCEPI is incorrectly recorded at that date in ALFRED
deleteat!(release_dates, ind_problematic_release);
deleteat!(data_vintages, ind_problematic_release);

#=
Setup validation problem
=#

# Selection sample
selection_sample = data_vintages[1][!,2:end] |> JMatrix{Float64};
selection_sample = permutedims(selection_sample);

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