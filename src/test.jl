# Libraries
using Distributed;
@everywhere using MessyTimeSeriesOptim;
@everywhere include("./get_real_time_datasets.jl");
using CSV, FileIO, JLD;
using Random, LinearAlgebra, MessyTimeSeries;
include("./macro_functions.jl");

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

# Remove `:PCEPI` from the data vintages, after having used it for deflating the series indicated in `tickers_to_deflate`
transform_vintages_array!(data_vintages, release_dates, tickers, tickers_to_deflate, n_cons_prices);

#=
Setup single tc run
=#

# Selection sample
full_sample = data_vintages[end][!,2:end] |> JMatrix{Float64};
full_sample = permutedims(full_sample);