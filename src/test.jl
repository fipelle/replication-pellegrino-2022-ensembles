# Libraries
using Distributed;
include("../../../MessyTimeSeriesOptim.jl/src/MessyTimeSeriesOptim.jl");
using Main.MessyTimeSeriesOptim;
@everywhere include("./get_real_time_datasets.jl");
using CSV, FileIO, JLD;
using Random, LinearAlgebra, MessyTimeSeries;
include("./macro_functions.jl");

#=
Generate data vintages
=#

# Macroeconomic indicators
tickers = ["PI", "PCTR", "PCE", "INDPRO", "PAYEMS", "EMRATIO", "UNRATE", "PCEPI", "CPIAUCNS", "CPILFENS"];
tickers_to_transform = [DataTransformations(:PI, :PCTR, :PIX, -)];
tickers_to_deflate = ["PIX", "PCE"];
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

# Remove `:PCEPI` from the data vintages, after having used it for deflating the series indicated in `tickers_to_deflate` and applied the transformations in `tickers_to_transform`
transform_vintages_array!(data_vintages, release_dates, tickers, tickers_to_transform, tickers_to_deflate, n_cons_prices);

#=
Setup single tc run
=#

# DFMSettings base options
λ, α, β = (1.25, 0.50, 1.25);
lags = 12;
compute_ep_cycle=true;
n_series = length(tickers);

# Setup run
iis_data = data_vintages[end-58][!, 2:end] |> JMatrix{Float64}; # up to 2019-12-31
iis_data = permutedims(iis_data);
optimal_hyperparams = [lags; λ; α; β];

# Estimate DFM
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices, false);
estim, std_diff_data = get_tc_structure(iis_data, optimal_hyperparams, model_args, model_kwargs, coordinates_params_rescaling);

# Run Kalman routines
sspace = ecm(estim, output_sspace_data = iis_data ./ std_diff_data);
status = kfilter_full_sample(sspace);
X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status);
smoothed_states = hcat([X_sm[i] for i=1:length(X_sm)]...);

#using Plots
#rm .* etc at the end for inflation
#vv=1; ind_trend=findall(sspace.B[vv,:] .!= 0.0); fig=plot(iis_data[vv,:]); plot!(fig, smoothed_states[ind_trend[1],:] .* std_diff_data[vv]); fig

units = ["Percent", "Index (2012=100)", "Bil. Chn. 2012", "Mil. of persons", "Percent", "Percent", "Percent", "Percent", "Percent"];
custom_rescaling = [1, 1, 1, 1000, 1, 1, 1, 1, 1];

# Indices
trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params = model_args;
n_trends = length(drifts_selection);
non_stationary_skeleton = vcat([ifelse(drifts_selection[i], [1.0; 0.0; 2.0; 0.0], [1.0; 0.0]) for i=1:n_trends]...);
ind_trends = findall(non_stationary_skeleton .== 1);
last_non_stationary = length(non_stationary_skeleton);
ind_idio_cycles = collect(last_non_stationary+1:2:last_non_stationary+n_series*2);
ind_bc_cycle = last_non_stationary+n_series*2+1;
ind_ep_cycle = ind_bc_cycle+estim.lags+1;

# States of interest
smoothed_trends = std_diff_data .* (sspace.B[:, ind_trends]*smoothed_states[ind_trends, :]);
smoothed_idio_cycles = std_diff_data .* smoothed_states[ind_idio_cycles, :];
smoothed_bc_cycle = std_diff_data .* (sspace.B[:, ind_bc_cycle:ind_bc_cycle+estim.lags-1]*smoothed_states[ind_bc_cycle:ind_bc_cycle+estim.lags-1, :]);
smoothed_ep_cycle = std_diff_data .* (sspace.B[:, ind_ep_cycle:end-1]*smoothed_states[ind_ep_cycle:end-1, :]);

# Custom rescaling
iis_data ./= custom_rescaling;
smoothed_trends ./= custom_rescaling;
smoothed_idio_cycles ./= custom_rescaling;
smoothed_bc_cycle ./= custom_rescaling;
smoothed_ep_cycle ./= custom_rescaling;

# Reference dates
ref_dates_fig = data_vintages[end-58][!,1];
ref_dates_grid = ref_dates_fig[1]:Month(1):ref_dates_fig[end];

using Colors;
using PGFPlotsX, LaTeXStrings;

c1 = colorant"rgba(0, 48, 158, .75)";
c2 = colorant"rgba(255, 0, 0, .75)";
c3 = colorant"rgba(255, 190, 0, .75)";

# PGFPlotsX options
push!(PGFPlotsX.CUSTOM_PREAMBLE, 
        raw"\usepgfplotslibrary{colorbrewer}",
        raw"\usepgfplotslibrary{colormaps}",
        raw"\usepgfplotslibrary{patchplots}",
        raw"\pgfplotsset
            {   
                tick label style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                label style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                legend style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                title style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
            }"
)

#=
--------------------------------------------------------------------------------------------------------------------------------
Historical decomposition
--------------------------------------------------------------------------------------------------------------------------------
=#

# Manual input
axs = Array{Any}(undef, 9);

for i=1:9
    legend_style_content = ifelse(i==1, raw"{column sep = 10pt, legend columns = -1, legend to name = grouplegend, draw=none,}", "");
    axs[i] = @pgf Axis(
        {   
            "bar width=0.1pt",
            date_coordinates_in = "x",
            xticklabel=raw"{\year}",
            title = names(data_vintages[1])[i+1],
            grid = "both",
            xmin=ref_dates_grid[1],
            xmax=ref_dates_grid[end],
            ylabel=units[i],
            "ylabel absolute",
            legend_style=legend_style_content,
        },

        Plot({ybar_stacked, color=c1, fill=c1}, Table([ref_dates_grid, smoothed_bc_cycle[i,:]])),
        ifelse(compute_ep_cycle, Plot({ybar_stacked, color=c2, fill=c2}, Table([ref_dates_grid, smoothed_ep_cycle[i,:]])), {}),
        Plot({ybar_stacked, color=c3, fill=c3}, Table([ref_dates_grid, smoothed_idio_cycles[i,:]])),
        {},
        Plot({no_marks, style={"densely dotted", "thick"}, color="black"}, Table([ref_dates_grid, smoothed_bc_cycle[i,:]+smoothed_ep_cycle[i,:]+smoothed_idio_cycles[i,:]])),
        ifelse(i==1, ifelse(compute_ep_cycle, Legend("Business cycle", "Energy price cycle", "Idiosyncratic cycle", "Total cycle"), Legend("Business cycle", "Idiosyncratic cycle", "Total cycle")), {}),
    );
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="3 by 3", vertical_sep="60pt", horizontal_sep="60pt"},
      no_markers,
      legend_pos="north west",
      height="150pt",
      width="225pt"
    },
    axs...),
    raw"\node at ($(group c2r3) + (0,-3.25cm)$) {\ref{grouplegend}};",
);

fig

#=
--------------------------------------------------------------------------------------------------------------------------------
Trends
--------------------------------------------------------------------------------------------------------------------------------
=#

#=
# Manual input
axs = Array{Any}(undef, 9);

for i=1:9
    legend_style_content = ifelse(i==1, raw"{column sep = 10pt, legend columns = -1, legend to name = grouplegend, draw=none,}", "");
    axs[i] = @pgf Axis(
        {   
            "bar width=0.1pt",
            date_coordinates_in = "x",
            xticklabel=raw"{\year}",
            title = names(data_vintages[1])[i+1],
            grid = "both",
            xmin=ref_dates_grid[1],
            xmax=ref_dates_grid[end],
            ylabel=units[i],
            "ylabel absolute",
            legend_style=legend_style_content,
        },

        Plot({no_marks, style={"thick"}, color="black"}, Table([ref_dates_grid, iis_data[i,:]])),
        Plot({no_marks, style={"thick"}, color="blue"}, Table([ref_dates_grid, smoothed_trends[i,:]])),
        ifelse(i==1, Legend("Data", "Trend"), {});
    );
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="3 by 3", vertical_sep="60pt", horizontal_sep="60pt"},
      no_markers,
      legend_pos="north west",
      height="150pt",
      width="225pt"
    },
    axs...),
    raw"\node at ($(group c2r3) + (0,-3.25cm)$) {\ref{grouplegend}};",
);
=#