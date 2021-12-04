# Libraries
using DataFrames, Dates, FileIO, JLD;
using Contour, DecisionTree, LinearAlgebra, Random, StableRNGs, MessyTimeSeries;
using PGFPlotsX, LaTeXStrings;
include("./macro_functions.jl");
using MessyTimeSeriesOptim;

# Manual input
compute_ep_cycle = false;
pre_covid = true;
n_series = 9;
n_cycles = 7; # shortcut to denote the variable that identifies the energy cycle
n_cons_prices = 2;

# Data
if compute_ep_cycle == false
    macro_output = jldopen("./BC_output/err_type_4.jld");
else
    macro_output = jldopen("./BC_and_EP_output/err_type_4.jld");
end
data_vintages = read(macro_output["data_vintages"]);

data = data_vintages[1][!, 2:end] |> JMatrix{Float64};

if pre_covid
    full_data = data_vintages[end-58][!, 2:end] |> JMatrix{Float64}; # 2019-12-31
else
    full_data = data_vintages[end][!, 2:end] |> JMatrix{Float64}; # 2020-12-31
end

data = permutedims(data);
full_data = permutedims(full_data);

# Optimal hyperparameters
candidates = read(macro_output["candidates"]);
errors = read(macro_output["errors"]);
optimal_hyperparams = candidates[:, argmin(errors)];

# Estimate DFM
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices);
estim, std_diff_data = get_tc_structure(data, optimal_hyperparams, model_args, model_kwargs);

# Run Kalman routines
sspace = ecm(estim, output_sspace_data = full_data ./ std_diff_data);
status = kfilter_full_sample(sspace);
X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status);
smoothed_states = hcat([X_sm[i] for i=1:length(X_sm)]...);

# Indices
trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params = model_args;
n_trends = length(drifts_selection);
non_stationary_skeleton = vcat([ifelse(drifts_selection[i], [1.0; 0.0; 2.0; 0.0], [1.0; 0.0]) for i=1:n_trends]...);
ind_trends = findall(non_stationary_skeleton .== 1);
ind_drifts = findall(non_stationary_skeleton .== 2);
last_trend = findlast(non_stationary_skeleton .== 1)+1;
ind_idio_cycles = collect(last_trend+1:2:last_trend+n_series*2);
ind_bc_cycle = last_trend+n_series*2+1;
ind_ep_cycle = ind_bc_cycle+estim.lags+1;

# States of interest
smoothed_trends = std_diff_data .* (sspace.B[:, ind_trends]*smoothed_states[ind_trends, :]);
smoothed_drifts = std_diff_data[findall(drifts_selection)] .* smoothed_states[ind_drifts, :];
smoothed_idio_cycles = std_diff_data .* smoothed_states[ind_idio_cycles, :];
smoothed_bc_cycle = std_diff_data .* (sspace.B[:, ind_bc_cycle:ind_bc_cycle+estim.lags-1]*smoothed_states[ind_bc_cycle:ind_bc_cycle+estim.lags-1, :]);
smoothed_ep_cycle = std_diff_data .* (sspace.B[:, ind_ep_cycle:end-1]*smoothed_states[ind_ep_cycle:end-1, :]);

#=
Plotting stage
=#

if pre_covid
    ref_dates_fig = data_vintages[end-58][!,1];
else
    ref_dates_fig = data_vintages[end][!,1];
end

ref_dates_grid = ref_dates_fig[1]:Month(1):ref_dates_fig[end];

using Colors;
c1 = colorant"rgba(0, 48, 158, .75)";
c2 = colorant"rgba(255, 0, 0, .75)";
c3 = colorant"rgba(255, 190, 0, .75)";

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
    { group_style = {group_size="3 by 3", vertical_sep="50pt"},
      no_markers,
      legend_pos="north west",
      height="150pt",
      width="225pt"
    },
    axs...),
    raw"\node at ($(group c2r3) + (0,-3.25cm)$) {\ref{grouplegend}};",
);

pgfsave("./img/trend_cycle_$(compute_ep_cycle)_$(pre_covid).pdf", fig);