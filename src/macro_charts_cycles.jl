# Libraries
using DataFrames, Dates, FileIO, JLD;
using Contour, DecisionTree, LinearAlgebra, Random, StableRNGs, MessyTimeSeries;
using PGFPlotsX, LaTeXStrings;
include("./macro_functions.jl");
using MessyTimeSeriesOptim;

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

# Series classification (WARNING: manual input required)
pre_covid = true;
compute_ep_cycle = false;
n_cycles = 7;               # shortcut to denote the variable that identifies the energy cycle (i.e., `WTISPLC` after having included it prior to `PCEPI`)
n_cons_prices = 2;          # number of consumer price indices in the dataset

units = ["Percent", "Index (2012=100)", "Bil. Chn. 2012", "Mil. of persons", "Percent", "Percent", "Percent", "Percent", "Percent"];
custom_rescaling = [1, 1, 1, 1000, 1, 1, 1, 1, 1]; # original scaling for PAYEMS is thous. of persons

# Data
if compute_ep_cycle == false
    macro_output = jldopen("./BC_output/err_type_4.jld");
else
    macro_output = jldopen("./BC_and_EP_output/err_type_4.jld");
end
data_vintages = read(macro_output["data_vintages"]);

n_series = size(data_vintages[end],2) - 1; # series - ref dates

if pre_covid
    iis_data = data_vintages[812][!, 2:end] |> JMatrix{Float64}; # release date: 2020-02-28
else
    iis_data = data_vintages[end][!, 2:end] |> JMatrix{Float64}; # release date: 2020-12-31
end

iis_data = permutedims(iis_data);

# Optimal hyperparameters
candidates = read(macro_output["candidates"]);
errors = read(macro_output["errors"]);
optimal_hyperparams = candidates[:, argmin(errors)];

# Estimate DFM
model_args, model_kwargs, coordinates_params_rescaling = get_dfm_args(compute_ep_cycle, n_series, n_cycles, n_cons_prices, false);
estim, std_diff_data = get_tc_structure(iis_data, optimal_hyperparams, model_args, model_kwargs, coordinates_params_rescaling);

# Run Kalman routines
sspace = ecm(estim, output_sspace_data = iis_data ./ std_diff_data);
status = kfilter_full_sample(sspace);
X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status);
smoothed_states = hcat([X_sm[i] for i=1:length(X_sm)]...);

# Recover tc structure from model_args
trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params = model_args;

# Common cycles offset
common_cycles_offset = (1+estim.lags)*(1+compute_ep_cycle);
bc_coordinates = size(smoothed_states, 1)-common_cycles_offset+1 : size(smoothed_states, 1)-common_cycles_offset+12;
ep_coordinates = bc_coordinates .+ estim.lags .+ 1;

# Smoothed states
smoothed_trends      = zeros(n_series, size(smoothed_states, 2));
smoothed_idio_cycles = zeros(n_series, size(smoothed_states, 2));
smoothed_bc_cycle    = zeros(n_series, size(smoothed_states, 2));
smoothed_ep_cycle    = zeros(n_series, size(smoothed_states, 2));

for i=1:n_series

    # Trend and idiosyncratic cycle
    ind_excluding_common_cycles = findall(sspace.B[i, 1:end-common_cycles_offset] .!= 0.0);
    smoothed_trends[i, :]      = std_diff_data[i] .* smoothed_states[ind_excluding_common_cycles[1], :] .* sspace.B[i, ind_excluding_common_cycles[1]];
    smoothed_idio_cycles[i, :] = std_diff_data[i] .* smoothed_states[ind_excluding_common_cycles[2], :] .* sspace.B[i, ind_excluding_common_cycles[2]];

    # Common cycles
    smoothed_bc_cycle[i, :] = std_diff_data[i] .* smoothed_states[bc_coordinates, :]' * sspace.B[i, bc_coordinates];
    if compute_ep_cycle
        smoothed_ep_cycle[i, :] = std_diff_data[i] .* smoothed_states[ep_coordinates, :]' * sspace.B[i, ep_coordinates];
    end
end

# Custom rescaling
iis_data ./= custom_rescaling;
smoothed_trends ./= custom_rescaling;
smoothed_idio_cycles ./= custom_rescaling;
smoothed_bc_cycle ./= custom_rescaling;
smoothed_ep_cycle ./= custom_rescaling;

#=
Plotting stage
=#

if pre_covid
    ref_dates_fig = data_vintages[812][!,1];
else
    ref_dates_fig = data_vintages[end][!,1];
end

ref_dates_grid = ref_dates_fig[1]:Month(1):ref_dates_fig[end];

using Colors;
c1 = colorant"rgba(0, 48, 158, .75)";
c2 = colorant"rgba(255, 0, 0, .75)";
c3 = colorant"rgba(255, 190, 0, .75)";


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

pgfsave("./img/cycles_$(compute_ep_cycle)_$(pre_covid).pdf", fig);

#=
--------------------------------------------------------------------------------------------------------------------------------
Trends
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

pgfsave("./img/trends_$(compute_ep_cycle)_$(pre_covid).pdf", fig);