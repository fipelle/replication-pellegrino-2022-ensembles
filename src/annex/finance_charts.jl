# Libraries
using CSV, DataFrames, Dates, FileIO, JLD, Logging;
using LinearAlgebra, MessyTimeSeries, MessyTimeSeriesOptim, DecisionTree, StableRNGs, Statistics;
using PGFPlotsX, LaTeXStrings;
include("./yrl/replication-pellegrino-2022-ensembles/src/macro_functions.jl");
include("./yrl/replication-pellegrino-2022-ensembles/src/finance_functions.jl");

"""
    get_label_importance(jld_path::String, labels::Vector{String}; aggregate::Bool=true)

Retrieve data from jld file and process it to be compatible with PGFPlotsX.
"""
function get_label_importance(jld_path::String, labels::Vector{String}; aggregate::Bool=true)
    
    # Recover output
    output = jldopen(jld_path);
    optimal_rf_instance = read(output["optimal_rf_instance"]);
    optimal_rf_importance = split_importance(optimal_rf_instance);

    # Data
    if !aggregate
        data = DataFrame([labels optimal_rf_importance], [:label, :importance]);
        sort!(data, [:importance], rev=true);
    else
        aggregate_labels = ["Autoregressive", "Augmentation"];
        optimal_rf_importance = [sum(optimal_rf_importance[1:12]), sum(optimal_rf_importance[13:end])];
        data = DataFrame([aggregate_labels optimal_rf_importance], [:label, :importance]);
    end

    # Return output data
    return data;
end

"""
    get_importance_label(data::DataFrame, i::Int64)

Convert i-th row of `data` into tuple.
"""
get_importance_label(data::DataFrame, i::Int64) = (data[i, :importance], data[i, :label]);

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

latex_ith(i::Int64) = ifelse(i==0, "", ifelse(i>0, "+$(i)", "$(i)"));

labels = vcat(
    ["\$ y_{t $(latex_ith(i))} \$" for i=11:-1:0], 
    # Levels
    ["\$ \\hat{\\psi}_{1,t $(latex_ith(i))} | t} \$" for i=-11:1:11],
    # Delta
    ["\$ \\hat{\\psi}_{1,t $(latex_ith(i)) | t} - \\hat{\\psi}_{1,t $(latex_ith(i-1)) | t} \$" for i=-10:1:11],
    # Wrt to time t
    ["\$ \\hat{\\psi}_{1,t | t} - \\hat{\\psi}_{1,t-$i | t} \$" for i=11:-1:2],
    ["\$ \\hat{\\psi}_{1,t+$i | t} - \\hat{\\psi}_{1,t | t} \$" for i=2:11],
)

# Colors
using Colors;
c1 = colorant"rgba(0, 48, 158, .75)";
c2 = colorant"rgba(255, 0, 0, .75)";

# ------------------------------------
# Aggregate
# ------------------------------------

best_kth = 2;
axs = Array{Any}(undef, 10);
for i in axes(axs, 1)

    # Data
    data_pre_covid  = get_label_importance("./yrl/replication-pellegrino-2022-ensembles/src/BC_output/1/output_equity_index_$(10+i)_false_true_true.jld", labels, aggregate=true);
    data_post_covid = get_label_importance("./mnt/replication-pellegrino-2022-ensembles/src/BC_output/1/output_equity_index_$(10+i)_false_true_true.jld", labels, aggregate=true);
    
    # Legend
    legend_style_content = ifelse(i==1, raw"{column sep = 10pt, legend columns = -1, legend to name = grouplegend, draw=none,}", "");

    # Most important predictors (pre covid) 
    most_important_labels = data_pre_covid[best_kth:-1:1, :label];

    axs[i] = @pgf Axis(
        {   xbar,
            grid = "both",
            "bar width"="10pt",
            xlabel = raw"importance",
            symbolic_y_coords=most_important_labels,
            ytick = "data",
            xmin=0, xmax=1.0,
            legend_style=legend_style_content,
            "enlarge y limits = 0.75",
        },

        Plot({color=c1, fill=c1}, Coordinates([get_importance_label(data_pre_covid, i) for i=1:best_kth])),
        Plot({color=c2, fill=c2}, Coordinates([ifelse(data_post_covid[i, :label] in most_important_labels, get_importance_label(data_post_covid, i), ()) for i in axes(data_post_covid, 1)])),
        ifelse(i==1, raw"\legend{Pre COVID-19, Post COVID-19}", "");
    );
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="2 by 5", vertical_sep="60pt", horizontal_sep="120pt", vertical_sep="50pt"},
      no_markers,
      legend_pos="north west",
      height="150pt",
      width="250pt"
    },
    axs...),
    raw"\node[yshift=-1.75cm] at ($(group c1r5.south)!0.5!(group c2r5.south)$) {\ref{grouplegend}};"
);

pgfsave("./img/importance_comparison_aggregate.pdf", fig);


# ------------------------------------
# Pre-covid
# ------------------------------------

best_kth = 10;
axs = Array{Any}(undef, 10);
for i in axes(axs, 1)

    # Data
    data_pre_covid  = get_label_importance("./yrl/replication-pellegrino-2022-ensembles/src/BC_output/1/output_equity_index_$(10+i)_false_true_true.jld", labels, aggregate=false);

    # Most important predictors (pre covid) 
    most_important_labels = data_pre_covid[best_kth:-1:1, :label];

    axs[i] = @pgf Axis(
        {   xbar,
            grid = "both",
            "bar width"="3pt",
            xlabel = raw"importance",
            symbolic_y_coords=most_important_labels,
            ytick = "data",
            xmin=0, xmax=0.6,
        },

        Plot({color=c1, fill=c1}, Coordinates([get_importance_label(data_pre_covid, i) for i=1:best_kth])),
    );
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="2 by 5", vertical_sep="60pt", horizontal_sep="120pt", vertical_sep="50pt"},
      no_markers,
      legend_pos="north west",
      height="200pt",
      width="250pt"
    },
    axs...),
);

pgfsave("./img/importance_pre_covid.pdf", fig);


# ------------------------------------
# Post-covid
# ------------------------------------

best_kth = 10;
axs = Array{Any}(undef, 10);
for i in axes(axs, 1)

    # Data
    data_post_covid  = get_label_importance("./mnt/replication-pellegrino-2022-ensembles/src/BC_output/1/output_equity_index_$(10+i)_false_true_true.jld", labels, aggregate=false);

    # Most important postdictors (post covid) 
    most_important_labels = data_post_covid[best_kth:-1:1, :label];

    axs[i] = @pgf Axis(
        {   xbar,
            grid = "both",
            "bar width"="3pt",
            xlabel = raw"importance",
            symbolic_y_coords=most_important_labels,
            ytick = "data",
            xmin=0, xmax=0.6,
        },

        Plot({color=c1, fill=c1}, Coordinates([get_importance_label(data_post_covid, i) for i=1:best_kth])),
    );
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="2 by 5", vertical_sep="60pt", horizontal_sep="120pt", vertical_sep="50pt"},
      no_markers,
      legend_pos="north west",
      height="200pt",
      width="250pt"
    },
    axs...),
);

pgfsave("./img/importance_post_covid.pdf", fig);
