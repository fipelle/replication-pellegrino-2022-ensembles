# Libraries
using Distributed;
using FileIO, JLD;
using DecisionTree, Distributions, LinearAlgebra, Random, Statistics;
using PGFPlotsX, LaTeXStrings;

# Colors
using Colors;
c1 = colorant"rgba(0, 48, 158, .75)";
c2 = colorant"rgba(255, 0, 0, .75)";
c3 = colorant"rgba(255, 190, 0, .75)";

# Load simulations output
simulations_output = load("./simulations/simulations.jld");

# Initialise chart
axs = Array{Any}(undef, 4);
titles = ["T=100, true cycle", "T=200, true cycle", "T=100, imperfect cycle", "T=200, imperfect cycle"];

# Loop over simulation sets
for (index, simulation_set) in enumerate(["T100_noise0", "T200_noise0", "T100_noise1", "T200_noise1"])
    ols_errors, rf_errors = simulations_output[simulation_set];

    axs[index] = @pgf Axis(
        {
            grid   = "both",
            xlabel = raw"Non-linearities strenght",
            ylabel = raw"In-sample error",
            xmin=0, xmax=1.0,
            ymin=0, ymax=0.5,
            title=titles[index]
        },
        Plot({color=c1}, Table(x=collect(0.0:0.1:1.0), y=ols_errors)),
        Plot({color=c2}, Table(x=collect(0.0:0.1:1.0), y=rf_errors[:,1])),
    );
end
