# Libraries
using Distributed;
@everywhere using MessyTimeSeriesOptim;
@everywhere include("./get_real_time_datasets.jl");
using CSV, FileIO, JLD;
using Random, LinearAlgebra, MessyTimeSeries;

"""
    simulate_data(
        T            :: Int64,
        lin_coeff    :: Float64,
        nlin_coeff_1 :: Float64,
        nlin_coeff_2 :: Float64,
        nlin_weight  :: Float64;
        seed         :: Int64 = 1,
        burnin       :: Int64 = 100,
    )

Return simulated data.
"""
function simulate_data(
    T            :: Int64,
    lin_coeff    :: Float64,
    nlin_coeff_1 :: Float64,
    nlin_coeff_2 :: Float64,
    nlin_weight  :: Float64;
    seed         :: Int64 = 1,
    burnin       :: Int64 = 100,
)

    # Set random seed
    Random.seed!(seed);

    # Pre-allocate memory for output
    cycle = zeros(T+burnin);
    target = zeros(T+burnin);
    
    # Loop over time
    for t=3:T+burnin
        
        # Cycle value for time t
        cycle[t] = 1.419*cycle[t-1] -0.544*cycle[t-2] + 0.0201*randn(); # parameters from Clark (1987, table 2)
        
        # Target value for time t
        nlin_threshold = cycle[t-1] <= 0;
        target[t] += nlin_weight*(nlin_coeff_1*nlin_threshold + nlin_coeff_2*(1-nlin_threshold));
        target[t] += (1-nlin_weight)*(lin_coeff*cycle[t-1]);
    end

    # Return output
    return cycle[burnin+1:end], target[burnin+1:end];
end

cycle, target = simulate_data(
    100,
    1.0,
    -0.20,
    +0.15,
    0.5,
    burnin=100
);
