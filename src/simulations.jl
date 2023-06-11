# Libraries
using Distributed;
using FileIO, JLD;
using DecisionTree, LinearAlgebra, Random, Statistics;

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

# Memory pre-allocation for output
ols_errors = zeros(11);
rf_errors = zeros(11, 2);

for simulation in collect(1:1000)
    
    @info("Simulation $(simulation)");

    # Loop over non linear weight
    for (index, nlin_weight) in enumerate(collect(0:0.1:1))
        
        cycle, target = simulate_data(
            100,
            1.0,
            -0.20,
            +0.15,
            nlin_weight,
            burnin=100,
            seed=simulation
        );

        # Estimation sample
        X = cycle[1:end-1];
        y = target[2:end];

        # OLS error
        ols = (X'*X)\X'*y;
        ols_iis_fc = ols*X;
        ols_errors[index] += mean((y-ols_iis_fc).^2);

        #=
        Tree ensemble error
        - for max tree depth 1, 2 and 3
        - with a large number of constituent trees
        =#

        for max_depth=[1,2]

            # Model instance
            rf_instance = RandomForestRegressor(rng=simulation, n_trees=1000, partial_sampling=1.0, n_subfeatures=1, max_depth=max_depth);

            # Fit
            DecisionTree.fit!(rf_instance, X[:,:], y);

            # Compute error
            rf_iis_fc = DecisionTree.predict(rf_instance, X[:,:]);
            rf_errors[index, max_depth] += mean((y-rf_iis_fc).^2); # WARNING: indexing over max_depth is fine as long as there aren't breaks or jumps in the grid
        end
    end
end

# Average over simulations
ols_errors ./= 1000;
rf_errors ./= 1000;

# Save output to disk
save("./simulations/simulations.jld",
    Dict(
        "ols_errors" => ols_errors,
        "rf_errors" => rf_errors,
    )
);
