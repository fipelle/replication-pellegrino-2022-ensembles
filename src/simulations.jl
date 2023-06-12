# Libraries
using Distributed;
using FileIO, JLD;
using DecisionTree, Distributions, LinearAlgebra, Random, Statistics;

"""
    simulate_data(
        T            :: Int64,
        lin_coeff    :: Float64,
        nlin_coeff_1 :: Float64,
        nlin_coeff_2 :: Float64,
        nlin_weight  :: Float64;
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
    burnin       :: Int64 = 100,
)

    # Pre-allocate memory for output
    cycle = zeros(T+burnin);
    target = zeros(T+burnin);
    
    # Loop over time
    for t=3:T+burnin
        
        # Cycle value for time t
        cycle[t] = 1.53*cycle[t-1] -0.59*cycle[t-2] + 0.0074*randn(); # parameters from Clark (1987, table 1) to resemble something akin to the business cycle

        # Target value for time t
        nlin_threshold = cycle[t-1] <= 0;
        target[t] += nlin_weight*(nlin_coeff_1*nlin_threshold + nlin_coeff_2*(1-nlin_threshold));
        target[t] += (1-nlin_weight)*(lin_coeff*cycle[t-1]);
    end

    # Return output
    return cycle[burnin+1:end], target[burnin+1:end];
end

"""
    run_simulations(
        T              :: Int64,
        no_simulations :: Int64,
        noise_factor   :: Float64
    )

Run simulations.
"""
function run_simulations(
    T              :: Int64,
    no_simulations :: Int64,
    noise_factor   :: Float64
)

    # Estimation sample length
    estim_length = fld(T, 2);

    # Memory pre-allocation for output
    ols_errors = zeros(11);
    rf_errors = zeros(11, 2); # (no nlin_weight x max_depths)
    
    for simulation in collect(1:no_simulations)
        
        if mod(simulation, 50) == 0
            @info("Simulation $(simulation)");
        end
        
        # Set random seed
        Random.seed!(simulation);

        # Draw parameters
        nlin_coeff_1 = rand(Uniform(-0.1, 0.0));
        nlin_coeff_2 = rand(Uniform(0.0, +0.1));
        
        # Loop over non linear weight
        for (index, nlin_weight) in enumerate(collect(0.0:0.1:1.0))
            
            cycle, target = simulate_data(
                T,
                1.0,
                nlin_coeff_1,
                nlin_coeff_2,
                nlin_weight,
                burnin=100,
            );

            # Is the cycle observed with some measurement error?
            if noise_factor > 0
                cycle .+= noise_factor .* randn(T);
            end

            # Estimation sample
            X_estim = cycle[1:estim_length-1];
            y_estim = target[2:estim_length];

            # OOS sample
            X_oos = cycle[estim_length:end-1];
            y_oos = target[estim_length+1:end];

            # OLS error
            ols = (X_estim'*X_estim)\X_estim'*y_estim;
            ols_oos_fc = ols*X_oos;
            ols_errors[index] += mean((y_oos-ols_oos_fc).^2);

            #=
            Tree ensemble error
            - for max tree depth 1, 2 and 3
            - with a large number of constituent trees
            =#

            for max_depth=[1,2]

                # Model instance
                rf_instance = RandomForestRegressor(rng=simulation, n_trees=1000, partial_sampling=1.0, n_subfeatures=1, max_depth=max_depth);

                # Fit
                DecisionTree.fit!(rf_instance, X_estim[:,:], y_estim);

                # Compute error
                rf_oos_fc = DecisionTree.predict(rf_instance, X_oos[:,:]);
                rf_errors[index, max_depth] += mean((y_oos-rf_oos_fc).^2); # WARNING: indexing over max_depth is fine as long as there aren't breaks or jumps in the grid
            end
        end
    end

    # Average over simulations
    ols_errors ./= no_simulations;
    rf_errors ./= no_simulations;

    # Return output
    return ols_errors, rf_errors;
end

# Run simulations
ols_errors_T100_noise0, rf_errors_T100_noise0 = run_simulations(100, 1000, 0.0);
ols_errors_T100_noise1, rf_errors_T100_noise1 = run_simulations(100, 1000, 0.0074); # parameters from Clark (1987, table 1) to resemble something akin to the business cycle
ols_errors_T200_noise0, rf_errors_T200_noise0 = run_simulations(200, 1000, 0.0);
ols_errors_T200_noise1, rf_errors_T200_noise1 = run_simulations(200, 1000, 0.0074); # parameters from Clark (1987, table 1) to resemble something akin to the business cycle

# Save output to disk
save("./simulations/simulations.jld",
    Dict(
        "T100_noise0" => (ols_errors_T100_noise0, rf_errors_T100_noise0),
        "T100_noise1" => (ols_errors_T100_noise1, rf_errors_T100_noise1),
        "T200_noise0" => (ols_errors_T200_noise0, rf_errors_T200_noise0),
        "T200_noise1" => (ols_errors_T200_noise1, rf_errors_T200_noise1)
    )
);
