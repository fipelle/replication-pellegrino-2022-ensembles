__precompile__()

module TimeSeriesEnsembles

    using DecisionTree, Random, StableRNGs, TSAnalysis;

    """
        compute_aggregator_subsamples!(target_subsamples::Vector{FloatVector}, predictors_subsamples::Vector{FloatMatrix}, target::FloatVector, predictors::FloatMatrix, rng::AbstractRNG, n_bootstrap_samples::Int64, extra_bootstrap::Bool)

    Compute bootstrap samples of (target, predictors) tuples.

        compute_aggregator_subsamples!(target_subsamples::Vector{FloatVector}, predictors_subsamples::Vector{FloatMatrix}, target::Vector{FloatVector}, predictors::Vector{FloatMatrix}, rng::AbstractRNG, n_bootstrap_samples::Int64, extra_bootstrap::Bool)
    
    Compute subsamples starting from user-defined partitions of both the target and predictors.

    # References
    Efron and Gong (1983, section 7) for the pair Bootstrap.
    """
    function compute_aggregator_subsamples!(target_subsamples::Vector{FloatVector}, predictors_subsamples::Vector{FloatMatrix}, target::FloatVector, predictors::FloatMatrix, rng::AbstractRNG, n_bootstrap_samples::Int64, extra_bootstrap::Bool)
        
        T = length(target);

        for i=1:n_bootstrap_samples
            selection = rand(rng, 1:T, T);
            push!(target_subsamples, target[selection]);
            push!(predictors_subsamples, predictors[selection, :]);
        end
    end

    function compute_aggregator_subsamples!(target_subsamples::Vector{FloatVector}, predictors_subsamples::Vector{FloatMatrix}, target::Vector{FloatVector}, predictors::Vector{FloatMatrix}, rng::AbstractRNG, n_bootstrap_samples::Int64, extra_bootstrap::Bool)

        # Loop over the user-defined partitions
        for i in axes(target, 1)

            # Compute a fixed number of bootstrap (target, predictors) subsamples for each user-defined partition
            if extra_bootstrap
                current_target_subsamples = Vector{FloatVector}();
                current_predictors_subsamples = Vector{FloatMatrix}();
                compute_aggregator_subsamples!(current_target_subsamples, current_predictors_subsamples, target[i], predictors[i], rng, n_bootstrap_samples, false);
                for j=1:n_bootstrap_samples
                    push!(target_subsamples, current_target_subsamples[j]);
                    push!(predictors_subsamples, current_predictors_subsamples[j]);
                end

            # Standard run
            else
                push!(target_subsamples, target[i]);
                push!(predictors_subsamples, predictors[i]);
            end
        end
    end

    """
        estimate_tree_aggregator(target::Union{FloatVector, Vector{FloatVector}}, predictors::Union{FloatMatrix, Vector{FloatMatrix}}; min_samples_leaf::Int64=1, rng::AbstractRNG=StableRNG(1), extra_bootstrap::Bool=false)
    
    Estimate a simple regression tree aggregator.

    # Arguments
    - `target`: response variable of interest or user-defined subsamples of the response variable of interest.
    - `predictors`: matrix of covariates or user-defined subsamples of the matrix of covariates.
    - `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 1).
    - `rng`: any AbstractRNG (default: StableRNG with seed 1).
    - `n_bootstrap_samples`: number bootstrap subsamples or extra bootstrap subsamples per user-defined partition.
    - `extra_bootstrap`: compute a fixed number of bootstrap (target, predictors) subsamples for each user-defined partition. It is not used when `target` and `predictors` are FloatVector and FloatMatrix.
    """
    function estimate_tree_aggregator(target::Union{FloatVector, Vector{FloatVector}}, predictors::Union{FloatMatrix, Vector{FloatMatrix}}; min_samples_leaf::Int64=1, rng::AbstractRNG=StableRNG(1), n_bootstrap_samples::Int64=500, extra_bootstrap::Bool=false)

        # Subsampling
        target_subsamples = Vector{FloatVector}();
        predictors_subsamples = Vector{FloatMatrix}();
        compute_aggregator_subsamples!(target_subsamples, predictors_subsamples, target, predictors, rng, n_bootstrap_samples, extra_bootstrap);

        # Memory pre-allocation: output
        n_trees = length(target_subsamples);
        trees = Vector{DecisionTreeRegressor}(undef, n_trees);

        # Estimate aggregator
        for i=1:n_trees
            trees[i] = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, rng=rng);
            DecisionTree.fit!(trees[i], predictors_subsamples[i], target_subsamples[i]);
        end

        # Return output
        return trees;
    end

    """
        forecast_tree_aggregator(predictors::FloatVector, trees::Vector{DecisionTreeRegressor})
        forecast_tree_aggregator(predictors::Vector{FloatVector}, trees::Vector{DecisionTreeRegressor})
    
    Compute aggregate forecast.

    # Arguments
    - `predictors`: vector of covariates (i.e., latest covariates)
    - `trees`: aggregator estimated via `estimate_tree_aggregator`
    """
    function forecast_tree_aggregator(predictors::FloatVector, trees::Vector{DecisionTreeRegressor})

        # Initial settings
        output = 0.0;
        n_trees = length(trees);

        # Compute aggregate forecast
        for tree in trees
            output += DecisionTree.predict(tree, predictors)[1];
        end
        output /= n_trees;

        # Return output
        return output;
    end

    function forecast_tree_aggregator(predictors::Vector{FloatVector}, trees::Vector{DecisionTreeRegressor})

        # Initial settings
        output = 0.0;
        n_trees = length(trees);

        # Compute aggregate forecast
        for i=1:n_trees
            output += DecisionTree.predict(trees[i], predictors[i])[1];
        end
        output /= n_trees;

        # Return output
        return output;
    end

    export estimate_tree_aggregator, forecast_tree_aggregator;
end
