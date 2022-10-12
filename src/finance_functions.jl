"""
    populate_factors_matrices!(factors_matrices::Vector{FloatMatrix}, factors_coordinates::IntVector, factors_associated_scaling::FloatVector, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

Populate `factors_matrices` to construct the data subsamples.
"""
function populate_factors_matrices!(factors_matrices::Vector{FloatMatrix}, factors_coordinates::IntVector, factors_associated_scaling::FloatVector, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

    @infiltrate

    # Run smoother and forecast to reconstruct the past, current and expected future value for the factors
    X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status, t-lags+1);
    X_fc = kforecast(sspace, status.X_post, lags-1);

    @infiltrate

    for i in axes(factors_matrices, 1)
        
        @infiltrate

        # Convenient pointer
        current_factor_matrix = factors_matrices[i];
        current_factor_coordinates = factors_coordinates[i];
        current_factor_associated_scaling = factors_associated_scaling[i];

        @infiltrate

        # Store smoother and forecast output
        for j=1:lags-1

            @infiltrate

            current_factor_matrix[j, t-lags+1] = X_sm[j][current_factor_coordinates] * current_factor_associated_scaling;
            current_factor_matrix[j+lags, t-lags+1] = X_fc[j][current_factor_coordinates] * current_factor_associated_scaling;
        
            @infiltrate
        end

        # Store final smoother output
        @infiltrate

        current_factor_matrix[lags, t-lags+1] = X_sm[lags][current_factor_coordinates] * current_factor_associated_scaling;
        
        @infiltrate
    end
end

"""
    get_target_and_predictors_estimation(current_business_cycle_matrix::FloatMatrix, current_equity_index::FloatMatrix, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return target and predictors for estimation.
"""
function get_target_and_predictors_estimation(current_business_cycle_matrix::FloatMatrix, current_equity_index::FloatMatrix, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Macro
    predictors_business_cycle = current_business_cycle_matrix[:, 1:end-1];

    if use_refined_BC

        # Changes
        predictors_business_cycle = vcat(predictors_business_cycle, diff(predictors_business_cycle, dims=1));

        if lags > 2
        
            # Present vs past (excl. BC_{t} - BC_{t-1} since it is already accounted for in the changes)
            predictors_business_cycle = vcat(predictors_business_cycle, predictors_business_cycle[lags, :]' .- predictors_business_cycle[1:lags-2, :]);
            
            # Future conditions vs present (excl. BC_{t+1} - BC_{t} since it is already accounted for in the changes)
            predictors_business_cycle = vcat(predictors_business_cycle, predictors_business_cycle[lags+2:end, :] .- predictors_business_cycle[lags, :]');
        end
    end

    # Finance
    target, predictors_equity_index = lag(current_equity_index, lags);
    predictors_equity_index = reverse(predictors_equity_index, dims=1); # for internal consistency with `predictors_business_cycle`

    # Predictors
    if include_factor_augmentation
        predictors = vcat(predictors_equity_index, predictors_business_cycle);
    else
        predictors = predictors_equity_index;
    end

    # Transpose target and predictors to match DecisionTree
    target = target[:];
    predictors = permutedims(predictors);

    # Return output
    return target, predictors;
end

"""
    get_target_and_predictors_forecasting(last_business_cycle_matrix_col::FloatVector, current_equity_index::FloatMatrix, next_equity_index_obs::Float64, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)
    get_target_and_predictors_forecasting(current_business_cycle_matrix::FloatMatrix, current_equity_index::FloatMatrix, next_equity_index_obs::Float64, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return target and predictors for forecasting.
"""
function get_target_and_predictors_forecasting(last_business_cycle_matrix_col::FloatVector, current_equity_index::FloatMatrix, next_equity_index_obs::Float64, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Macro
    predictors_business_cycle = copy(last_business_cycle_matrix_col);

    if use_refined_BC

        # Changes
        predictors_business_cycle = vcat(predictors_business_cycle, diff(predictors_business_cycle, dims=1));

        if lags > 2
        
            # Present vs past (excl. BC_{t} - BC_{t-1} since it is already accounted for in the changes)
            predictors_business_cycle = vcat(predictors_business_cycle, predictors_business_cycle[lags] .- predictors_business_cycle[1:lags-2]);
            
            # Future conditions vs present (excl. BC_{t+1} - BC_{t} since it is already accounted for in the changes)
            predictors_business_cycle = vcat(predictors_business_cycle, predictors_business_cycle[lags+2:end] .- predictors_business_cycle[lags]);
        end
    end

    # Finance
    predictors_equity_index = current_equity_index[1, end-lags+1:end];

    # Predictors
    if include_factor_augmentation
        predictors = vcat(predictors_equity_index, predictors_business_cycle);
    else
        predictors = predictors_equity_index;
    end

    # Outturn
    outturn = next_equity_index_obs;

    # Return output
    return outturn, predictors;
end

get_target_and_predictors_forecasting(current_business_cycle_matrix::FloatMatrix, current_equity_index::FloatMatrix, next_equity_index_obs::Float64, lags::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool) = get_target_and_predictors_forecasting(current_business_cycle_matrix[:, end], current_equity_index, next_equity_index_obs, lags, include_factor_augmentation, use_refined_BC);

"""
"""
function update_estimation_samples!(
    estimation_samples_target, 
    estimation_samples_predictors, 
    business_cycle_matrix_selection::FloatMatrix, 
    current_equity_index::Int64, 
    lags::Int64, 
    include_factor_augmentation::Bool, 
    use_refined_BC::Bool
    )
    
    @infiltrate
    estimation_target, estimation_predictors = get_target_and_predictors_estimation(business_cycle_matrix_selection, current_equity_index, lags, include_factor_augmentation, use_refined_BC);
    @infiltrate

    push!(estimation_samples_target, estimation_target);
    push!(estimation_samples_predictors, estimation_predictors);
end

"""
"""
function update_validation_samples!(
    validation_samples_target, 
    validation_samples_predictors, 
    current_business_cycle_matrix::FloatMatrix, 
    current_equity_index::Int64, 
    next_equity_index_obs::Float64, 
    lags::Int64, 
    include_factor_augmentation::Bool, 
    use_refined_BC::Bool
    )

    @infiltrate
    validation_target, validation_predictors = get_target_and_predictors_forecasting(current_business_cycle_matrix, current_equity_index, next_equity_index_obs, lags, include_factor_augmentation, use_refined_BC);
    @infiltrate

    push!(validation_samples_target, validation_target);
    push!(validation_samples_predictors, validation_predictors);
end

"""
    get_macro_data_partitions(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool, compute_ep_cycle::Bool, n_cycles::Int64, coordinates_params_rescaling::Vector{Vector{Int64}})

Return macro data partitions compatible with tree ensembles.
"""
function get_macro_data_partitions(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool, compute_ep_cycle::Bool, n_cycles::Int64, coordinates_params_rescaling::Vector{Vector{Int64}})
    
    # Extract data from `macro_vintage`
    macro_data = macro_vintage[:, 2:end] |> JMatrix{Float64};
    macro_data = permutedims(macro_data);
    
    # Initial settings
    lags = Int64(optimal_hyperparams[1]);

    # Memory pre-allocation for output arrays
    estimation_samples_target = FloatVector[];              # TBC this should be equal to equity_index[lags+1:t0]
    estimation_samples_predictors = FloatMatrix[];          # TBC this should refer to lags:t0-1
    validation_samples_target = FloatVector[];              # TBC this should be equal to equity_index[t0+1:T] or equity_index[t0+1:T+1]
    validation_samples_predictors = Vector{FloatVector}();  # TBC this should refer to t0:T-1 or t0:T

    # Get trend-cycle model structure (estimated with data up to t0 - included)
    estim, std_diff_data = get_tc_structure(macro_data[:, 1:t0], optimal_hyperparams, model_args, model_kwargs, coordinates_params_rescaling);

    # Estimate the trend-cycle model with (estimated with data up to t0 - included)
    sspace = ecm(estim, output_sspace_data=macro_data./std_diff_data); # using the optional keyword argument `output_sspace_data` allows to construct the validation samples
    status = DynamicKalmanStatus();

    # Factors data
    factors_matrices = [zeros(2*lags-1, sspace.Y.T-lags+1)]; # `2*12-1` denotes the present plus 6 lags (realised) and 6 forward points (expected), sspace.Y.T refers to the full `macro_data` (i.e., not just up to t0) due to the keyword argument discussed above
    factors_coordinates = [findlast(sspace.B[1, :] .== 1)];  # business cycle's coordinates
    factors_associated_scaling = [std_diff_data[1]];

    if compute_ep_cycle
        push!(factors_matrices, zeros(2*lags-1, sspace.Y.T-lags+1));
        push!(factors_coordinates, findlast(sspace.B[n_cycles, :] .== 1)); # energy price cycle's coordinates
        push!(factors_associated_scaling, std_diff_data[n_cycles]);
    end

    @infiltrate

    # Compute the business cycle vintages required to structure the estimation and validation samples in a pseudo out-of-sample fashion
    for t in axes(macro_data, 2)
        
        # Kalman filter iteration
        kfilter!(sspace, status);

        # Populate `factors_matrices`
        if t >= lags
            populate_factors_matrices!(factors_matrices, factors_coordinates, factors_associated_scaling, sspace, status, t, lags);
        end

        if t >= t0

            @infiltrate

            # Input data based on model estimated with data up to t0 (included)
            current_equity_index = permutedims(equity_index[1:t]);
            next_equity_index_obs = equity_index[t+1]; # Float64
            current_business_cycle_matrix = business_cycle_matrix[:, 1:t-lags+1]; # also include most recent obs.

            @infiltrate
            
            # Update `estimation_samples_target` and `estimation_samples_predictors`
            if t == t0
                update_estimation_samples!(
                    estimation_samples_target, 
                    estimation_samples_predictors, 
                    current_business_cycle_matrix,
                    current_equity_index,
                    lags, 
                    include_factor_augmentation, 
                    use_refined_BC
                    )
            
            # Update `validation_samples_target` and `validation_samples_predictors`
            else
                update_validation_samples!(
                    validation_samples_target, 
                    validation_samples_predictors, 
                    current_business_cycle_matrix, 
                    current_equity_index, 
                    next_equity_index_obs, 
                    lags, 
                    include_factor_augmentation, 
                    use_refined_BC
                    )
            end

            @infiltrate
        end
    end

    # Return output
    return estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors;
end
