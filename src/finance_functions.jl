"""
    populate_factors_matrices!(factors_matrices::Vector{FloatMatrix}, factors_coordinates::IntVector, factors_associated_scaling::FloatVector, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

Populate `factors_matrices` to construct the data subsamples.
"""
function populate_factors_matrices!(factors_matrices::Vector{FloatMatrix}, factors_coordinates::IntVector, factors_associated_scaling::FloatVector, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

    # Run smoother and forecast to reconstruct the past, current and expected future value for the factors
    X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status, t-lags+1); # smooth up to time period `t-lags+1` (i.e., 1 in the first call and referring to time t==lags)
    X_fc = kforecast(sspace, status.X_post, lags-1);                # compute `lags-1` predictions for the states

    # Loop over factor selection
    for i in axes(factors_matrices, 1)
        
        # Convenient pointers
        current_factor_matrix = factors_matrices[i];
        current_factor_coordinates = factors_coordinates[i];
        current_factor_associated_scaling = factors_associated_scaling[i];

        # Store lags and expectations for the factor
        for j=1:lags-1
            current_factor_matrix[j, t-lags+1] = X_sm[j][current_factor_coordinates] * current_factor_associated_scaling;       # from 1 to lags-1
            current_factor_matrix[j+lags, t-lags+1] = X_fc[j][current_factor_coordinates] * current_factor_associated_scaling;  # from lags+1 to 2*lags-1 (if lags==12 -> 13 to 23 as expected)
        end
        
        # Store present conditions for the factor
        current_factor_matrix[lags, t-lags+1] = X_sm[lags][current_factor_coordinates] * current_factor_associated_scaling;
    end
end

"""
    transform_latest_in_factors_matrices(factors_matrices::Vector{FloatMatrix}, t::Int64, lags::Int64)

Return a transformed version of the latest non-zero columns in `factors_matrices`.
"""
function transform_latest_in_factors_matrices(factors_matrices::Vector{FloatMatrix}, t::Int64, lags::Int64)

    # Pre-allocate container for output
    transformed_factor_vectors = Vector{FloatVector}(undef, length(factors_matrices));

    # Loop over factor selection
    for i in axes(factors_matrices, 1)

        # Convenient pointer
        current_factor_matrix = factors_matrices[i];

        # Initialise `transformed_current_factor_vector` with the latest non-zero column in `current_factor_matrix`
        transformed_current_factor_vector = current_factor_matrix[:, t-lags+1];
        
        # Changes
        transformed_current_factor_vector = vcat(transformed_current_factor_vector, diff(transformed_current_factor_vector));

        if lags > 2
            
            # Present vs past (excl. BC_{t} - BC_{t-1} since it is already accounted for in the changes)
            transformed_current_factor_vector = vcat(transformed_current_factor_vector, transformed_current_factor_vector[lags] .- transformed_current_factor_vector[1:lags-2]);

            # Future conditions vs present (excl. BC_{t+1} - BC_{t} since it is already accounted for in the changes)
            transformed_current_factor_vector = vcat(transformed_current_factor_vector, transformed_current_factor_vector[lags+2:2*lags-1] .- transformed_current_factor_vector[lags]);
        end

        # Update output
        transformed_factor_vectors[i] = transformed_current_factor_vector;
    end

    # Return output
    return transformed_factor_vectors;
end

"""
    get_latest_in_factors_matrices(factors_matrices::Vector{FloatMatrix}, t::Int64, lags::Int64)

Return the latest non-zero columns in `factors_matrices`.
"""
get_latest_in_factors_matrices(factors_matrices::Vector{FloatMatrix}, t::Int64, lags::Int64) = [factors_matrices[i][:, t-lags+1] for i in axes(factors_matrices, 1)];

"""
    populate_predictors_matrix!(predictors_matrix::FloatMatrix, equity_index::FloatVector, transformed_factor_vectors::Vector{FloatVector}, t::Int64, lags::Int64)

Populate `predictors_matrix` to construct the data subsamples.
"""
function populate_predictors_matrix!(predictors_matrix::FloatMatrix, equity_index::FloatVector, transformed_factor_vectors::Vector{FloatVector}, t::Int64, lags::Int64)
    
    # Autoregressive predictors
    for i=1:lags
        predictors_matrix[i, t-lags+1] = equity_index[t-lags+i];
    end
    
    # Add transformed_factor_matrices to predictors (if required)
    if length(transformed_factor_vectors) > 0
        predictors_matrix[lags+1:end, t-lags+1] = vcat(transformed_factor_vectors...);
    end
end

"""
    get_macro_data_partitions(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool, compute_ep_cycle::Bool, n_cycles::Int64, coordinates_params_rescaling::Vector{Vector{Int64}}, existing_estim::Union{Nothing, EstimSettings}=nothing, existing_std_diff_data::Union{Nothing, FloatVector}=nothing)

Return macro data partitions compatible with tree ensembles.
"""
function get_macro_data_partitions(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool, compute_ep_cycle::Bool, n_cycles::Int64, coordinates_params_rescaling::Vector{Vector{Int64}}, existing_estim::Union{Nothing, EstimSettings}=nothing, existing_std_diff_data::Union{Nothing, FloatVector}=nothing)
    
    # Extract data from `macro_vintage`
    macro_data = macro_vintage[:, 2:end] |> JMatrix{Float64};
    macro_data = permutedims(macro_data);
    
    # Check size of `equity_index`
    if length(equity_index) != size(macro_data, 2) + 1
        error("`equity_index` must have an extra entry at the end compared to the macro vintage! Note that the first observation must refer to the same point in time.");
    end

    # Initial settings
    lags = Int64(optimal_hyperparams[1]);

    # Predictors    
    predictors_matrix = zeros(lags + include_factor_augmentation*(1+compute_ep_cycle)*(use_refined_BC*(6*lags-7) + (1-use_refined_BC)*(2*lags-1)), size(macro_data, 2)-lags+1); # includes both the autoregressive part and the factor augmentation (if any) and its transformations (if required)
    
    if include_factor_augmentation

        # Get trend-cycle model structure (estimated with data up to t0 - included)
        estim, std_diff_data = get_tc_structure(macro_data[:, 1:t0], optimal_hyperparams, model_args, model_kwargs, coordinates_params_rescaling, existing_estim, existing_std_diff_data);

        # Estimate the trend-cycle model with (estimated with data up to t0 - included)
        sspace = ecm(estim, output_sspace_data=macro_data./std_diff_data); # using the optional keyword argument `output_sspace_data` allows to construct the validation samples
        status = DynamicKalmanStatus();

        # Factors data
        factors_matrices = [zeros(2*lags-1, sspace.Y.T-lags+1)]; # `2*lags-1` denotes the present plus `lags-1` lags (realised) and `lags-1` forward points (expected), sspace.Y.T refers to the full `macro_data` (i.e., not just up to t0) due to the keyword argument discussed above
        factors_coordinates = [findlast(sspace.B[1, :] .== 1)];  # business cycle's coordinates
        factors_associated_scaling = [std_diff_data[1]];

        if compute_ep_cycle
            push!(factors_matrices, zeros(2*lags-1, sspace.Y.T-lags+1));
            push!(factors_coordinates, findlast(sspace.B[n_cycles, :] .== 1)); # energy price cycle's coordinates
            push!(factors_associated_scaling, std_diff_data[n_cycles]);
        end
    end

    # Compute the business cycle vintages required to structure the estimation and validation samples in a pseudo out-of-sample fashion
    for t in axes(macro_data, 2)
        
        # Kalman filter iteration
        if include_factor_augmentation
            kfilter!(sspace, status);
        end

        if t >= lags
            
            if include_factor_augmentation

                # Populate `factors_matrices` (the first reference data is the time period t==lags)
                populate_factors_matrices!(factors_matrices, factors_coordinates, factors_associated_scaling, sspace, status, t, lags);

                # Generate `transformed_factor_vectors` (i.e., transform the latest column in the entries of `factors_matrices`)
                if use_refined_BC
                    transformed_factor_vectors = transform_latest_in_factors_matrices(factors_matrices, t, lags);
                else
                    transformed_factor_vectors = get_latest_in_factors_matrices(factors_matrices, t, lags);
                end
                
            else
                transformed_factor_vectors = Vector{FloatVector}();
            end
            
            # Populate `predictors_matrix` (the first reference data is the time period t==lags)
            populate_predictors_matrix!(predictors_matrix, equity_index, transformed_factor_vectors, t, lags);
        end
    end

    # The first column in `predictors_matrix` includes data on the `equity_index` from 1 to lags thus `target_vector` starts from lags+1
    target_vector = equity_index[lags+1:end]; # it has the same size than `predictors_matrix` since `equity_index` has an extra entry at the end
    
    # Estimation is up to t==t0
    estimation_samples_target = target_vector[1:t0-lags+1];
    estimation_samples_predictors = predictors_matrix[:, 1:t0-lags+1];
    
    # Validation is for t>t0
    validation_samples_target = target_vector[t0-lags+2:end];
    validation_samples_predictors = predictors_matrix[:, t0-lags+2:end];

    # Return output
    return estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors;
end

"""
    estimate_and_validate(estimation_samples_target::FloatVector, estimation_samples_predictors::FloatMatrix, validation_samples_target::FloatVector, validation_samples_predictors::FloatMatrix, model::Any, model_settings::NamedTuple)

Estimate and validate `model` given the settings in `model_settings`.
"""
function estimate_and_validate(estimation_samples_target::FloatVector, estimation_samples_predictors::FloatMatrix, validation_samples_target::FloatVector, validation_samples_predictors::FloatMatrix, model::Any, model_settings::NamedTuple)
    
    # Generate `model` instance
    model_instance = model(; model_settings...);
    @infiltrate

    # Estimation
    ScikitLearn.fit!(model_instance, permutedims(estimation_samples_predictors), estimation_samples_target); # in ScikitLearn all input predictors matrices are vertical - i.e., of shape (n_sample, n_feature)
    @infiltrate

    # Validation sample forecasts
    validation_forecasts = ScikitLearn.predict(model_instance, permutedims(validation_samples_predictors)); # in ScikitLearn all input predictors matrices are vertical - i.e., of shape (n_sample, n_feature)
    @infiltrate

    # Compute validation error
    validation_error = mean((validation_samples_target .- validation_forecasts).^2);
    @infiltrate

    # Return output
    return model_instance, validation_forecasts, validation_error;
end
