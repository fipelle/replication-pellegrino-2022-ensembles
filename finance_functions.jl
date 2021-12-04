"""
    populate_business_cycle_matrix!(business_cycle_matrix::FloatMatrix, business_cycle_position::Int64, std_diff_data::FloatMatrix, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

Populate `business_cycle_matrix` to construct the estimation samples.
"""
function populate_business_cycle_matrix!(business_cycle_matrix::FloatMatrix, business_cycle_position::Int64, std_diff_data::FloatMatrix, sspace::KalmanSettings, status::DynamicKalmanStatus, t::Int64, lags::Int64)

    # Compute business_cycle_matrix for t (current and past values)
    X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status, t-lags+1);
    for i=1:lags
        business_cycle_matrix[i, t-lags+1] = X_sm[i][business_cycle_position] .* std_diff_data[1];
    end

    X_fc = kforecast(sspace, status.X_post, lags-1);
    for i=1:lags-1
        business_cycle_matrix[i+lags, t-lags+1] = X_fc[i][business_cycle_position] .* std_diff_data[1];
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
    get_selection_samples(macro_data::JMatrix{Float64}, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)
    get_selection_samples(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return selection samples. `macro_data` and `equity_index` are in the format required by TSAnalysis.jl.
"""
function get_selection_samples(macro_data::JMatrix{Float64}, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Initial settings
    lags = Int64(optimal_hyperparams[1]);

    # Memory pre-allocation for output
    estimation_samples_target = Vector{FloatVector}();
    estimation_samples_predictors = Vector{FloatMatrix}();
    validation_samples_target = Vector{Float64}();
    validation_samples_predictors = Vector{FloatVector}();

    # Get trend-cycle model structure (estimated with data up to t0 - included)
    estim, std_diff_data = get_tc_structure(macro_data[:, 1:t0], optimal_hyperparams, model_args, model_kwargs);

    # Get trend-cycle model structure (estimated with full data)
    estim_full, std_diff_data_full = get_tc_structure(macro_data, optimal_hyperparams, model_args, model_kwargs);

    # Estimate the trend-cycle model with (estimated with data up to t0 - included)
    sspace = ecm(estim, output_sspace_data=macro_data./std_diff_data);
    status = DynamicKalmanStatus();
    business_cycle_matrix = zeros(2*lags-1, sspace.Y.T-lags+1);

    # Estimate the trend-cycle model (estimated with full data)
    sspace_full = ecm(estim_full);
    status_full = DynamicKalmanStatus();
    business_cycle_matrix_full = zeros(2*lags-1, sspace.Y.T-lags+1);

    # Business cycle position in sspace
    business_cycle_position = findlast(sspace.B[1,:] .== 1);

    # Compute the business cycle vintages required to structure the estimation and validation samples in a pseudo out-of-sample fashion
    for t in axes(macro_data, 2)

        # Kalman filter iteration
        kfilter!(sspace, status);
        kfilter!(sspace_full, status_full);

        if t >= lags

            populate_business_cycle_matrix!(business_cycle_matrix, business_cycle_position, std_diff_data, sspace, status, t, lags);
            populate_business_cycle_matrix!(business_cycle_matrix_full, business_cycle_position, std_diff_data_full, sspace_full, status_full, t, lags);

            if t >= t0

                # Input data based on model estimated with data up to t0 (included)
                current_equity_index = permutedims(equity_index[1:t]);
                next_equity_index_obs = equity_index[t+1]; # Float64
                current_business_cycle_matrix = business_cycle_matrix[:, 1:t-lags+1]; # also include most recent obs.

                # Build predictors: estimation sample
                if (t == t0) || (t == sspace.Y.T)

                    # Get target and predictors (estimation)
                    if t == t0
                        estimation_target, estimation_predictors = get_target_and_predictors_estimation(current_business_cycle_matrix, current_equity_index, lags, include_factor_augmentation, use_refined_BC);
                    else
                        # `business_cycle_matrix_full` is fine since t == sspace.Y.T
                        estimation_target, estimation_predictors = get_target_and_predictors_estimation(business_cycle_matrix_full, current_equity_index, lags, include_factor_augmentation, use_refined_BC);
                    end

                    # Update output
                    push!(estimation_samples_target, estimation_target);
                    push!(estimation_samples_predictors, estimation_predictors);
                end

                # Build predictors: validation sample
                if t != t0

                    # Get target and predictors (forecasting)
                    validation_target, validation_predictors = get_target_and_predictors_forecasting(current_business_cycle_matrix, current_equity_index, next_equity_index_obs, lags, include_factor_augmentation, use_refined_BC);

                    # Update output
                    push!(validation_samples_target, validation_target);
                    push!(validation_samples_predictors, validation_predictors);
                end
            end
        end
    end

    # Return output 
    return sspace_full, std_diff_data_full, business_cycle_position, estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors;
end

function get_selection_samples(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Extract data from `macro_vintage`
    macro_data = macro_vintage[:, 2:end] |> JMatrix{Float64};
    macro_data = permutedims(macro_data);

    # Return selection samples
    return get_selection_samples(macro_data, equity_index, t0, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);
end

"""
    get_selection_samples_bootstrap(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return selection samples compatible with tree aggregators based on pair bootstrap.
"""
function get_selection_samples_bootstrap(macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)
    
    # Train and validation samples
    ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, estimation_samples_target, estimation_samples_predictors, validation_samples_target, validation_samples_predictors = get_selection_samples(macro_vintage, equity_index, t0, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);

    # Estimation and selection samples
    training_samples_target = estimation_samples_target[1];
    training_samples_predictors = estimation_samples_predictors[1];
    selection_samples_target = estimation_samples_target[2];
    selection_samples_predictors = estimation_samples_predictors[2];

    # Return output
    return ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, training_samples_target, training_samples_predictors, selection_samples_target, selection_samples_predictors, validation_samples_target, validation_samples_predictors;
end

"""
    get_selection_samples_custom(io::IOStream, subsampling_function::Function, subsample::Float64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return selection samples compatible with tree aggregators based on custom subsampling methods.
"""
function get_selection_samples_custom(io::IOStream, subsampling_function::Function, subsample::Float64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, t0::Int64, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Build macro_finance_data for estimation
    macro_data = macro_vintage[:, 2:end] |> JMatrix{Float64};
    macro_finance_data = [equity_index[1:size(macro_data,1)+1]'; macro_data' missing.*ones(n_series)];

    # Compute d̂ when needed 
    if (subsampling_function === artificial_jackknife) && isnan(subsample)
        n, T = size(@view macro_finance_data[:, 1:end-1]);
        d = optimal_d(n, T);
        subsample = d/(n*T);
    end

    # Construct subsamples
    if subsampling_function === block_jackknife
        custom_data = subsampling_function(macro_finance_data, subsample); # use all block jackknife samples
    else
        custom_data = subsampling_function(macro_finance_data, subsample, max_samples);
    end

    # Memory pre-allocation
    ecm_kalman_settings = Vector{KalmanSettings}();
    ecm_std_diff_data = Vector{FloatMatrix}();
    training_samples_target = Vector{FloatVector}();
    selection_samples_target = Vector{FloatVector}();
    training_samples_predictors = Vector{FloatMatrix}();
    selection_samples_predictors = Vector{FloatMatrix}();

    # Loop over subsamples
    for i in axes(custom_data, 3)
        
        @info("Subsample $(i) out of $(size(custom_data, 3))");
        flush(io);

        # Retrieve data
        current_equity_index = custom_data[1, :, i];
        current_macro_data = custom_data[2:end, 1:end-1, i];

        # Start from first not missing
        first_not_missing = findfirst(.~ismissing.(current_equity_index));
        if max(first_not_missing, sum(ismissing.(current_equity_index))) > 0.6*length(current_equity_index)
            @info("- Skipping subsample $(i): not enough datapoints");
            continue; # i.e., not enough datapoints -> skip subsample
        else
            current_equity_index = current_equity_index[first_not_missing:end];
            current_macro_data = current_macro_data[:, first_not_missing:end];
        end

        # Interpolate `current_equity_index` missings with an expanding one-sided mean
        for t in findall(ismissing.(current_equity_index))
            current_equity_index[t] = mean_skipmissing(current_equity_index[1:t-1]);
        end
        current_equity_index = current_equity_index |> FloatVector;
        
        # Update training and selection samples
        sspace_full, std_diff_data_full, _, estimation_samples_target, estimation_samples_predictors, _, _ = get_selection_samples(current_macro_data, current_equity_index, t0, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);
        push!(ecm_kalman_settings, sspace_full);
        push!(ecm_std_diff_data, std_diff_data_full);
        push!(training_samples_target, estimation_samples_target[1]);
        push!(selection_samples_target, estimation_samples_target[2]);
        push!(training_samples_predictors, estimation_samples_predictors[1]);
        push!(selection_samples_predictors, estimation_samples_predictors[2]);
    end

    # Train and validation samples
    _, _, business_cycle_position, _, _, validation_samples_target, validation_samples_predictors = get_selection_samples(first_data_vintage, equity_index, t0, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);

    # Return output
    return ecm_kalman_settings, ecm_std_diff_data, business_cycle_position, training_samples_target, training_samples_predictors, selection_samples_target, selection_samples_predictors, validation_samples_target, validation_samples_predictors;
end

"""
    get_oos_samples(ecm_kalman_settings::KalmanSettings, ecm_std_diff_data::FloatMatrix, business_cycle_position::Int64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)
    get_oos_samples(ecm_kalman_settings::Vector{KalmanSettings}, ecm_std_diff_data::Vector{FloatMatrix}, business_cycle_position::Int64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

Return the outturn and up-to-date predictors. 
"""
function get_oos_samples(ecm_kalman_settings::KalmanSettings, ecm_std_diff_data::FloatMatrix, business_cycle_position::Int64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Initial settings
    lags = Int64(optimal_hyperparams[1]);

    # Data
    macro_data = macro_vintage[:, 2:end] |> JMatrix{Float64};
    macro_data = permutedims(macro_data);

    # Update KalmanSettings with up-to-date data
    MessyTimeSeriesOptim.update_sspace_data!(ecm_kalman_settings, macro_data./ecm_std_diff_data);

    # Initialise `last_business_cycle_matrix_col`
    last_business_cycle_matrix_col = zeros(2*lags-1);

    # Compute business cycle
    status = kfilter_full_sample(ecm_kalman_settings);
    X_sm, P_sm, X0_sm, P0_sm = ksmoother(ecm_kalman_settings, status, ecm_kalman_settings.Y.T-lags+1);
    for i=1:lags
        last_business_cycle_matrix_col[i] = X_sm[i][business_cycle_position] .* ecm_std_diff_data[1];
    end

    # Compute business cycle future conditions
    X_fc = kforecast(ecm_kalman_settings, status.X_post, lags-1);
    for i=1:lags-1
        last_business_cycle_matrix_col[i+lags] = X_fc[i][business_cycle_position] .* ecm_std_diff_data[1];
    end

    # Equity index
    current_equity_index = permutedims(equity_index[1:ecm_kalman_settings.Y.T]);
    next_equity_index_obs = equity_index[ecm_kalman_settings.Y.T+1];

    # Outturn and predictors
    outturn, predictors = get_target_and_predictors_forecasting(last_business_cycle_matrix_col, current_equity_index, next_equity_index_obs, lags, include_factor_augmentation, use_refined_BC);

    # Return output
    return outturn, predictors;
end

function get_oos_samples(ecm_kalman_settings::Vector{KalmanSettings}, ecm_std_diff_data::Vector{FloatMatrix}, business_cycle_position::Int64, macro_vintage::AbstractDataFrame, equity_index::FloatVector, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, include_factor_augmentation::Bool, use_refined_BC::Bool)

    # Memory pre-allocation
    outturn = 0.0;
    predictors = Vector{FloatVector}();

    # Construct a range of predictors using the estimated vector of kalman settings
    for i in axes(ecm_kalman_settings, 1)
        outturn, current_predictors = get_oos_samples(ecm_kalman_settings[i], ecm_std_diff_data[i], business_cycle_position, macro_vintage, equity_index, optimal_hyperparams, model_args, model_kwargs, include_factor_augmentation, use_refined_BC);
        push!(predictors, current_predictors);
    end

    # Return output
    return outturn, predictors;
end