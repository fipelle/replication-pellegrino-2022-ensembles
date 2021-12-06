"""
    get_dfm_args(compute_ep_cycle::Bool, n_series::Int64, n_cycles::Int64, n_cons_prices::Int64)

Return DFM args, kwargs and coordinates_params_rescaling for extracting the business cycle from the macro data of interest.
"""
function get_dfm_args(compute_ep_cycle::Bool, n_series::Int64, n_cycles::Int64, n_cons_prices::Int64)

    # DFM cycle setup
    if compute_ep_cycle
        cycles_skeleton = hcat([[zeros(i-1); 1; 2*ones(n_series-i)] for i=1:n_cycles]...)[:,[1, n_cycles]]; # BC and EP
    else
        cycles_skeleton = [1; 2*ones(n_series-1)][:,:]; # BC only
    end
    cycles_free_params = cycles_skeleton .> 1;

    # DFM trend setup
    n_trends = n_series-n_cons_prices+1;
    trends_skeleton = Matrix(1.0I, n_trends, n_trends);
    trends_skeleton = [trends_skeleton; zeros(n_cons_prices-1, n_trends-1) 1.0];
    coordinates_params_rescaling = Vector{Vector{Int64}}(undef, n_cons_prices);
    lin_coordinates_params_rescaling = LinearIndices(trends_skeleton)[end-n_cons_prices+1:end, end];
    for i=1:n_cons_prices
        coordinates_params_rescaling[i] = [n_series-n_cons_prices+i, lin_coordinates_params_rescaling[i]];
    end
    trends_free_params = zeros(size(trends_skeleton)) .== 1;

    # DFM drift setup
    drifts_selection = vcat(false, true, true, true, [false for i=1:n_trends-4]...) |> BitArray{1};

    # Build `model_args` and `model_kwargs`
    model_args = (trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params);
    model_kwargs = (tol=1e-3, check_quantile=true, verb=false);

    # Return output
    return model_args, model_kwargs, coordinates_params_rescaling;
end

"""
    get_tc_structure(data::Union{FloatMatrix, JMatrix{Float64}}, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, coordinates_params_rescaling::Vector{Vector{Int64}})

Get trend-cycle model structure.
"""
function get_tc_structure(data::Union{FloatMatrix, JMatrix{Float64}}, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, coordinates_params_rescaling::Vector{Vector{Int64}})

    # Standardise data
    std_diff_data = std_skipmissing(diff(data, dims=2));
    zscored_data = data ./ std_diff_data;

    # Build estim
    estim = DFMSettings(zscored_data, Int64(optimal_hyperparams[1]), model_args..., optimal_hyperparams[2:end]...; model_kwargs...);
    MessyTimeSeriesOptim.rescale_estim_params!(coordinates_params_rescaling, estim, std_diff_data);

    # Return output
    return estim, std_diff_data;
end