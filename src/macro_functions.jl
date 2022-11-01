struct DataTransformations
    ticker1::Symbol
    ticker2::Symbol
    new_ticker::Symbol
    operator::Function
end

"""
    transform_vintages_array!(data_vintages::Vector{DataFrame}, release_dates::Vector{Date}, tickers::Vector{String}, tickers_to_transform::Vector{DataTransformations}, tickers_to_deflate::Vector{String}, n_cons_prices::Int64)

Transform the vintages as indicated in `tickers_to_transform`, deflate the series indicated in `tickers_to_deflate` and then remove `PCEPI`.
"""
function transform_vintages_array!(data_vintages::Vector{DataFrame}, release_dates::Vector{Date}, tickers::Vector{String}, tickers_to_transform::Vector{DataTransformations}, tickers_to_deflate::Vector{String}, n_cons_prices::Int64)

    # Compute reference value to the first obs. of the last PCEPI vintage
    first_obs_last_vintage_PCEPI = data_vintages[end][1, :PCEPI]; # this is used for removing base year effects in previous vintages

    # Loop over every data vintage
    for i in axes(data_vintages, 1)

        # Pointer
        vintage = data_vintages[i];

        # Rescale PCE deflator
        if ismissing(vintage[1, :PCEPI])
            error("Base year effect cannot be removed from PCEPI"); # TBC: this line could be generalised further - not needed for the current empirical application
        end
        vintage[!, :PCEPI] ./= vintage[1, :PCEPI];
        vintage[!, :PCEPI] .*= first_obs_last_vintage_PCEPI;

        # Custom transformations
        for transformation in tickers_to_transform

            # Apply transformation and replace entries in `transformation.ticker1`
            vintage[!, transformation.ticker1] = broadcast(transformation.operator, vintage[!, transformation.ticker1], vintage[!, transformation.ticker2]);
            
            # Rename `transformation.ticker1` to `transformation.new_ticker`
            rename!(vintage, (transformation.ticker1 => transformation.new_ticker));

            # Remove `transformation.ticker2`
            select!(vintage, Not(transformation.ticker2));
        end
        
        # Custom real variables
        for ticker in Symbol.(tickers_to_deflate)

            # Deflate
            vintage[!, ticker] ./= vintage[!, :PCEPI];
            vintage[!, ticker] .*= 100;

            # Rename
            rename!(vintage, (ticker => Symbol("R$(ticker)")));
        end

        # Remove PCEPI
        select!(vintage, Not(:PCEPI));

        # Compute YoY% for all prices
        for ticker in Symbol.(tickers[end-n_cons_prices:end])
            vintage[13:end, ticker] .= 100*(vintage[13:end, ticker] ./ vintage[1:end-12, ticker] .- 1);
        end

        # Remove first 12 observations (consequence of taking the YoY% transformation for all prices)
        deleteat!(vintage, 1:12);
    end

    # Remove problematic ALFRED data vintages for PCEPI
    ind_problematic_release = findfirst(release_dates .== Date("2009-08-04")); # PCEPI is incorrectly recorded at that date in ALFRED
    deleteat!(release_dates, ind_problematic_release);
    deleteat!(data_vintages, ind_problematic_release);

    # Update tickers accordingly (`tickers` is always <= `new_tickers` in length)
    empty!(tickers);
    new_tickers = names(data_vintages[end])[2:end];
    for i in axes(new_tickers, 1)
        push!(tickers, new_tickers[i]);
    end
end

"""
    get_dfm_args(compute_ep_cycle::Bool, n_series::Int64, n_cycles::Int64, n_cons_prices::Int64, use_rw_and_i2_trends::Bool)

Return DFM args, kwargs and coordinates_params_rescaling for extracting the business cycle from the macro data of interest.
"""
function get_dfm_args(compute_ep_cycle::Bool, n_series::Int64, n_cycles::Int64, n_cons_prices::Int64, use_rw_and_i2_trends::Bool)

    # DFM cycle setup
    if compute_ep_cycle
        cycles_skeleton = hcat([[zeros(i-1); 1; 2*ones(n_series-i)] for i=1:n_cycles]...)[:,[1, n_cycles]]; # BC and EP
    else
        cycles_skeleton = [1; 2*ones(n_series-1)][:,:]; # BC only
    end
    cycles_free_params = cycles_skeleton .> 1;

    # DFM trend setup
    n_trends = n_series-n_cons_prices+1;
    
    # Setup `trends_skeleton`
    trends_skeleton = Matrix(1.0I, n_trends, n_trends);                          # idiosyncratic trends for all series
    trends_skeleton = [trends_skeleton; zeros(n_cons_prices-1, n_trends-1) 1.0]; # common trend between cpi and core cpi inflation

    # Setup `coordinates_params_rescaling` for trend inflation
    coordinates_params_rescaling = Vector{Vector{Int64}}(undef, n_cons_prices);
    lin_coordinates_params_rescaling = LinearIndices(trends_skeleton)[end-n_cons_prices+1:end, end];
    for i=1:n_cons_prices
        coordinates_params_rescaling[i] = [n_series-n_cons_prices+i, lin_coordinates_params_rescaling[i]];
    end
    trends_free_params = zeros(size(trends_skeleton)) .== 1;

    # DFM drift setup
    drifts_selection = vcat(false, true, true, true, [false for i=1:n_trends-4]...) |> BitArray{1};
    if ~use_rw_and_i2_trends
        drifts_selection[drifts_selection .== 0] .= 1; # use i2 trends only
    end

    # Build `model_args` and `model_kwargs`
    model_args = (trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params);
    model_kwargs = (tol=1e-3, check_quantile=true, verb=false);

    # Return output
    return model_args, model_kwargs, coordinates_params_rescaling;
end

"""
    get_tc_structure(data::Union{FloatMatrix, JMatrix{Float64}}, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, coordinates_params_rescaling::Vector{Vector{Int64}}, existing_estim::Nothing=nothing, existing_std_diff_data::Nothing=nothing)

Get trend-cycle model structure.
"""
function get_tc_structure(data::Union{FloatMatrix, JMatrix{Float64}}, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, coordinates_params_rescaling::Vector{Vector{Int64}}, existing_estim::Nothing=nothing, existing_std_diff_data::Nothing=nothing)

    # Retrieve arguments
    trends_skeleton = model_args[1];
    drifts_selection = model_args[3];

    # Initialise `std_diff_data`
    std_diff_data = zeros(size(data, 1), 1); # the final `, 1` is needed to run MessyTimeSeriesOptim.rescale_estim_params!(...) since it expects a matrix of floats

    # Aggregate data
    for i in axes(data, 1)
        coordinates_trends = findall(view(trends_skeleton, i, :) .!= 0);
        max_order = maximum(view(drifts_selection, coordinates_trends)); # either 1 (smooth trend) or 0 (random walk)
        std_diff_data[i] = compute_scaling_factor(data[i, :], max_order==0);
    end

    zscored_data = data ./ std_diff_data;

    # Build estim
    estim = DFMSettings(zscored_data, Int64(optimal_hyperparams[1]), model_args..., optimal_hyperparams[2:end]...; model_kwargs...);
    MessyTimeSeriesOptim.rescale_estim_params!(coordinates_params_rescaling, estim, std_diff_data);

    # Return output
    return estim, std_diff_data;
end

function get_tc_structure(data::Union{FloatMatrix, JMatrix{Float64}}, optimal_hyperparams::FloatVector, model_args::Tuple, model_kwargs::NamedTuple, coordinates_params_rescaling::Vector{Vector{Int64}}, existing_estim::EstimSettings, existing_std_diff_data::Vector{Float64})
    return existing_estim, existing_std_diff_data;
end
