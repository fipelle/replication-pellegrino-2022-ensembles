using Dates;
using FredData;
using DataFrames;
using Infiltrator;

"""
    get_fred_vintages(tickers::Array{String,1}, frequencies::Array{String,1}, fred_options::Dict{Symbol, String}, rm_base_year_effect::BitArray{1})

Build a DataFrame of real-time vintages from FRED.

# Arguments
- `tickers`: Array of strings with the FRED tickers for the series of interest.
- `frequencies`: Corresponding frequencies of interest.
- `fred_options`: kwargs of interest for `get_data(...)` in the form of a dictionary.
- `rm_base_year_effect`: BitArray{1} with the same lenght of `tickers`. The `true` entries correspond to the series for which the effect of a change in the base year across vintages must be removed.
"""
function get_fred_vintages(tickers::Array{String,1}, frequencies::Array{String,1}, fred_options::Dict{Symbol, String}, rm_base_year_effect::BitArray{1})

    if length(tickers) != length(rm_base_year_effect)
        error("`tickers` and `rm_base_year_effect` must have the same dimension!");
    end

    # Load API key
    f = Fred();

    # Memory pre-allocation for final output
    df = DataFrame();

    # Loop over tickers
    for i in eachindex(tickers)

        # Slow down the requests to avoid unexpected crashes
        if i > 1
            sleep(rand(1:0.5:5));
        end

        # Current ticker and frequency
        ticker = tickers[i];
        frequency = frequencies[i];

        # Get data from FRED
        df_ticker = get_data(f, ticker, frequency=frequency; fred_options...).data;
        # TBD: do we want to have a try-catch within a for loop to try a few times to get the data from fred?

        # Keep columns compatible with DataFrames constructed by `get_local_vintages(...)`
        df_ticker = df_ticker[!, [:realtime_start, :date, :value]];

        # Set reference dates to eop in order to correctly align mixed frequency vintages
        if frequency == "m"
            df_ticker[!,:date] = Dates.lastdayofmonth.(df_ticker[!,:date]);

        elseif frequency == "q"
            df_ticker[!,:date] = Dates.lastdayofquarter.(df_ticker[!,:date]);

        elseif frequency == "sa"
            df_ticker[!,:date] = Dates.lastdayofquarter.(df_ticker[!,:date] .+ Month(3));

        elseif frequency == "a"
            df_ticker[!,:date] = Dates.lastdayofyear.(df_ticker[!,:date]);
        end

        # Convert NaNs to missings
        df_ticker[!,:value] = df_ticker[!,:value] |> Array{Union{Missing, Float64}};
        df_ticker[isnan.(df_ticker[!,:value]), :value] .= missing;

        # Remove base year effect
        if rm_base_year_effect[i]

            # Find rows for the entries corresponding to the first reference date
            ind_first_reference_date = findall(df_ticker[!,:date] .== minimum(df_ticker[!,:date]));
            
            #=
            Check whether there is a base effect to remove in the first place.
            - Note that right now the code adjusts the data if there are multiple entries for the first reference date.
              This implies that the code should be generalised (TBD) to handle peculiar cases in which the first reference
              date is excluded at source from all vintages except one. 
            =#
            
            if length(ind_first_reference_date) > 1
            
                # Compute corresponding SubDataFrame
                sub_df_ticker = @view df_ticker[ind_first_reference_date, :];

                if sum(ismissing.(sub_df_ticker[!,:value])) > 0
                    error("The effect of the base year cannot be removed in $(ticker) with the current approach. Try changing observation_start to a more recent date or set rm_base_year_effect to false for $(ticker).");
                end

                # Compute adjustment factors (i.e., use the latest base year)
                adjustment_factors = sub_df_ticker[end,:value] ./ sub_df_ticker[!,:value];
                
                # Loop over every release date
                for release_date in unique(df_ticker[!,:realtime_start])

                    # Current adjustment factor
                    adjustment_factor = adjustment_factors[findlast(sub_df_ticker[!,:realtime_start] .<= release_date)];
                    
                    # Apply adjustment
                    df_ticker[df_ticker[!,:realtime_start] .== release_date, :value] .*= adjustment_factor;
                end
            end
        end

        # Rename: value -> ticker, realtime_start -> release_dates, date -> reference_dates
        rename!(df_ticker, (:value => Symbol(ticker)));
        rename!(df_ticker, (:realtime_start => :release_dates));
        rename!(df_ticker, (:date => :reference_dates));

        # Add `df_ticker` data to `df`
        if i == 1
            df = df_ticker;
        else
            df = outerjoin(df, df_ticker, on=[:reference_dates, :release_dates]);
        end
    end

    # Chronological order of releases
    sort!(df, :release_dates);

    # Return output
    return df;
end

"""
    get_local_vintages(tickers::Array{String,1}, data::Array{Union{Missing, Float64}}, reference_dates::Array{Date,1}, release_dates::Array{Union{Missing, Date}})

Build a DataFrame of unrevised vintages from external data sources.

# Arguments
- `tickers`: Array of strings with the FRED tickers for the series of interest.
- `data`: Matrix of (potentially incomplete) unrevised data.
- `reference_dates`: Array of reference dates.
- `release_dates`: Matrix of (potentially incomplete) release dates. This matrix is number of reference dates x number of series dimensional.

# Notes
`reference_dates` and `release_dates` must be end of period dates.
"""
function get_local_vintages(tickers::Array{String,1}, data::Array{Union{Missing, Float64}}, reference_dates::Array{Date,1}, release_dates::Array{Union{Missing, Date}})

    # Number of releases
    n_releases = length(reference_dates);
    if n_releases != size(release_dates,1);
        error("The reference dates are not correctly aligned with the releases");
    end

    @warn("`get_local_vintages(...)` is experimental!");

    # Memory pre-allocation for final output
    df = DataFrame();

    # Loop over tickers
    for i in eachindex(tickers)

        # Incomplete release dates for current series
        release_dates_ticker = release_dates[:,i];
        
        # Fill missing release dates
        for j=n_releases:-1:1
            if ismissing(release_dates_ticker[j]) && (j < n_releases)
                release_dates_ticker[j] = release_dates_ticker[j+1];
            end
        end

        # Observed datapoints for which there is a release date
        ind_observed_ticker = .~ismissing.(data[:,i]) .& .~ismissing.(release_dates_ticker);

        # Generate `df_ticker`
        df_ticker = DataFrame(:reference_dates => reference_dates[ind_observed_ticker], 
                              :release_dates => release_dates_ticker[ind_observed_ticker], 
                              Symbol(ticker) => data[ind_observed_ticker, i]);

        # Add `df_ticker` data to `df`
        if i == 1
            df = df_ticker;
        else
            df = outerjoin(df, df_ticker, on=[:reference_dates, :release_dates]);
        end
    end

    # Chronological order of releases
    sort!(df, :release_dates);

    # Return output
    return df;
end

"""
    get_vintages_array(df::DataFrame, sampling_frequency::String)

Generate an array of arrays representing an array of data vintages from `df`.

With mixed-frequency datasets `sampling_frequency` denotes the highest data sampling frequency. `sampling_frequency` uses the same notation employed by FRED for the data frequency WITHOUT period descriptions.
"""
function get_vintages_array(df::DataFrame, sampling_frequency::String)
    
    # Unique release and reference dates
    unique_release_dates = sort(unique(df[!,:release_dates]));
    unique_reference_dates = sort(unique(df[!,:reference_dates]));

    # Problem size
    n_releases = length(unique_release_dates);

    # Memory pre-allocation for convenient DataFrame
    df_vintage = DataFrame();

    # Memory pre-allocation for final output
    data_vintages = Array{DataFrame,1}(undef, n_releases);

    # Compute `full_reference_dates`
    start_sample = minimum(unique_reference_dates);
    end_sample = maximum(unique_reference_dates);

    if sampling_frequency == "d"
        full_reference_dates = collect(start_sample:Dates.Day(1):end_sample);

    elseif sampling_frequency == "w"
        full_reference_dates = collect(start_sample:Dates.Week(1):end_sample);

    elseif sampling_frequency == "bw"
        full_reference_dates = collect(start_sample:Dates.Week(2):end_sample);

    elseif sampling_frequency == "m"
        full_reference_dates = collect(start_sample:Dates.Month(1):end_sample);

    elseif sampling_frequency == "q"
        full_reference_dates = collect(start_sample:Dates.Month(3):end_sample);

    elseif sampling_frequency == "sa"
        full_reference_dates = collect(start_sample:Dates.Month(6):end_sample);

    elseif sampling_frequency == "a"
        full_reference_dates = collect(start_sample:Dates.Year(1):end_sample);
    else
        error("sampling_frequency is not correctly specified!");
    end

    # Loop over the releases
    for i=1:n_releases

        # Current release (only new observations and revisions)
        ind_release = findall(df[!,:release_dates] .== unique_release_dates[i]);
        df_release = df[ind_release, 2:end];
        sort!(df_release, :reference_dates);

        # Initial data vintage
        if i == 1            
            df_vintage = df_release;
        
        # Following vintages
        else
            
            #=
            Revisions and new observations
            - `df_revisions` includes revisions to points in time observed in previous vintages. This broad definition includes data revisions and advanced measurements (i.e, new observations) referring to previously observed points in time.
            - `df_new_observations` includes observations for points in time not observed in previous vintages.
            =#
            
            df_revisions = semijoin(df_release, df_vintage, on=:reference_dates);
            df_new_observations = antijoin(df_release, df_vintage, on=:reference_dates);

            @infiltrate
            
            # Update `df_vintage` inplace with the data revisions (if any) looping over each reference period in `df_revisions`
            for df_revision in eachrow(df_revisions)

                @infiltrate

                # Alignment point between `df_revision` and `df_vintage`
                row_to_revise = findfirst(view(df_vintage, !, :reference_dates) .== df_revision[:reference_dates]);

                @infiltrate

                # Revise each observed entry/ticker in `df_revision`
                for ticker in eachindex(df_revision)
                    
                    @infiltrate

                    if (ticker != :reference_dates) && ~ismissing(df_revision[ticker])
                        df_vintage[row_to_revise, ticker] = df_revision[ticker];
                    end
                end

                @infiltrate
            end

            @infiltrate

            # Add new observations
            append!(df_vintage, df_new_observations);

            @infiltrate

            # Add missing rows (if necessary)
            full_reference_dates_release = full_reference_dates[full_reference_dates .<= maximum(df_vintage[!,:reference_dates])];
            if df_vintage[!,:reference_dates] != full_reference_dates_release
                df_vintage = outerjoin(DataFrame(:reference_dates=>full_reference_dates_release), df_vintage, on=:reference_dates);
            end

            @infiltrate

            # Sort df_vintage
            sort!(df_vintage, :reference_dates);
        end

        # Store current vintage
        data_vintages[i] = copy(df_vintage);
    end

    # Return output
    return data_vintages, unique_release_dates;
end

"""
    get_financial_vintages(tickers::Array{String,1}, fred_options::Dict{Symbol, String}, actual_realtime_start::Date)

Download financial spot prices.
"""
function get_financial_vintages(tickers::Array{String,1}, fred_options::Dict{Symbol, String}, actual_realtime_start::Date)

    # Memory pre-allocation for final output
    df_financial = DataFrame();

    # Loop over tickers
    for (i, ticker) in enumerate(tickers)
        df_ticker = get_fred_vintages(["$ticker"], ["m"], fred_options, [0.0] .== 1);

        # All data in `tickers` are daily variables available at the end of the reference day (the release calendar on ALFRED is incomplete and misleading)
        df_ticker[!, :release_dates] = copy(df_ticker[!,:reference_dates]);
        df_ticker[df_ticker[!,:release_dates] .<= actual_realtime_start, :release_dates] .= actual_realtime_start;

        if i == 1
            df_financial = df_ticker;
        else
            df_financial = outerjoin(df_financial, df_ticker, on=[:reference_dates, :release_dates]);
        end
    end

    # Chronological order of releases
    sort!(df_financial, :release_dates);

    # Return financial data
    return df_financial;
end