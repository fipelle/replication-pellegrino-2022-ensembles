using CSV, DataFrames, Dates, MessyTimeSeries;
using FredData: get_data, Fred;

"""
    get_wilshire_selection(fred_tickers::Array{String,1}, observation_start::String, observation_end::String)

Get Wilshire data from FRED.
"""
function get_wilshire_selection(fred_tickers::Array{String,1}, observation_start::String, observation_end::String)

    f = Fred();
    df = DataFrame();

    # Changes
    for unit in ["pch"]
        for (i, ticker) in enumerate(fred_tickers)

            # Download current series
            df_current = get_data(f, ticker, observation_start = observation_start, observation_end = observation_end, frequency = "m", aggregation_method = "eop", units = unit).data::DataFrame;
            df_current = df_current[!, [:date, :value]];

            # Rename `:value`
            rename!(df_current, Dict(:value => "$(fred_tickers[i])_$(unit)"));

            # Add series to `df`
            if size(df) == (0, 0)
                df = df_current;
            else
                df = outerjoin(df, df_current, on = :date);
            end
        end
    end

    # Chronological order
    sort!(df, :date);

    # EOM dates
    df[!, 1] = Dates.lastdayofmonth.(df[!, 1]);

    # Compute squared MoM returns
    df_r2 = df[!, 2:length(fred_tickers)+1].^2;
    for i=1:length(fred_tickers)
        rename!(df_r2, Dict(Symbol("$(fred_tickers[i])_pch") => Symbol("$(fred_tickers[i])_r2")));
    end

    # Append squared MoM returns and sign, then re-order
    df = [df df_r2];

    # Return output
    return df;
end

# Load info from CSV
fred_tickers = ["WILL5000IND",        # Wilshire 5000 Total Market Index
                "WILLLRGCAP",         # Wilshire US Large-Cap Total Market Index
                "WILLLRGCAPVAL",      # Wilshire US Large-Cap Value Total Market Index
                "WILLLRGCAPGR",       # Wilshire US Large-Cap Growth Total Market Index
                "WILLMIDCAP",         # Wilshire US Mid-Cap Total Market Index
                "WILLMIDCAPVAL",      # Wilshire US Mid-Cap Value Total Market Index
                "WILLMIDCAPGR",       # Wilshire US Mid-Cap Growth Total Market Index
                "WILLSMLCAP",         # Wilshire US Small-Cap Total Market Index
                "WILLSMLCAPVAL",      # Wilshire US Small-Cap Value Total Market Index
                "WILLSMLCAPGR"];      # Wilshire US Small-Cap Growth Total Market Index

df = get_wilshire_selection(fred_tickers, "1984-01-01", "2021-01-31");
CSV.write("./wilshire_selection.csv", df);