# Libraries
using DataFrames, Dates, FileIO, JLD;
using Contour, DecisionTree, Random, StableRNGs;
using PGFPlotsX, LaTeXStrings;
include("./../src/MessyTimeSeriesOptim.jl");
using Main.MessyTimeSeriesOptim;

function learn_grid_structure(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, rng::AbstractRNG=StableRNG(1), n_trees::Int64=1000)

    # Initial settings
    n = length(x);
    trees = Vector{DecisionTreeRegressor}(undef, n_trees);

    # Bagging
    for i=1:n_trees
        trees[i] = DecisionTreeRegressor(min_samples_leaf=25, rng=rng);
        current_selection = rand(rng, 1:n, n);
        current_x = x[current_selection];
        current_y = y[current_selection];
        current_z = z[current_selection];
        DecisionTree.fit!(trees[i], [current_x current_y], current_z);
    end

    return trees;
end

function interpolate_z(x::Float64, y::Float64, trees::Vector{DecisionTreeRegressor})

    # Initial settings
    output = 0.0;
    n_trees = length(trees);
    covariates = [x y];

    # Bagging prediction
    for tree in trees
        output += DecisionTree.predict(tree, covariates)[1];
    end
    output /= n_trees;

    return output;
end

# Plot: grid
push!(PGFPlotsX.CUSTOM_PREAMBLE, 
        raw"\usepgfplotslibrary{colorbrewer}",
        raw"\usepgfplotslibrary{colormaps}",
        raw"\usepgfplotslibrary{patchplots}",
        raw"\pgfplotsset
            {   
                tick label style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                label style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                legend style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
                title style = {font = {\fontsize{12 pt}{12 pt}\selectfont}},
            }"
)

# GroupPlot axs per row
gpr = Array{Any,1}(undef, 2);
for i=1:2
    gpr[i] = Array{Any,1}(undef, 2);
end

axs_titles = ["BC", "BC and EP"];

# Interpolated grid
x_grid = range(0.001, stop=1, length=100);
y_grid = range(-5.0, stop=10, length=100);

# Loop over each model
for i in 1:2

    #=
    ----------------------------------------------------------------------------------------------------
    Load input data and interpolate grid
    ----------------------------------------------------------------------------------------------------
    =#

    # Raw output
    if i == 1
        f = load("./BC_output/err_type_4.jld");
    else
        f = load("./BC_and_EP_output/err_type_4.jld");
    end

    candidates = f["candidates"];
    errors = f["errors"];
    
    # Explored grid
    x = candidates[3,:];
    y = log.(candidates[2,:] .* (candidates[4,:].^(candidates[1,:].-1)));
    z = errors;

    # Learn structure of the grid and interpolate z
    trees = learn_grid_structure(x, y, z);
    z_grid = [interpolate_z(xx, yy, trees) for xx in x_grid, yy in y_grid];

    # Meta source: minimum meta_source* is equal to 0
    meta_source = z_grid;
    meta_source_scatter = z;

    # Loop over subplot rows
    for j=1:2

        #=
        ----------------------------------------------------------------------------------------------------
        Current subplot settings
        ----------------------------------------------------------------------------------------------------
        =#

        if j == 1
            x_dir_settings = "normal";
            view_angle = (45, 15);
        
        elseif j == 2
            x_dir_settings = "reverse";
            view_angle = (90, 90);
        end

        subtitle = ifelse(j==1, axs_titles[i], "");
        axis_on_top = ifelse(j==2, "true", "false");
        colormap_settings = ifelse((j==1) & (i==2), "true", "false");
        z_label_name = ifelse(i==1, L"Loss", "");

        #=
        ----------------------------------------------------------------------------------------------------
        Generate subplot
        ----------------------------------------------------------------------------------------------------
        =#
        
        if j==1
            subplot_p1 = @pgf Plot3({surf, point_meta="explicit"}, Coordinates(x_grid, y_grid, z_grid, meta=meta_source));
        elseif j==2
            subplot_p1 = @pgf Plot3({contour_filled="{number=30, labels={false}}", shader="interp", "patch type"="bilinear", point_meta="explicit"}, Coordinates(x_grid, y_grid, z_grid, meta=meta_source));
        end
        
        if j == 1
            subplot_p2 = @pgf Plot3({only_marks, scatter, mark_size="1pt", point_meta="explicit"}, Coordinates(x, y, 1050 .* ones(length(x)), meta=meta_source_scatter));
        elseif j == 2
            subplot_p2 = @pgf Plot3({only_marks, scatter, mark_size="0.75pt", point_meta="explicit", opacity="50"}, Coordinates(x, y, z, meta=meta_source_scatter));
        end

        gpr[j][i] = @pgf Axis(
        {
            "axis on top"=axis_on_top,
            height = "10cm",
            width = "10cm",
            grid = "major",
            grid_style = "{darkgray, very thin}",
            tick_style = "{darkgray, very thin}",
            "colormap/jet",
            "colorbar"=colormap_settings,
            "colorbar style"=raw"
            {
                height = 2*\pgfkeysvalueof{/pgfplots/parent axis height} + \pgfkeysvalueof{/pgfplots/group/vertical sep},
                ytick={1070,1090,...,1150},
                yticklabel style=
                {
                    text width=2.5em,
                    align=right,
                    /pgf/number format/.cd,
                        fixed,
                        fixed zerofill
                }
            }",
            "colorbar/width" = "0.8cm",
            point_meta_min = "1070",
            point_meta_max = "1150",
            title = subtitle,
            view = view_angle,
            xlabel = L"\alpha",
            ylabel = L"\ln(\lambda\,\beta^{11})",
            zlabel = z_label_name,
            zmin = "1050",
            zmax = "1150",
            x_dir=x_dir_settings,
            xtick_distance = "0.25",  # set the distance betwen each tick
            ytick_distance = "5",     # set the distance betwen each tick
            ztick_distance = "25",    # set the distance betwen each tick
        },
            subplot_p1,
            subplot_p2,
        );
    end
end

fig = @pgf TikzPicture(GroupPlot(
    { group_style = {group_size="2 by 2", vertical_sep="50pt", horizontal_sep="60pt"},
      height="150pt",
      width="225pt"
    },
    gpr[1]..., gpr[2]...,)
);

pgfsave("./img/bc_hyperparams_surface.pdf", fig);