function get_slurm_content(equity_index_id::Int64, subsample::Float64, subsampling_mnemonic::String, subsampling_function_id::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)
    
    subsample_str = replace("$(ifelse(isnan(subsample), "default", subsample))", "."=>"_");
    job_name = "m$(subsampling_function_id)_ind_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str)";
    slurm_logs_folder = "./logs/$(subsampling_mnemonic)"
    julia_specs = "julia finance_forecasts.jl false $(equity_index_id) $(include_factor_augmentation) $(use_refined_BC) $(subsample) $(subsampling_function_id) \"./BC_output\"";

    slurm_content = """
    #!/bin/bash
    
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --partition=m64c512g
    #SBATCH --job-name=$(job_name)
    #SBATCH --error=$(slurm_logs_folder)/%x_%N_%j.err
    #SBATCH --output=$(slurm_logs_folder)/%x_%N_%j.out
    #SBATCH --chdir="../"
    
    module add apps/julia
    $(julia_specs)""";

    return slurm_content;
end

# Loop over the subsampling methods
for subsampling_method in [0,3,4]
    
    # Loop over the equity indices
    for equity_index_id=1:20
        
        subsampling_function_id = copy(subsampling_method);

        # pair bootstrap
        if subsampling_method == 0
            subsample = NaN;
            subsampling_mnemonic = "pair_bootstrap";

        # block bootstrap
        elseif subsampling_method == 1
            subsample = 0.8; # retains 80% of the original data
            subsampling_mnemonic = "block_bootstrap";

        # block jackknife
        elseif subsampling_method == 2
            subsample = 0.2; # retains 80% of the original data
            subsampling_mnemonic = "block_jackknife";

        # artificial jackknife with d̂
        elseif subsampling_method == 3
            subsample = NaN;
            subsampling_mnemonic = "artificial_jackknife";

        # artificial_jackknife with subsample=50%
        elseif subsampling_method == 4
            subsample = 0.5;
            subsampling_mnemonic = "artificial_jackknife";
            subsampling_function_id -= 1;
        end

        # With and without ADL structure / derived BC features
        for (include_factor_augmentation, use_refined_BC) in [(false, false); (true, true)] #(true, false); 

            # Get slurm content
            slurm_content = get_slurm_content(equity_index_id, subsample, subsampling_mnemonic, subsampling_function_id, include_factor_augmentation, use_refined_BC)

            # Setup slurm and backup
            open("index.sl", "w") do io
                write(io, slurm_content)
            end;

            subsample_str = replace("$(ifelse(isnan(subsample), "default", subsample))", "."=>"_");
            
            open("./logs/$(subsampling_mnemonic)/ind_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str).sl", "w") do io
                write(io, slurm_content)
            end;

            # Run sbatch
            run(`sbatch index.sl`);

            # Wait before starting the next iteration
            sleep(2.5);
        end
    end
end