function get_sbatch_content(equity_index_id::Int64, regression_model::Int64, compute_ep_cycle::Bool, include_factor_augmentation::Bool, include_factor_transformations::Bool)
    
    julia_log_folder_path = ifelse(compute_ep_cycle, "./BC_and_EP_output", "./BC_output");
    sbatch_name = "m$(equity_index_id)_$(regression_model)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations)";
    sbatch_command = "julia finance_forecasts.jl $(equity_index_id) $(regression_model) $(compute_ep_cycle) $(include_factor_augmentation) $(include_factor_transformations) $(julia_log_folder_path)"

    sbatch_content = """
    #!/bin/bash -login
    #!/bin/bash

    #SBATCH --partition=xlarge
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task 1
    #SBATCH --job-name=$(sbatch_name)
    #SBATCH --error=./logs/$(regression_model)/%x_%N_%j.err
    #SBATCH --output=./logs/$(regression_model)/%x_%N_%j.out
    #SBATCH --chdir="../"
    
    module add apps/julia/1.6.7
    $(sbatch_command)""";

    return sbatch_content;
end

# Loop over the equity indices
for equity_index_id=11:20
    for regression_model=1:2
        for compute_ep_cycle=[false; true]
            for (include_factor_augmentation, include_factor_transformations) in [(false, false), (true, false), (true, true)]

                # Get sbatch content
                sbatch_content = get_sbatch_content(equity_index_id, regression_model, compute_ep_cycle, include_factor_augmentation, include_factor_transformations);

                # Setup sbatch
                open("index.sbatch", "w") do io
                    write(io, sbatch_content)
                end;
                
                # Save backup sbatch
                open("./logs/$(regression_model)/scheduler_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).sbatch", "w") do io
                    write(io, sbatch_content)
                end;

                # Run sbatch
                run(`sbatch index.sbatch`);

                # Wait before starting the next iteration
                sleep(2.5);
            end
        end
    end
end
