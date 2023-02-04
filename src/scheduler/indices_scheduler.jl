function get_qsub_content(equity_index_id::Int64, regression_model::Int64, compute_ep_cycle::Bool, include_factor_augmentation::Bool, include_factor_transformations::Bool)
    
    julia_log_folder_path = ifelse(compute_ep_cycle, "./BC_and_EP_output", "./BC_output");
    qsub_log_output = "\$HOME/Documents/replication-pellegrino-2022-ensembles-ridge/src/scheduler/logs/$(regression_model)/\$JOB_NAME.\$JOB_ID.output";
    qsub_name = "m$(equity_index_id)_$(regression_model)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations)";
    qsub_command = "julia finance_forecasts.jl $(equity_index_id) $(regression_model) $(compute_ep_cycle) $(include_factor_augmentation) $(include_factor_transformations) $(julia_log_folder_path)"

    qsub_content = """
    #!/bin/bash -login
    #\$ -wd \$HOME/Documents/replication-pellegrino-2022-ensembles-ridge/src
    #\$ -V
    #\$ -j y
    #\$ -o $(qsub_log_output)
    #\$ -N $(qsub_name)
    #\$ -M f.pellegrino1@lse.ac.uk
    #\$ -m bea
    #\$ -l h_rt=144:0:0
    #\$ -l h_vmem=32G
    #\$ -l h='(vnode01|vnode02|vnode03|vnode06|vnode08|vnode13|vnode16)'
    #\$ -pe smp 1

    module load apps/anaconda3/2022.05/bin
    module load apps/julia/1.6.2
    $(qsub_command)""";
    
    return qsub_content;
end

# Loop over the equity indices
for equity_index_id=11:20
    for regression_model=1:2
        for compute_ep_cycle=[false] #[false; true]
            for (include_factor_augmentation, include_factor_transformations) in [(false, false), (true, true)] #[(false, false), (true, false), (true, true)]

                # Get qsub content
                qsub_content = get_qsub_content(equity_index_id, regression_model, compute_ep_cycle, include_factor_augmentation, include_factor_transformations);

                # Setup qsub
                open("index.qsub", "w") do io
                    write(io, qsub_content)
                end;
                
                # Save backup qsub
                open("./logs/$(regression_model)/scheduler_equity_index_$(equity_index_id)_$(compute_ep_cycle)_$(include_factor_augmentation)_$(include_factor_transformations).qsub", "w") do io
                    write(io, qsub_content)
                end;

                # Run qsub
                run(`qsub index.qsub`);

                # Wait before starting the next iteration
                sleep(2.5);
            end
        end
    end
end
