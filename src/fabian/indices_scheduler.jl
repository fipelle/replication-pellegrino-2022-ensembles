function get_qsub_content(equity_index_id::Int64, subsample::Float64, subsampling_mnemonic::String, subsampling_function_id::Int64, include_factor_augmentation::Bool, use_refined_BC::Bool)
    
    subsample_str = replace("$(ifelse(isnan(subsample), "default", subsample))", "."=>"_");

    qsub_log_output = "\$HOME/Documents/replication-pellegrino-2022-ensembles/src/fabian/logs/$(subsampling_mnemonic)/\$JOB_NAME.\$JOB_ID.output";
    qsub_name = "m$(subsampling_function_id)_ind_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str)";
    qsub_command = "julia finance_forecasts.jl false $(equity_index_id) $(include_factor_augmentation) $(use_refined_BC) $(subsample) $(subsampling_function_id) \"./BC_output\"";

    qsub_content = """
    #!/bin/bash -login
    #\$ -wd \$HOME/Documents/replication-pellegrino-2022-ensembles/src
    #\$ -V
    #\$ -j y
    #\$ -o $(qsub_log_output)
    #\$ -N $(qsub_name)
    #\$ -M f.pellegrino1@lse.ac.uk
    #\$ -m bea
    #\$ -l h_rt=144:0:0
    #\$ -l h_vmem=64G
    #\$ -pe smp 1
    #\$ -l h='(vnode01|vnode02|vnode03|vnode06|vnode08|vnode13)'

    module load apps/julia/1.6.2
    $(qsub_command)""";

    return qsub_content;
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
        for (include_factor_augmentation, use_refined_BC) in [(false, false); (true, true)]

            # Get qsub content
            qsub_content = get_qsub_content(equity_index_id, subsample, subsampling_mnemonic, subsampling_function_id, include_factor_augmentation, use_refined_BC)

            # Setup qsub and backup
            open("index.qsub", "w") do io
                write(io, qsub_content)
            end;

            subsample_str = replace("$(ifelse(isnan(subsample), "default", subsample))", "."=>"_");
            
            open("./logs/$(subsampling_mnemonic)/ind_$(equity_index_id)_$(include_factor_augmentation)_$(use_refined_BC)_$(subsample_str).qsub", "w") do io
                write(io, qsub_content)
            end;

            # Run qsub
            run(`qsub index.qsub`);

            # Wait before starting the next iteration
            sleep(2.5);
        end
    end
end