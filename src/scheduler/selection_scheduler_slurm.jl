function get_slurm_content(compute_ep_cycle::Bool, log_folder_path::String)
    
    job_name = "select_$(compute_ep_cycle)";
    slurm_logs_folder = "./logs"
    julia_specs = "julia -p32 macro_selection.jl $(compute_ep_cycle) 4 $(log_folder_path)";

    slurm_content = """
    #!/bin/bash
    
    #SBATCH --nodes=1
    #SBATCH --ntasks=32
    #SBATCH --partition=m64c512g
    #SBATCH --job-name=$(job_name)
    #SBATCH --error=$(slurm_logs_folder)/%x_%N_%j.err
    #SBATCH --output=$(slurm_logs_folder)/%x_%N_%j.out
    #SBATCH --chdir="../"
    
    module add apps/julia
    $(julia_specs)""";

    return slurm_content;
end

for (compute_ep_cycle, log_folder_path) in [(false, "./BC_output"), (true, "./BC_and_EP_output")]

        # Get slurm content
        slurm_content = get_slurm_content(compute_ep_cycle, log_folder_path);

        # Setup slurm and backup
        open("index.sl", "w") do io
            write(io, slurm_content)
        end;
        
        open("./logs/selection_$(compute_ep_cycle).sl", "w") do io
            write(io, slurm_content)
        end;

        # Run sbatch
        run(`sbatch index.sl`);

        # Wait before starting the next iteration
        sleep(5.0);
    end
end