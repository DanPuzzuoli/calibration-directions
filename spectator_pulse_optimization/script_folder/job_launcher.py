######################################################
# Python script for launching multiple jobs on the CCC
######################################################

import os
import pickle
import yaml

# current folder
abs_path = os.path.abspath('.')

job_runner_file = f"{abs_path}/single_job_runner.sh"
config_file = f"{abs_path}/job_config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

log_folder = config["log_folder"]
log_file_template = f"{abs_path}/{log_folder}/log"

result_folder = config["result_folder"]
result_file_template = f"{abs_path}/{result_folder}/opt_state"

initial_guesses_folder = config["initial_guesses_folder"]
initial_guess_file_template = f"{abs_path}/{initial_guesses_folder}/opt_state"


conda_environment = config["conda_environment"]

import subprocess

if __name__ == "__main__":

    num_guesses = config["num_guesses"]

    # check for pre-existing initial guesses
    file_names = os.listdir("initial_guesses/")
    prefix = 'opt_state'
    suffix = '.pkl'
    existing_initial_guesses_idx = []
    for file_name in file_names:
        if file_name[:len(prefix)] == prefix and file_name[-len(suffix):] == suffix:
            idx_marker = int(file_name[len(prefix):-len(suffix)])
            existing_initial_guesses_idx.append(idx_marker)
    
    # fill out initial guess files
    num_init_files = len(existing_initial_guesses_idx)
    idx = 0
    new_idx_list = []
    while num_init_files < num_guesses:
        if idx not in existing_initial_guesses_idx:

            # create random initial guess
            opt_state = {
                "params": None
            }

            with open(f'initial_guesses/{prefix}{idx}{suffix}', 'wb') as f:
                pickle.dump(opt_state, f)
        
            num_init_files += 1
            new_idx_list.append(idx)
        idx += 1
    
    initial_guess_indices = new_idx_list + existing_initial_guesses_idx


    for run_idx in initial_guess_indices:
        log_file = f"{log_file_template}{run_idx}.txt"
        input_file = f"{initial_guess_file_template}{run_idx}.pkl"
        result_file = f"{result_file_template}{run_idx}.pkl"

        if os.path.isfile(result_file):
            print(f"Result file {result_file} already exists, skipping job.")
        else:
            bash_command = (
                f"jbsub "
                f"-e {log_file} " # log file
                f"-cores 1 -q x86_24h -mem 5G " # machine requested
                f"sh {job_runner_file} -e {conda_environment} -c {config_file} -i {input_file} -r {result_file}" # shell script with config file passed as
            )
            subprocess.run(bash_command, shell=True)
