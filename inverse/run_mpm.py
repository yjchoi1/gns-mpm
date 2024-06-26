import os
import json
import shutil
import subprocess


def run_mpm(path, mpm_input,
            guess, analysis_dt, analysis_nsteps, output_steps,
            record):

    # %% PREPARE MPM INPUT FILE FOR EACH PHI
    # open initial mpm input with arbitrary phi value
    f = open(os.path.join(path, mpm_input))
    # modify phi in the mpm input file
    mpm_input_guess = json.load(f)
    mpm_input_guess["materials"][0]["friction"] = guess
    mpm_input_guess["materials"][0]["residual_friction"] = guess
    mpm_input_guess["analysis"]["dt"] = analysis_dt
    mpm_input_guess["analysis"]["nsteps"] = analysis_nsteps
    mpm_input_guess["post_processing"]["output_steps"] = output_steps
    f.close()

    # %% PREPARE MPM INPUTS
    # make mpm folder for current phi guess
    mpm_folder = os.path.join(path, f"mpm_phi{guess}")
    os.makedirs(mpm_folder, exist_ok=True)
    # mpm input files should be located in the `path`
    file_paths = [
        f'{path}/mpm_metadata.json',
        f'{path}/mesh.txt',
        f'{path}/particles.txt',
        f'{path}/entity_sets.json',
        f'{path}/initial_config.png'
    ]
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'{file_path} not found. Consider changing current metadata.json to mpm_metadata.json')
    # copy mpm inputs to mpm folder for current phi guess
    shutil.copyfile(f'{path}/mpm_metadata.json', f'{mpm_folder}/metadata.json')
    shutil.copyfile(f'{path}/mesh.txt', f'{mpm_folder}/mesh.txt')
    shutil.copyfile(f'{path}/particles.txt', f'{mpm_folder}/particles.txt')
    shutil.copyfile(f'{path}/entity_sets.json', f'{mpm_folder}/entity_sets.json')
    shutil.copyfile(f'{path}/initial_config.png', f'{mpm_folder}/initial_config.png')
    # make mpm input file for phi guess
    with open(os.path.join(mpm_folder, f"mpm_input.json"), 'w') as f:
        json.dump(mpm_input_guess, f, indent=2)
    f.close()

    # record current mpm input
    record[f"mpm_input_files_phi"] = mpm_input_guess

    # %% RUN MPM
    # write bash file to run mpm
    with open(f'{path}/run_mpm.sh', 'w') as rsh:
        rsh.write(
            f'''\
            #! /bin/bash
            module reset
            /work/08264/baagee/frontera/mpm/build/mpm -i /mpm_input.json -f "{path}/mpm_phi{guess}/"
            ''')

    # run mpm
    with open(f'{path}/run_mpm.sh', 'rb') as bashfile:
        script = bashfile.read()
    with open(f"{mpm_folder}/mpm_out.txt", 'w') as outfile:
        rc = subprocess.call(script, shell=True, stdout=outfile)

    return record
