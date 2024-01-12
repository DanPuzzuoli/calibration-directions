# parse args
while getopts e:c:i:r: flag
do
    case "${flag}" in
        e) e=${OPTARG};; # python environment
        c) c=${OPTARG};; # config file
        i) i=${OPTARG};; # input file
        r) r=${OPTARG};; # results file name
    esac
done

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate $e

# run single job python script
python single_job_script.py --config_file $c --input_file $i --result_file $r
