# Please make sure your environment is well set. Use the following command in powershell to set raw/preprocess/result path, pytorch need to be installed individually

cd <YOUR_ROOT_PATH>
conda activate <CONDA_NAME>
pip install -e .

$env:nnUNet_raw="D:\...\nnUNet_raw"
$env:nnUNet_preprocessed="D:\...\nnUNet_preprocessed"
$env:nnUNet_results="D:\...\nnUNet_results"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"

python convert_to_nnunet.py --src <path_to_your_source_data> --dataset-id <DATASET_ID> --dataset-name <DATASET_NAME>

# run above script will automatically convert sourced dataset into raw data that nnUNetv2 can accept, generate case_to_subtype.csv and split_final.json
# raw data and case_to_subtype.csv will be saved under nnUNet_raw\dataset-id_dataset-name, case_to_subtype.csv and split_final.json will be saved under nnUNet_preprocessed\dataset-name_dataset-id


# after converting data to nnUNet input, preprocess the raw data
# please remember activate your conda env and set raw/preprocess/result environment

$env:nnUNet_raw="\...\nnUNet_raw"
$env:nnUNet_preprocessed="\...\nnUNet_preprocessed"
$env:nnUNet_results="\...\nnUNet_results"

# if you haven't set environment yet, replace them to your raw/preprocess/result path and run in powershell

nnUNetv2_plan_and_preprocess -d <DATASET_ID> -c 2d [3d_lowres 3d_fullres] -pl nnUNetPlannerResEncM

# Use the above CLL command in powershell will preprocess raw data and create a plans.json under nnUNet_preprocessed folder
# If you are not satisfied with plans it generated, you can modifiy it and preprocess again or directly generate plans.json first, then preprocess after you modify it use the following command

nnUNetv2_extract_fingerprint -d <DATASET_ID> --verify_dataset_integrity
nnUNetv2_plan_experiment -d <DATASET_ID> -c 2d [3d_lowres 3d_fullres] -pl nnUNetPlannerResEncM
nnUNetv2_preprocess -d <DATASET_ID> -c 2d [3d_lowres 3d_fullres] -pl <NAME_TO_YOUR_PLAN>
 


# = = = = = = = = = = = = = = = = = = = = = = = Warning = = = = = = = = = = = = = = = = = = = = = = = = =
# Please check plans.json file under nnUNet_preprocessed. Make sure the patch_size and batch_size is applicable to your configuration

nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainer -p <NAME_TO_YOUR_PLAN>

# Use the above CLL command in powershell will preprocess raw data and create a plans.json under nnUNet_preprocessed folder. The name of plans must be consistent with the preprocess output.

# The checkpoint and log will be saved under nnUNet_results

# Use the following command to predict target dataset based on trained model

nnUNetv2_predict -i <path_to_your_input_directory> -o <path_to_your_output_directory> -d 701 -tr nnUNetTrainer -c 2d -f 0 -chk checkpoint_best.pth -p <NAME_TO_YOUR_PLAN>

# This command will output predicted segment files and a predicted case_to_subtype.csv