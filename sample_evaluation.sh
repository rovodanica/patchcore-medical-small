# Recommended: run in colab (patchcore_drive.ipynb) 
################################ FastMRI-IXI ################################
### IM224:
datapath=/path/to/data # My setup: ~/patchcore-medical/data/fastmrixi/
root=home/patchcore-medical
loadpath=$root/results/FastmriIXI_Results
modelfolder="IM224_WR50_L2-3_P1_D1024-1024_PS-3_AN-1_S0"
savefolder=evaluated_results/anomaly_test/$modelfolder

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder --save_segmentation_images --save_roc_pr_curves \
patch_core_loader -p $loadpath/$modelfolder/models/fastmrixi_fastmrixi/ --faiss_on_gpu \
dataset --resize 224 --imagesize 224 -d fastmrixi fastmrixi $datapath

### IM320:
datapath=/path/to/data
root=/home/patchcore-medical
loadpath=$root/results/FastmriIXI_Results
modelfolder="BEST_IM320_WR50_L2-3_P1_D1024-1024_PS-5_AN-3_S0_0"
savefolder=evaluated_results/anomaly_test/$modelfolder

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder --save_segmentation_images --save_roc_pr_curves \
patch_core_loader -p $loadpath/$modelfolder/models/fastmrixi_fastmrixi/ --faiss_on_gpu \
dataset --resize 320 --imagesize 320 -d fastmrixi fastmrixi $datapath

# Please see further examples in patchcore_drive.ipynb



################################ MVTEC ################################

datapath=/path/to/data/from/mvtec
loadpath=/path/to/pretrained/patchcore/model

modelfolder=IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
