@echo off
cd /d "%~dp0"
echo ========================================================
echo   CAPSTONE-MEDSEG: NNU-NET SPECIALIST PIPELINE RUNNER
echo ========================================================

echo.
echo 1. Setting up nnU-Net Environment Variables...
set nnUNet_raw=%CD%\data\nnunet_raw
set nnUNet_preprocessed=%CD%\data\nnunet_preprocessed
set nnUNet_results=%CD%\data\nnunet_results

echo   nnUNet_raw          = %nnUNet_raw%
echo   nnUNet_preprocessed = %nnUNet_preprocessed%
echo   nnUNet_results      = %nnUNet_results%

echo.
echo 2. Converting raw NIfTI datasets into nnU-Net v2 format...
python src\nnunet_pipeline.py --dataset all

echo.
echo ========================================================
echo [TASK 1/3] RUNNING NNU-NET FOR HEART (DATASET 002)
echo Estimated time: ~1.5 to 2 hours
echo ========================================================
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 2d 0 -tr nnUNetTrainer_250epochs

echo.
echo ========================================================
echo [TASK 2/3] RUNNING NNU-NET FOR LIVER (DATASET 003)
echo Estimated time: ~45 minutes
echo ========================================================
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
nnUNetv2_train 3 2d 0 -tr nnUNetTrainer_250epochs

echo.
echo ========================================================
echo [TASK 3/3] RUNNING NNU-NET FOR BRAIN TUMOUR (DATASET 001)
echo Estimated time: ~2.5 to 3.5 hours
echo ========================================================
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_250epochs

echo.
echo ========================================================
echo   ALL NNU-NET RUNS COMPLETE! COMMITTING RESULTS TO GIT
echo ========================================================
git add logs/
git add data\nnunet_results\*\summary.json
git commit -m "completed nnU-Net specialist baselines for Heart, Liver, and BrainTumour"
git pull origin main --rebase
git push origin main
git push origin aadrika-episodic-training

echo ========================================================
echo   SUCCESS! ALL NNU-NET BENCHMARKS COMPLETED & PUSHED!
echo ========================================================
pause
