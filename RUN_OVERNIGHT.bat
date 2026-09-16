@echo off
cd /d "%~dp0"
echo ========================================================
echo   CAPSTONE-MEDSEG: MASTER OVERNIGHT GPU RUNNER
echo   (Launch before leaving lab - runs all experiments and pushes to Git!)
echo ========================================================

echo.
echo ========================================================
echo [1/4] STARTING 40-EPOCH BALANCED EPISODIC TRAINING
echo ========================================================
python -u src\train_episodic.py --epochs 40 --episodes_per_epoch 500 --lr 5e-4 --results_file logs\episodic_zeroshot_results.txt

echo.
echo ========================================================
echo [2/4] RUNNING FEW-SHOT SUPPORT CONTEXT ABLATION (K=1, 2, 4, 8)
echo ========================================================
python -u src\evaluate_shot_ablation.py --checkpoint models\checkpoints\best_fusion_episodic.pt --shots 1 2 4 8 --output_txt logs\shot_ablation_results.txt --output_png logs\shot_ablation_curve.png

echo.
echo ========================================================
echo [3/4] RUNNING MULTI-SEED ROBUSTNESS RUN (SEED 123)
echo ========================================================
python -u src\train_episodic.py --epochs 30 --episodes_per_epoch 500 --lr 5e-4 --num_layers 3 --seed 123 --checkpoint models\checkpoints\best_fusion_episodic_seed123.pt --results_file logs\episodic_zeroshot_results_seed123.txt

echo.
echo ========================================================
echo [4/4] RUNNING MULTI-SEED ROBUSTNESS RUN (SEED 456)
echo ========================================================
python -u src\train_episodic.py --epochs 30 --episodes_per_epoch 500 --lr 5e-4 --num_layers 3 --seed 456 --checkpoint models\checkpoints\best_fusion_episodic_seed456.pt --results_file logs\episodic_zeroshot_results_seed456.txt

echo.
echo ========================================================
echo   ALL EXPERIMENTS COMPLETE! COMMITTING & PUSHING TO GIT
echo ========================================================
git add logs/
git add -f models/checkpoints/*.pt
git commit -m "completed full overnight GPU experiments (40 epochs, ablation, seeds 123/456)"
git pull origin main --rebase
git push origin main
git push origin aadrika-episodic-training

echo.
echo ========================================================
echo   SUCCESS! ALL EXPERIMENTS FINISHED & PUSHED TO GITHUB!
echo ========================================================
pause
