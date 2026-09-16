@echo off
cd /d "%~dp0"
echo ===================================================
echo   PUSHING ALL EXPERIMENT RESULTS AND CHECKPOINTS
echo ===================================================

echo 1. Adding logs directory...
git add logs/

echo 2. Force-adding .pt model checkpoints (bypassing .gitignore)...
git add -f models/checkpoints/*.pt

echo 3. Committing changes...
git commit -m "completed 30-epoch and 40-epoch training results and checkpoints"

echo 4. Pulling remote updates with rebase...
git pull origin main --rebase

echo 5. Pushing to GitHub main branch...
git push origin main

echo ===================================================
echo   ALL DONE! Check your GitHub repo to verify.
echo ===================================================
pause
