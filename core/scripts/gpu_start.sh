sbatch --nodelist="icsnode$1" --exclusive ./scripts/main_gpu.sh && ./scripts/squeue_interactive.sh
