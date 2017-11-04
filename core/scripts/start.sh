sbatch --nodelist="icsnode$1" --exclusive ./scripts/main.sh && ./scripts/squeue_interactive.sh
