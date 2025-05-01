cd beauty_and_BMI_analysis

# Activate venv
. .\analyze_beauty.venv\Scripts\Activate.ps1

# Force single-threaded BLAS/OpenMP to reduce peak RAM
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:TORCH_NUM_THREADS="1"

# Run in medium-RAM mode
python evaluate_images.py --medium-ram

deactivate
exit
