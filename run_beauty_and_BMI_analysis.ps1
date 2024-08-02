# Activate the virtual environment
. .\beauty_and_BMI_analysis\analyze_beauty.venv\Scripts\Activate.ps1

# Run the Python script
python .\beauty_and_BMI_analysis\evaluate_images.py

# Deactivate the virtual environment (optional)
deactivate


Read-Host -Prompt "Press Enter to exit"

# Exit the PowerShell script
exit

