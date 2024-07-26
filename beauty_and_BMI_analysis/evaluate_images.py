import subprocess
import os

def run_in_env(command, env_path):
    activate_script = os.path.join(env_path, 'Scripts', 'Activate.ps1')  # Using PowerShell script for activation
    full_command = f'powershell.exe -Command "{activate_script}; {command}"'
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

if __name__ == "__main__":
    # Get the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to virtual environments relative to the script's directory
    analyze_beauty_env_path = os.path.join(base_dir, 'analyze_beauty_env')
    analyze_bmi_env_path = os.path.join(base_dir, 'analyze_bmi_env')

    # Commands to run scripts in respective environments using relative paths
    analyze_beauty_script_command = f"python {os.path.join(base_dir, 'analyze_beauty.py')}"
    analyze_bmi_script_command = f"python {os.path.join(base_dir, 'analyze_bmi.py')}"

    # Run the model_script.py in MediaPipe environment
    output, error = run_in_env(analyze_beauty_script_command, analyze_beauty_env_path)
    print("analyze_beauty output:", output)
    print("analyze_beauty error:", error)

    # Run the vit_script.py in TensorFlow environment
    output, error = run_in_env(analyze_bmi_script_command, analyze_bmi_env_path)
    print("analyze_bmi output:", output)
    print("analyze_bmi error:", error)
