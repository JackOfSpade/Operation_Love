import subprocess
import os


def activate_and_run_pyinstaller(venv_path, script_name, additional_data=None):
    # Command to activate virtual environment. This varies based on OS.
    if os.name == "posix":  # UNIX-like OS
        activate_command = f"source {venv_path}/bin/activate"
    else:  # Windows
        activate_command = f"{venv_path}\\Scripts\\activate.bat"

    # Command to run PyInstaller after activation
    pyinstaller_command = f"pyinstaller --onefile {script_name}"

    # If additional data is provided, append the --add-data flag
    if additional_data:
        pyinstaller_command += f" --add-data={additional_data}"

    # Combine commands
    full_command = f"{activate_command} && {pyinstaller_command}"

    # Execute commands
    subprocess.run(full_command, shell=True)


def create_executable(script_name, additional_data=None):
    try:
        # Determine the venv path based on the script's directory
        script_directory = os.path.dirname(os.path.abspath(script_name))
        venv_path = os.path.join(script_directory, "venv")

        # Activate the correct environment and run PyInstaller
        activate_and_run_pyinstaller(venv_path, script_name, additional_data)

        print(f"Executable created for {script_name} in the 'dist' folder.")
    except Exception as e:
        print(f"An error occurred while creating the executable: {e}")


if __name__ == "__main__":
    create_executable("..\\bulk_image_ocr\\bulk_image_ocr.py")
    create_executable("..\\chatgpt\\chatgpt.py")