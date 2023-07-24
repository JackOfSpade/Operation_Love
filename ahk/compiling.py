import subprocess

def compile_code(script_name):
    # The command to compile the script
    command = f"pyinstaller --onefile {script_name}"

    # Use subprocess to run the command
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    # Replace 'your_script.py' with the name of your actual script
    compile_code("main.py")