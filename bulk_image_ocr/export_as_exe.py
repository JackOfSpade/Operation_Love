from PyInstaller.__main__ import run


def create_executable(script_name):
    try:
        # PyInstaller expects its arguments as if they were passed on the
        # command line, hence the list format. '--onefile' indicates to
        # package as a single executable file.
        opts = ['--onefile', script_name]

        # Run the PyInstaller
        run(opts)

        print(f"Executable created for {script_name} in the 'dist' folder.")
    except Exception as e:
        print(f"An error occurred while creating the executable: {e}")


if __name__ == "__main__":
    # Replace 'your_script.py' with the name of your Python script
    create_executable("bulk_image_ocr.py")
