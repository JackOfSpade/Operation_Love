import cv2
import dlib
import numpy as np
import os
import shutil
import subprocess
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from retinaface import RetinaFace
import uuid


def run_in_env(script_command, env_path):
    activate_env_command = f'{env_path}\\Scripts\\Activate.ps1'
    command = f'{activate_env_command}; {script_command}'
    result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True)
    return result.stdout, result.stderr


def detect_and_crop_face_retinaface(image_path, margin=0.5):
    # No dimension resize necessary. Inference code from original repos already does the proper dimension transformations.
    # Just make sure all features of the face are captured.

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not loaded properly.")

    # Convert the image to RGB (RetinaFace expects RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(rgb_image)

    if faces is None or len(faces) == 0:
        return None

    # Assuming we are only interested in the first detected face
    face_key = next(iter(faces.keys()))
    face = faces[face_key]
    face_box = face['facial_area']
    x, y, x_end, y_end = face_box

    w = x_end - x
    h = y_end - y

    # Calculate margin size
    x_margin = int(w * margin)
    y_margin = int(h * margin)

    # Adjust the bounding box with the margin
    x_start = max(x - x_margin, 0)
    y_start = max(y - y_margin, 0)
    x_end = min(x_end + x_margin, image.shape[1])
    y_end = min(y_end + y_margin, image.shape[0])

    cropped_face = image[y_start:y_end, x_start:x_end]

    return cropped_face


def main():
    screenshots_dir = "../Screenshots"
    analyze_beauty_script_path = "analyze_beauty.py"
    analyze_bmi_script_path = "analyze_bmi.py"
    analyze_beauty_venv_path = "./analyze_beauty.venv"
    analyze_bmi_venv_path = "./analyze_bmi.venv"
    valid_extensions = {'.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.gif', '.webp'}
    cropped_folder_path = screenshots_dir + "/cropped"
    # Make folder if it doesn't exist
    os.makedirs(cropped_folder_path, exist_ok=True)

    image_files = [f for f in os.listdir(screenshots_dir) if
                   os.path.isfile(os.path.join(screenshots_dir, f)) and os.path.splitext(f)[
                       1].lower() in valid_extensions]

    for image_file_name in image_files:
        ext = os.path.splitext(image_file_name)[1]

        image_path = os.path.join(screenshots_dir, image_file_name)

        # Generate UUID name
        image_file_name = f"{uuid.uuid4()}"
        print("\n" + image_file_name)
        image_file_name = image_file_name + ext
        new_image_path = os.path.join(screenshots_dir, image_file_name)

        os.rename(image_path, new_image_path)

        # Set original variable
        image_path = new_image_path

        # Load image to ensure it is valid
        image = cv2.imread(image_path)

        if image is None:
            continue

        cropped_image_path = cropped_folder_path + "/Cropped  " + image_file_name
        uncropped_image_path = screenshots_dir + "/" + image_file_name

        # Method 1
        face = detect_and_crop_face_retinaface(image_path=image_path)

        decision = None

        if face is not None:
            # Save the cropped faces to a file
            cv2.imwrite(cropped_image_path, face)

            analyze_beauty_script_command = f'python {analyze_beauty_script_path} "{cropped_image_path}" "{uncropped_image_path}"'
            output, error = run_in_env(analyze_beauty_script_command, analyze_beauty_venv_path)
            new_file_path = None
            parent_new_file_path = None
            beauty_score = None

            if output and not error:
                lines = output.split('\n')

                for line in lines:
                    # -----Test----------------------------------------------
                    # print("line: " + line)
                    # print("len(line.strip()) > 0: " + str(len(line.strip()) > 0))
                    # print("line.lower().startswith('beauty'): " + str(line.lower().startswith("beauty")))
                    # print("'cropped' in line.lower(): " + str("cropped" in line.lower()))
                    #--------------------------------------------------------
                    if len(line.strip()) > 0:
                        if line.lower().startswith("beauty"):
                            print(line)
                            beauty_score = float(line.split(": ")[1].strip())
                        elif "/cropped/" in line.lower():
                            new_file_path = line
                        else:
                            parent_new_file_path = line


                analyze_bmi_script_command = f'python {analyze_bmi_script_path} "{new_file_path}" "{parent_new_file_path}"'
                output, error = run_in_env(analyze_bmi_script_command, analyze_bmi_venv_path)
                parent_new_file_path2 = None
                BMI_score = None

                if output and not error:
                    lines = output.split('\n')

                    for line in lines:
                        if len(line.strip()) > 0:
                            if line.lower().startswith("bmi"):
                                print(line)
                                BMI_score = float(line.split(": ")[1].strip())
                            else:
                                parent_new_file_path2 = line

                    # Beauty Score range: 1 - 5
                    # Underweight: BMI less than 18.5
                    # Normal weight: BMI 18.5 – 24.9 (mid: 21.7)
                    # Overweight: BMI 25 – 29.9
                    # Obesity: BMI 30 or greater

                    # Baseline
                    baseline_beauty_score = 2.5
                    baseline_BMI_score = 24.9
                    
                    # 10% adjustment for possible inaccuracies
                    baseline_beauty_score = 2.25
                    baseline_BMI_score = 22.41

                    # Calculate percentage difference from baseline
                    beauty_percentage_diff = abs((beauty_score - baseline_beauty_score) / baseline_beauty_score)
                    BMI_percentage_diff = abs((BMI_score - baseline_BMI_score) / baseline_BMI_score)
                    
                    # Weights for perceived importance
                    beauty_weight = 1
                    BMI_weight = 3       # 0.5% change in BMI is equivalent to a 1.5% change in beauty score

                    # Calculate weighted percentage differences
                    weighted_beauty_diff = beauty_weight * beauty_percentage_diff
                    weighted_BMI_diff = BMI_weight * BMI_percentage_diff

                    if (beauty_score >= 3.5 and BMI_score <= 18.5):
                        super_like_folder = screenshots_dir + "/super_like"
                        os.makedirs(super_like_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, super_like_folder)

                        decision = "super_like"
                    elif ((beauty_score >= baseline_beauty_score and BMI_score <= baseline_BMI_score)
                    or
                    (beauty_score < baseline_beauty_score and BMI_score < baseline_BMI_score and weighted_BMI_diff >= weighted_beauty_diff)
                    or
                    (BMI_score > baseline_BMI_score and beauty_score > baseline_beauty_score and weighted_beauty_diff >= weighted_BMI_diff)):
                        like_folder = screenshots_dir + "/like"
                        os.makedirs(like_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, like_folder)

                        decision = "like"
                    else:
                        dislike_folder = screenshots_dir + "/dislike"
                        os.makedirs(dislike_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, dislike_folder)

                        decision = "dislike"
                else:
                    print(f"analyze_bmi error: {error}")
                    decision = "like"
            else:
                print(f"analyze_beauty error: {error}")
                decision = "like"
        else:
            print("No face detected")
            decision = "No face detected"

            directory = os.path.dirname(uncropped_image_path)
            base_name, ext = os.path.splitext(os.path.basename(uncropped_image_path))
            new_base_name = f"{base_name} No Face Detected"
            new_file_path = os.path.join(directory, new_base_name + ext)
            new_file_path = new_file_path.replace("\\", "/")
            os.rename(uncropped_image_path, new_file_path)


        with open(os.path.join(os.getcwd(), 'beauty_and_BMI_analysis_result.txt'), 'w') as file:
            file.write(decision)

if __name__ == "__main__":
    main()
