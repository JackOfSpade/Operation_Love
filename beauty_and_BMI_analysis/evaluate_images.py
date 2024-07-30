import os
import subprocess
from PIL import Image
import cv2
import shutil
import dlib


def run_in_env(script_command, env_path):
    activate_env_command = f'{env_path}\\Scripts\\Activate.ps1'
    command = f'{activate_env_command}; {script_command}'
    result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True)
    return result.stdout, result.stderr


def resize_image(image, max_size=(800, 800)):
    image.thumbnail(max_size, Image.LANCZOS)
    return image


def detect_and_crop_face(image_path, margin=0.5):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    try:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not loaded properly.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cnn_face_detector(gray_image, 1)

        if len(faces) == 0:
            return None

        face_rect = faces[0].rect
        x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())

        # Calculate margin size
        x_margin = int(w * margin)
        y_margin = int(h * margin)

        # Adjust the bounding box with the margin
        x_start = max(x - x_margin, 0)
        y_start = max(y - y_margin, 0)
        x_end = min(x + w + x_margin, image.shape[1])
        y_end = min(y + h + y_margin, image.shape[0])

        cropped_face = image[y_start:y_end, x_start:x_end]
    except Exception as e:
        print(f"Resizing image because of: {e}")
        pil_image = Image.open(image_path)
        width, height = pil_image.size

        # Calculate new dimensions (10% reduction)
        max_width = int(width * 0.9)
        max_height = int(height * 0.9)

        resized_image = resize_image(pil_image, max_size=(max_width, max_height))
        resized_image.save(image_path)

        cropped_face = detect_and_crop_face(image_path)

    return cropped_face


def main():
    screenshots_dir = "../Screenshots"
    analyze_beauty_script_path = "analyze_beauty.py"
    analyze_bmi_script_path = "analyze_bmi.py"
    analyze_beauty_venv_path = "./analyze_beauty.venv"
    analyze_bmi_venv_path = "./analyze_bmi.venv"
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    cropped_folder_path = screenshots_dir + "/cropped"
    # Make folder if it doesn't exist
    os.makedirs(cropped_folder_path, exist_ok=True)

    image_files = [f for f in os.listdir(screenshots_dir) if
                   os.path.isfile(os.path.join(screenshots_dir, f)) and os.path.splitext(f)[
                       1].lower() in valid_extensions]

    for image_file in image_files:
        print("\n" + os.path.splitext(image_file)[0])

        image_path = os.path.join(screenshots_dir, image_file)

        # Load image to ensure it is valid
        image = cv2.imread(image_path)

        if image is None:
            continue

        cropped_image_path = cropped_folder_path + "/cropped_" + image_file
        uncropped_image_path = screenshots_dir + "/" + image_file

        # Load and process the image
        face = detect_and_crop_face(image_path)

        if face is not None:
            # Save the cropped face to a file
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
                    if beauty_score >= 4 and BMI_score <= 21.7:
                        super_like_folder = screenshots_dir + "/super_like"
                        os.makedirs(super_like_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, super_like_folder)
                    elif ((beauty_score >= 3 and BMI_score <= 21.7)
                          or (beauty_score >= 3.5 and BMI_score <= 24.9)
                          or (beauty_score >= 2.5 and BMI_score < 18.5)):
                        like_folder = screenshots_dir + "/like"
                        os.makedirs(like_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, like_folder)
                    else:
                        dislike_folder = screenshots_dir + "/dislike"
                        os.makedirs(dislike_folder, exist_ok=True)
                        # Move the file
                        shutil.move(parent_new_file_path2, dislike_folder)

                else:
                    print(f"analyze_bmi error: {error}")
            else:
                print(f"analyze_beauty error: {error}")
        else:
            print("No face detected")
            directory = os.path.dirname(uncropped_image_path)
            base_name, ext = os.path.splitext(os.path.basename(uncropped_image_path))
            new_base_name = f"{base_name} No Face Detected"
            new_file_path = os.path.join(directory, new_base_name + ext)
            new_file_path = new_file_path.replace("\\", "/")
            os.rename(uncropped_image_path, new_file_path)

if __name__ == "__main__":
    main()
