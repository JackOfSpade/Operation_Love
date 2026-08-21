import os
import sys

from ComboLoss.main.inference import FacialBeautyPredictor


def main(cropped_image_path, uncropped_image_path):
    # Define paths
    pretrained_model_path = './ComboLoss/models/ComboNet_SCUTFBP5500.pth'

    # Load the pretrained model
    predictor = FacialBeautyPredictor(pretrained_model_path)


    if cropped_image_path is not None:
        # Get the beauty score using the new model
        beauty_score = predictor.infer(cropped_image_path)
        print(f"Beauty Score: {beauty_score['beauty']:.2f}")

        # Split the path into directory, base name, and extension
        directory = os.path.dirname(cropped_image_path)
        base_name, ext = os.path.splitext(os.path.basename(cropped_image_path))
        parent_directory = os.path.dirname(uncropped_image_path)
        parent_base_name, parent_ext = os.path.splitext(os.path.basename(uncropped_image_path))

        # Define the new file name with the additional information
        new_base_name = f"{base_name}  Beauty Score {beauty_score['beauty']:.2f}"
        new_parent_base_name = f"{parent_base_name}  Beauty Score {beauty_score['beauty']:.2f}"

        # Create the new file path by joining the directory, new base name, and extension
        new_file_path = os.path.join(directory, new_base_name + ext)
        new_file_path = new_file_path.replace("\\", "/")
        parent_new_file_path = os.path.join(parent_directory, new_parent_base_name + parent_ext)
        parent_new_file_path = parent_new_file_path.replace("\\", "/")

        # Rename the file
        os.rename(cropped_image_path, new_file_path)
        print(f"{new_file_path}")
        os.rename(uncropped_image_path, parent_new_file_path)
        print(f"{parent_new_file_path}")


if __name__ == "__main__":
    cropped_image_path = sys.argv[1]
    uncropped_image_path = sys.argv[2]
    main(cropped_image_path, uncropped_image_path)
