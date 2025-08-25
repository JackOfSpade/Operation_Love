import cv2
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from urllib.request import urlretrieve
from face_to_bmi_vit.scripts.loader import vit_transforms
from face_to_bmi_vit.scripts.models import get_model


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights_if_not_exist(weights_path, url):
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print(f"Downloading pre-trained weights to {weights_path}...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, weights_path, reporthook=t.update_to)
        print("Download complete.")

def analyze_bmi(face, model, device):
    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face = transforms.ToTensor()(face)
    face = vit_transforms(face).unsqueeze(0).to(device)
    with torch.no_grad():
        bmi = model(face)
    return bmi.item()

def main(new_file_path, parent_new_file_path):
    # vit_model = get_model()
    # weights_path = 'face_to_bmi_vit/models/aug_epoch_7.pt'
    # weights_url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
    # download_weights_if_not_exist(weights_path, weights_url)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # vit_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device), weights_only=False))
    # vit_model.eval()


    if new_file_path is not None:
        #face = cv2.imread(new_file_path)
        #bmi = analyze_bmi(face, vit_model, device)
        bmi = 21.70
        print(f"BMI: {bmi:.2f}")


        # Split the path into directory, base name, and extension
        directory = os.path.dirname(new_file_path)
        base_name, ext = os.path.splitext(os.path.basename(new_file_path))
        parent_directory = os.path.dirname(parent_new_file_path)
        parent_base_name, parent_ext = os.path.splitext(os.path.basename(parent_new_file_path))

        # Define the new file name with the additional information
        new_base_name = f"{base_name}  BMI {bmi:.2f}"
        new_parent_base_name = f"{parent_base_name}  BMI {bmi:.2f}"

        # Create the new file path by joining the directory, new base name, and extension
        new_file_path2  = os.path.join(directory, new_base_name + ext)
        new_file_path2 = new_file_path2.replace("\\", "/")
        parent_new_file_path2 = os.path.join(parent_directory, new_parent_base_name + parent_ext)
        parent_new_file_path2 = parent_new_file_path2.replace("\\", "/")

        # Rename the file
        os.rename(new_file_path, new_file_path2)
        os.rename(parent_new_file_path, parent_new_file_path2)
        print(f"{parent_new_file_path2}")



if __name__ == "__main__":
    new_file_path = sys.argv[1]
    parent_new_file_path = sys.argv[2]
    main(new_file_path, parent_new_file_path)
