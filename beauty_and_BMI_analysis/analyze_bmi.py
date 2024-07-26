import torch
import os
import cv2
from PIL import Image
from torchvision import transforms
from face_to_bmi_vit.scripts.loader import vit_transforms
from face_to_bmi_vit.scripts.models import get_model
from urllib.request import urlretrieve
from tqdm import tqdm


def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haarcascade_path = './mp_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face, (x, y, w, h)


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

def main():
    vit_model = get_model()
    weights_path = 'face_to_bmi_vit/models/aug_epoch_7.pt'
    weights_url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
    download_weights_if_not_exist(weights_path, weights_url)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    vit_model.eval()

    image_path = "temp_face.jpg"
    image = cv2.imread(image_path)
    face, _ = detect_and_crop_face(image)
    if face is not None:
        bmi = analyze_bmi(face, vit_model, device)
        print(f"BMI: {bmi}")
    else:
        print("No face detected")

if __name__ == "__main__":
    main()
