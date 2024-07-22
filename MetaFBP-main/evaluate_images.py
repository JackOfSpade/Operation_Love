import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
from model.dynamic_maml import DynamicMAML


def load_model(model_path, args):
    my_model = DynamicMAML(args)
    state_dict = torch.load(model_path)
    my_model.load_state_dict(state_dict['state_dict'])
    my_model.eval()
    return my_model


def preprocess_image(image_path, mtcnn):
    image = Image.open(image_path).convert('RGB')
    # Detect and crop face using MTCNN
    face, _ = mtcnn(image, return_prob=True)
    if face is not None:
        image = face

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Standard size for ResNet models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def evaluate_image(my_model, my_image_tensor):
    with torch.no_grad():
        my_output = my_model(my_image_tensor)  # Assuming the model can be called directly
    return my_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images to evaluate')
    # Add any other arguments that DynamicMAML might need
    parser.add_argument('--n-way', type=int, default=5, help='n way')
    parser.add_argument('--k-spt', type=int, default=5, help='Number of support shots')
    parser.add_argument('--k-qry', type=int, default=5, help='Number of query shots')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--update-lr', type=float, default=0.01, help='Learning rate for updates')
    parser.add_argument('--update-step', type=int, default=5, help='Number of update steps')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone network')
    parser.add_argument('--dy-mode', type=str, default='rebirth', choices=['rebirth', 'tuning'], help='Dynamic mode')

    args = parser.parse_args()

    mtcnn = MTCNN(keep_all=False)  # Initialize MTCNN for face detection
    model = load_model(args.model_path, args)
    image_paths = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if
                   img.endswith(('jpg', 'jpeg', 'png', 'webp'))]

    for image_path in image_paths:
        image_tensor = preprocess_image(image_path, mtcnn)
        output = evaluate_image(model, image_tensor)
        print(f"Image: {image_path}, Prediction: {output.item()}")  # Adjust based on your model's output format
