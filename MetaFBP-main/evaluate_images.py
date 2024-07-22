import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
from model.dynamic_maml import DynamicMAML

def load_model(model_path, args):
    my_model = DynamicMAML(args)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('net'):
            new_key = k.replace('net.', '')
            new_state_dict[new_key] = v
        elif k.startswith('meta'):
            new_key = k.replace('meta.', '')
            new_state_dict[f'meta.{new_key}'] = v
        else:
            new_state_dict[k] = v

    my_model.load_state_dict(new_state_dict, strict=False)
    my_model.eval()
    return my_model

def preprocess_image(image_path, mtcnn):
    image = Image.open(image_path).convert('RGB')
    face, _ = mtcnn(image, return_prob=True)
    if face is not None:
        image = face

    if isinstance(image, torch.Tensor):
        image = image.unsqueeze(0)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)

    return image

def evaluate_image(my_model, my_image_tensor):
    with torch.no_grad():
        my_output = my_model(my_image_tensor)
    return my_output

def scale_prediction(score, min_score, max_score, new_min=1, new_max=10):
    # Scale the score to the new range
    scaled_score = (score - min_score) / (max_score - min_score) * (new_max - new_min) + new_min
    return scaled_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--n-way', type=int, default=5, help='n way')
    parser.add_argument('--k-spt', type=int, default=5, help='Number of support shots')
    parser.add_argument('--k-qry', type=int, default=5, help='Number of query shots')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--update-lr', type=float, default=0.01, help='Learning rate for updates')
    parser.add_argument('--update-step', type=int, default=5, help='Number of update steps')
    parser.add_argument('--update-step-test', type=int, default=10, help='Update steps for fine-tuning')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone network')
    parser.add_argument('--dy-mode', type=str, default='rebirth', choices=['rebirth', 'tuning'], help='Dynamic mode')
    parser.add_argument('--meta-lr', type=float, default=0.001, help='Meta-level outer learning rate')
    parser.add_argument('--task-num', type=int, default=4, help='Meta batch size, namely task num')
    parser.add_argument('--cpu-only', action='store_true', help='Run all on CPU')
    parser.add_argument('--pretrain-type', type=str, default='none', choices=['imagenet', 'fea', 'none'],
                        help='Pretraining type')
    parser.add_argument('--work-dir', type=str, default='./save', help='Working directory')
    parser.add_argument('--dataset', type=str, default='FBP5500', help='Dataset name')

    args = parser.parse_args()
    args.cpu_only = True

    mtcnn = MTCNN(keep_all=False, device='cpu')
    model = load_model(args.model_path, args)
    image_paths = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if
                   img.endswith(('jpg', 'jpeg', 'png', 'webp'))]

    # Estimated min and max values using normal distribution
    mean_score = -0.028930393321185625
    std_score = 0.011737048968824352
    min_score = mean_score - 3 * std_score
    max_score = mean_score + 3 * std_score

    for image_path in image_paths:
        image_tensor = preprocess_image(image_path, mtcnn)
        output = evaluate_image(model, image_tensor)
        # Ensure the output is correctly interpreted as a scalar value
        score = output.item() if isinstance(output, torch.Tensor) else float(output)
        print(f"Raw output for {image_path}: {score}")  # Debug print for raw output
        scaled_output = scale_prediction(score, min_score=min_score, max_score=max_score, new_min=1, new_max=10)
        print(f"Image: {image_path}, Prediction: {scaled_output:.2f}")
