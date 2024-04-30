import os
import clip
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def detect_and_crop_face(image, device):
    # Load the pre-trained face detection model with new API
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

    # Convert PIL image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device)

    # Perform face detection
    with torch.no_grad():
        prediction = model([image_tensor])

    # Process detection results
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']  # Assuming scores are available

    # Check if any boxes are detected
    if boxes.shape[0] > 0:
        # Select the box with the highest score (you could also choose based on size)
        max_score_index = scores.argmax()
        box = boxes[max_score_index].cpu().numpy()
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        return cropped_image
    return None


def main(test):
    script_directory = os.getcwd()
    output_text = ""

    # Test
    if test:
        directory = os.path.join(script_directory, 'test photos')
        directory_cropped = os.path.join(script_directory, 'test photos/cropped photos')
    else:
        if script_directory.endswith("open_ai_clip"):
            directory = os.path.join(script_directory, '..', 'Screenshots')
        else:
            directory = os.path.join(script_directory, 'Screenshots')

        directory_cropped = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Most computationally demanding CLIP model
    model, preprocess = clip.load("ViT-L/14", device=device)

    text = clip.tokenize(["beautiful face", "ugly face", "indeterminate"]).to(device)

    all_files = os.listdir(directory)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    total_beautiful_prob = 0
    total_ugly_prob = 0
    num_images = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)

        # Convert the image to RGB (if not already in this format)
        image = image.convert('RGB')

        # Detect and crop face
        cropped_image = detect_and_crop_face(image, device)
        if cropped_image is None:
            # For debugging only
            print(f"num_images -= 1: cropped_image is None\n") 
            num_images -= 1
            continue

        if test:
            # Define the output file path with the appropriate extension
            output_file_path = os.path.join(directory_cropped, f"cropped_{image_file}")
            # Save the cropped image directly
            cropped_image.save(output_file_path)

        # Preprocess the cropped image for CLIP model input
        image_processed = preprocess(cropped_image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image_processed, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if probs[0][2] > 0.5:  # Assuming the third category is "non-face"
            # For debugging only
            print(f"num_images -= 1: indeterminate probability: {probs[0][2]:.3f}\n")
            num_images -= 1
            continue

        total_beautiful_prob += probs[0][0]
        total_ugly_prob += probs[0][1]

    if num_images == 0:
        avg_beautiful_prob = avg_ugly_prob = 0
    else:
        avg_beautiful_prob = total_beautiful_prob / num_images
        avg_ugly_prob = total_ugly_prob / num_images

    # Decision based on average probabilities
    if avg_beautiful_prob > avg_ugly_prob:
        final_label = "beautiful"
    elif avg_beautiful_prob < avg_ugly_prob:
        final_label = "ugly"
    elif avg_beautiful_prob == avg_ugly_prob:
        final_label = "neutral"

    result_path = os.path.join(script_directory, 'open_ai_clip_result.txt')

    with open(result_path, 'w') as f:
        # For debugging only
        print(f"Number of valid images: {num_images:.3f}\n")
    
        output_text += final_label + "\n"              
        output_text += f"Average probability of being considered beautiful: {avg_beautiful_prob:.3f}\n"
        output_text += f"Average probability of being considered ugly: {avg_ugly_prob:.3f}"
        f.write(output_text)
        print(output_text)


if __name__ == "__main__":
    main(test=False)
