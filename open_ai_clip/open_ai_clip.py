import os
import sys
import torch
import clip
from PIL import Image


def main():
    # Get the directory of the current script
    script_directory = os.getcwd()

    # Check the directory from which the script is being run
    if script_directory.endswith("open_ai_clip"):
        directory = os.path.join(script_directory, '..', 'Screenshots')
    else:
        directory = os.path.join(script_directory, 'Screenshots')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(["beautiful girl", "ugly girl", "indeterminate"]).to(device)

    # List all files in the script directory
    all_files = os.listdir(directory)

    # Filter out the image files
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    total_beautiful_prob = 0
    total_ugly_prob = 0
    num_images = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Skip the image if the model thinks it's not a face with high probability
        if probs[0][2] > 0.5:  # Assuming that the third category is "non-face"
            num_images -= 1
            continue

        total_beautiful_prob += probs[0][0]
        total_ugly_prob += probs[0][1]

    if num_images == 0:
        avg_beautiful_prob = 0
        avg_ugly_prob = 0
    else:
        # Calculate average probabilities
        avg_beautiful_prob = total_beautiful_prob / num_images
        avg_ugly_prob = total_ugly_prob / num_images

    # Making a decision based on the average probabilities
    if avg_beautiful_prob > 0.5 and avg_ugly_prob < 0.5:
        final_label = "beautiful"
    elif avg_beautiful_prob < 0.5 and avg_ugly_prob > 0.5:
        final_label = "ugly"
    else:
        final_label = "neutral"

    result_path = os.path.join(script_directory, 'open_ai_clip_result.txt')

    with open(result_path, 'w') as f:
        output_text = final_label + "\n"
        output_text += f"Average probability of beautiful girl: {avg_beautiful_prob:.3f}\n"
        output_text += f"Average probability of ugly girl: {avg_ugly_prob:.3f}"

        # Write to file
        f.write(output_text)

        # Print to console
        print(output_text)

if __name__ == "__main__":
    main()
