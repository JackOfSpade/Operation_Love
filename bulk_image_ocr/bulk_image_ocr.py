import os
from PIL import Image
import pytesseract


def main():
    # Define the name of the directory to be scanned using a relative path
    directory = 'C:\\Users\\Bull\\Desktop\\Github\\Operation_Love\\Screenshots'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Create or open a text file for storing OCR data
    with open("bulk_images_ocr_text.txt", "w") as output_file:
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Perform OCR
                image_path = os.path.join(directory, filename)
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)

                # Write OCR text to the output file
                output_file.write(text)
                output_file.write("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
