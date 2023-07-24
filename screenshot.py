import hashlib
import os
from PIL import Image
from driver_config import driver
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys

def get_hash(image_path):
    # Open, resize and convert image to grayscale
    image = Image.open(image_path).resize((8, 8), Image.ANTIALIAS).convert('L')

    # Calculate average pixel intensity
    pixels = list(image.getdata())
    avg_pixel = sum(pixels) / len(pixels)

    # Create a hash string
    bits = "".join(['1' if (px > avg_pixel) else '0' for px in pixels])
    hash_format = hashlib.md5(bits.encode('utf-8'))
    return hash_format.hexdigest()


def remove_duplicates(directory):
    image_hashes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            image_hash = get_hash(file_path)
            if image_hash not in image_hashes:
                image_hashes[image_hash] = file_path
            else:
                # This is a duplicate image
                os.remove(file_path)


def take_screenshot(dating_app):
    # This function will take a screenshot of a profile picture.
    for i in range(6):
        if dating_app == "tinder":
            # Locate the element
            element = driver.find_element(By.XPATH, "//div[@role='img' and contains(@class, 'StretchedBox')]")

            # Take a screenshot of the element
            element.screenshot(f"./profile_pics/screenshot_{i}.png")

            # Press space
            driver.find_element(By.XPATH, "//body").send_keys(Keys.SPACE)

    remove_duplicates("./profile_pics/")
