# Decision Module: This module generates an attractiveness score
# and makes a decision of like or dislike based on that.

from driver_config import driver
from selenium.webdriver.common.by import By
import os

driver.get('https://photo-ranker.com/')

def upload_profile_pics(driver, file_dir):
    # Get a list of all files in the directory
    files = os.listdir(file_dir)

    # Generate the file input selector
    file_input_selector = "//input[@type='file']"

    # Initialize an empty list to store all file paths
    all_file_paths = []

    # Loop through all files
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(file_dir, file)

        # Add the file path to the list
        all_file_paths.append(file_path)

    # Join all file paths into a single string, separated by '\n'
    all_files_string = "\n".join(all_file_paths)

    # Find the file input element and send all file paths to it at once
    file_input = driver.find_element(By.XPATH, file_input_selector)
    file_input.send_keys(all_files_string)
    return len(all_file_paths)



def make_decision():
    # Call the function to upload all images in a directory
    number_of_profile_pics = upload_profile_pics(driver, "./profile_pics")

    span = driver.find_element(By.XPATH, "//span[text()='Analyse Images']")
    span.click()

    span = driver.find_element(By.XPATH, "//span[text()='Show Score']")
    span.click()

    # gather the numbers in all tag similar to <h3 class="mantine-Text-root mantine-Title-root mantine-10djyvg">8.36</h3>
    # find the average of all those numbers, skip ones that are not numbers
    numbers = driver.find_elements(By.XPATH, "//h3[contains(@class, 'mantine-Text-root')]")
    total = 0
    count = 0
    for number in numbers:
        try:
            number = float(number.text)
            total += number
            count += 1
        except:
            pass

    return total / count

