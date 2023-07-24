# Automation Module: This module handles all interactions with the
# dating app. It navigates to the appropriate tab in the browser,
# goes to user profiles, and likes or dislikes. This module uses
# Selenium to automate these tasks.


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common.exceptions import StaleElementReferenceException

from driver_config import driver
import tkinter as tk
from tkinter import simpledialog


def get_verification_code():
    # Create a new Tk root window (you might already have one somewhere)
    root = tk.Tk()
    # Hide the root window
    root.withdraw()
    # Create a simple dialog asking for the verification code
    code = simpledialog.askstring("Verification", "Please enter the verification code:", parent=root)
    # Destroy the root window
    root.destroy()
    return code


def login(dating_app):
    if dating_app == "tinder":
        # This part of your script navigates to the login page and enters your credentials
        driver.get('https://tinder.com')

        wait = WebDriverWait(driver, 10)

        WebDriverWait(driver, 900).until(
            expected_conditions.presence_of_element_located((By.XPATH, "//div[text()='Log in']"))
        ).click()

        wait = WebDriverWait(driver, 10)

        WebDriverWait(driver, 900).until(
            expected_conditions.presence_of_element_located((By.XPATH, "//div[text()='Log in with phone number']"))
        ).click()

        WebDriverWait(driver, 900).until(
            expected_conditions.presence_of_element_located((By.XPATH, 'phone_number'))
        ).send_keys('716-305-8819')

        WebDriverWait(driver, 900).until(
            expected_conditions.presence_of_element_located((By.XPATH, "//div[text()='Continue']"))
        ).click()

        # At this point, you pause the script and ask the user to enter the verification code
        verification_code = get_verification_code()

        # Then you find the input field for the verification code and submit it
        # Split the verification code into individual digits and enter them one by one into input fields with
        # the attribute aria-label="OTP code digit 1" and aria-label="OTP code digit 2", etc. (up to 6)

        # Convert the verification code to a list of digits
        verification_code = list(str(verification_code))

        for i in range(6):
            WebDriverWait(driver, 900).until(
                expected_conditions.presence_of_element_located(
                    (By.XPATH, f"//input[@aria-label='OTP code digit {i + 1}']"))
            ).send_keys(verification_code[i])

        WebDriverWait(driver, 900).until(
            expected_conditions.presence_of_element_located((By.XPATH, "//div[text()='Continue']"))
        ).click()


def navigate_to_tab(dating_app):
    # Get a list of all window handles currently open
    windows = driver.window_handles

    # Loop through the list of window handles
    for window in windows:
        # Switch to the window
        driver.switch_to.window(window)

        # Check if the title of the current tab contains the desired string
        if dating_app.lower() in driver.title.lower():
            # If it does, we're done
            return

    # If we get to this point, no tab with the given title was found
    print(f"No tab with title '{dating_app}' found")


def navigate_to_discover(dating_app):
    # This function will navigate to a user's profile.
    if dating_app == "eharmony":
        # find and click this element: <span class="iconText">Matches</span>
        span = driver.find_element(By.XPATH, "//span[text()='Matches']")
        span.click()


def remaining_super_likes(dating_app):
    if dating_app == "tinder":
        a = driver.find_element(By.XPATH, "//a[@title='My Profile']")
        a.click()
        # Find the remaining_super_likes in "X remaining" in this element: <div class="iconCombo Pos(r) P(16px) W(100%) CenterAlign Bdrs(8px) Pt(28px) Mend(12px) Mend(24px)--ml Mstart(5px)--ml Cur(p) Bxsh($bxsh-btn) Bgc($c-ds-background-primary) Ta(c) Cur(p) focus-button-style" tabindex="0" role="button"><div class="iconCombo__icon Pos(a) Start(50%) T(0) Translate(-50%,-50%)"><span class="Sq(48px) Bxsh($bxsh-btn) Bgc($c-ds-background-button-primary-overlay) P(8px) Bdrs(50%) CenterAlign Mx(a)"><svg focusable="false" aria-hidden="true" role="presentation" viewBox="0 0 24 24" width="24px" height="24px" class="Expand"><path d="M21.06 9.06l-5.47-.66c-.15 0-.39-.25-.47-.41l-2.34-5.25c-.47-.99-1.17-.99-1.56 0L8.87 7.99c0 .16-.23.4-.47.4l-5.47.66c-1.01 0-1.25.83-.46 1.65l4.06 3.77c.15.16.23.5.15.66L5.6 20.87c-.16.98.4 1.48 1.33.82l4.69-2.79h.78l4.69 2.87c.78.58 1.56 0 1.25-.98l-1.02-5.75s0-.4.23-.57l3.91-3.86c.78-.82.78-1.64-.39-1.64v.08z" fill="var(--fill--background-super-like, none)"></path></svg></span></div><div class="Fx($flx1)"><p class="Fz($s) Fw($semibold) Fz($ms)--m Mb(10px) Mt(0) My(4px)">5 remaining</p><div class="Fz($xs) C($c-ds-text-secondary) C($c-ds-text-super-like)!"><div>Get More Super Likes</div></div></div></div>
        # Specifically in here: <p class="Fz($s) Fw($semibold) Fz($ms)--m Mb(10px) Mt(0) My(4px)">5 remaining</p> Note that the remaining_super_likes will change.
        p = driver.find_element(By.XPATH, "//p[contains(text(),'remaining')]")
        # Get the text from the element
        text = p.text
        # Split the text on the space character
        split_text = text.split(" ")
        # Get the first item in the list
        remaining_super_likes = split_text[0]
        # Convert the string to an integer
        remaining_super_likes = int(remaining_super_likes)
        # Go back on the page
        driver.back()
        # Return the remaining_super_likes
        return remaining_super_likes


def like(dating_app):
    # This function will like a profile.
    if dating_app == "tinder":
        span = driver.find_element(By.XPATH, "//span[text()='Like']")
        span.click()


def super_like(dating_app):
    # This function will super like a profile.
    if dating_app == "tinder":
        span = driver.find_element(By.XPATH, "//span[text()='Super Like']")
        span.click()


def dislike(dating_app):
    # This function will dislike on a profile.
    if dating_app == "tinder":
        span = driver.find_element(By.XPATH, "//span[text()='Nope']")
        span.click()
