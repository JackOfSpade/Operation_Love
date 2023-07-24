# Main Controller: This module will coordinate the
# operations of the other modules. It will instruct the
# Automation Module to navigate to a profile, then tell the Screenshot
# Module to take a picture, pass that picture to
# the Decision Module to generate a score and make a decision from that.

; Set-up:
; Download windows snipping tool and set its setting to auto-save
; Open 2 separate chrome window, one with the dating site and one with https://photo-ranker.com/'

import automation
import screenshot
import decision
import compiling

def main(dating_app):
    # This function coordinates all the other modules.
    is_empty = False

    # Pre-check is_empty

    while (not is_empty):
        automation.login(dating_app)
        # automation.navigate_to_tab(dating_app)
        # automation.navigate_to_discover(dating_app)
        # remaining_super_likes = automation.remaining_super_likes(dating_app)
        # screenshot.take_screenshot(dating_app)
        # average_score = decision.make_decision()

        # if average_score >= 8 and remaining_super_likes > 0:
        #     automation.super_like(dating_app)
        # elif average_score >= 6 and remaining_super_likes > 0:
        #     automation.like(dating_app)
        # else:
        #     automation.dislike(dating_app)


        if dating_app == "tinder" and not is_empty:
            is_empty = True



if __name__ == "__main__":
    dating_app_list = ["tinder", "bumble", "okcupid", "match", "eharmony"]

    main("tinder")
	
	
f12::
{
	exitApp
}