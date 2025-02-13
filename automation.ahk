; Automation Module: This module handles all interactions with the
; dating app. It navigates to the appropriate tab in the browser,
; goes to user profiles, and likes or dislikes. This module uses
; Selenium to automate these tasks.

#include helper_functions.ahk
#include main.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

hinge_opener := "Hi, how's your week going?"

navigate_to_discover(dating_app)
{
    if dating_app == "tinder"
    {
		winactivate "Tinder"
		sleep 500
    }
	else if dating_app == "bumble"
    {
		winactivate "Bumble"
		sleep 500
    }
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		sleep 500
    }
	else if dating_app == "okcupid"
	{
		winactivate "OkCupid"
		sleep 500
	}
	else if dating_app == "2redbeans"
	{
		winactivate "2RedBeans"
		sleep 500
		
		loop 17
		{
			send "{down}"
			sleep 100
		}
		
		; Wait for scrolling to stop
		sleep 1000
	}
	else if dating_app == "photofeeler"
    {
		; don't winactivate "Vote" because continous refresh make loading time inconsistent
    }
}



; remaining_super_likes(dating_app)
; {
; 	remainingSuperLikes := 0
; 	if dating_app == "tinder"
;     {
; 		; Click profile
; 		MouseClick "left", 35, 168
; 		sleep 3000
; 			
; 		remainingSuperLikes := ocr(234, 394, 249, 420, 500)
; 		
; 		; Go back to discover
; 		MouseClick "left", 35, 168
; 		sleep 3000
; 	}
; 	else if dating_app == "bumble"
; 	{		
; 		; Click profile (have to click multiple times for it to trigger for some reason sometimes)
; 		mouseClick "left", 713, 1029
; 		sleep 1000
; 		mouseClick "left", 713, 1029
; 		sleep 1000
; 		mouseClick "left", 713, 1029
; 		sleep 1000
; 		
; 		remainingSuperLikes := ocr(1060, 502, 1090, 539, 100)
; 		
; 		; Go back to discover
; 		MouseClick "left", 962, 1042
; 		sleep 2000		
; 	}	
; 	else if dating_app == "hinge"
;     {
; 		; Superlike manually in standouts since it doesn't replenish (only 1 free rose/week)
;     }
; 	else if dating_app == "okcupid"
; 	{
; 		; Click superlike
; 		MouseClick "left", 986, 410
; 		sleep 1000
; 		
; 		remainingSuperLikes := ocr(1120, 705, 1132, 720, 100)
; 		
; 		; Click out of superlike popup
; 		MouseClick "left", 676, 583
; 		sleep 500
; 	}
; 	else if dating_app == "2redbeans"
; 	{
; 	}
; 	
; 	if !IsNumber(remainingSuperLikes)
; 	{
; 		remainingSuperLikes := 0
; 	}
; 	
; 	return remainingSuperLikes
; 	
; }
		   
; super_like(dating_app, root_directory)
; {
; 	if dating_app == "tinder"
; 	{		
; 		; Fix random profile popups
; 		loop 4
; 		{
; 			; Tinder arrow keys sometimes don't work on first try
; 			send "{down}"
; 			sleep 500
; 		}
; 		
; 		; Click super like
; 		mouseClick "left", 1145, 820
; 		sleep 2000
; 		
; 		; Add comment
; 		mouseClick "left", 988, 792
; 		sleep 1000
; 		send "Hi"
; 		; Need to sleep >500ms or else Tinder won't register the send; probably due to on-hover scripts running on the send button.
; 		sleep 1000
; 		; Click send
; 		mouseClick "left", 1299, 794
; 		
; 	}
; 	else if dating_app == "bumble"
; 	{
; 		mouseClick "left", 1162, 751
; 		sleep 500
; 		; diff spot
; 		mouseClick "left", 1175, 886	
; 	
; 	}
; 	else if dating_app == "hinge"
;     {
; 		like(dating_app, root_directory)
;     }
; 	else if dating_app == "okcupid"
; 	{
; 		;; Click superlike
; 		;MouseClick "left", 986, 410
; 		;sleep 1000
; 		;
; 		;; Click send without message
; 		;mouseClick "left", 1154, 662
; 		;
; 		;; Wait for superlike animation to finish
; 		;sleep 4000
; 		;
; 		;sleep 1000
; 		;
; 		;; Click out of it's a match popup
; 		;MouseClick "left", 168, 514
; 		;sleep 500
; 		
; 		like(dating_app, root_directory)
; 	}
; 	else if dating_app == "2redbeans"
; 	{
; 	}
; 	else if dating_app == "photofeeler"
; 	{
; 		random_number := Random(1, 2)
; 		
; 		if random_number == 1
; 		{
; 			mouseClick "left", 746, 347
; 		}
; 		else if random_number == 2
; 		{
; 			mouseClick "left", 746, 387
; 		}
; 		
; 		sleep 499
; 		
; 		
; 		random_number := Random(1, 2)
; 		
; 		if random_number == 1
; 		{
; 			mouseClick "left", 946, 343
; 		}
; 		else if random_number == 2
; 		{
; 			mouseClick "left", 938, 382
; 		}
; 		
; 		sleep 500
; 		
; 		
; 		random_number := Random(1, 2)
; 		
; 		if random_number == 1
; 		{
; 			mouseClick "left", 1079, 344
; 		}
; 		else if random_number == 2
; 		{
; 			mouseClick "left", 1194, 391
; 		}
; 		
; 		sleep 501
; 		
; 		; Click submit
; 		mouseClick "left", 1177, 730
; 	}
; }

   
like(dating_app, root_directory)
{

	start_of_like_function_label:
	
	global hinge_opener
	
	if dating_app == "tinder"
	{		
		; Pull down profile description if up.
		loop 4
		{
			; Tinder arrow keys sometimes don't work on first try
			send "{down}"
			sleep 500
		}
		
		; Click like
		mouseClick "left", 1225, 845
		
		sleep 1000
		
		; Click away super like upgrade popup
		mouseClick "left", 964, 771
		
		sleep 500
		
		; Click away second time's a charm (compliments promo). Need to click 2 times in case one of the profiles are enlarged.
		; mouseClick "left", 702, 67
		; sleep 500
		; mouseClick "left", 702, 67
	}
	else if dating_app == "bumble"
	{
		; click_and_drag(724, 539, 1182, 536, 2000)
		mouseClick "left", 1261, 965
	}
	else if dating_app == "hinge"
    {		
		; Swipe down button
		mouseClick "left", 1724, 39
		sleep 1000
		
		; Click like locations, but change from AirDroid menu coords first by clicking on an area away from the app area.
		mouseClick "left", 230, 559
		sleep 1000
		; like location 1
		mouseClick "left", 1186, 778
		sleep 1000
		; like location 2
		mouseClick "left", 1186, 862
		sleep 1000
		
		; Depending on resolution of the laptop, some resolution make the "Send Like" button disappear if try to add comments -----------------------
		; Click textbox
		mouseClick "left", 953, 791
		
		sleep 1000
		
		send hinge_opener
		
		sleep 6000
		; ----------------------------------------------------------------
		
		; Send like after comment
		mouseClick "left", 1017, 870
		
		sleep 3000
		
		; Click away popups
		;;;;;;;;;;;;;;;;;;;
		sleep 100
		
		; Use up all your roses so the rose suggestion pop-up don't show up. It is difficult to account for it on device change b/c it only show up occasionally.
    }	
	else if dating_app == "okcupid"
	{
		; Click like
		mouseClick "left", 849, 410		
		
		sleep 1000
		
		; Click out of superlike popup (click "like them anyway")
		MouseClick "left", 692, 589
		sleep 500
		
		; Click out of it's a match popup. Consequentially, clicks out of zoomed picture popup if no superlike popup.
		MouseClick "left", 168, 514
		sleep 500
	}
	else if dating_app == "2redbeans"
	{
		mouseClick "left", 222, 229
		sleep 3000
		mouseClick "left", 620, 311
		send "Hi, how's your week going?"
		sleep 1000
		; Click send
		mouseClick "left", 900, 365
		sleep 500
		; Click send (something coords get messed up for some reason)
		mouseClick "left", 993, 370
		send 1000
		
		restart_2redbeans()
		
		; Clear from search list
		dislike(dating_app)
	}
	else if dating_app == "photofeeler"
	{
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 746, 387
		}
		else if random_number == 2
		{
			mouseClick "left", 723, 431
		}
		
		sleep 499
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 938, 382
		}
		else if random_number == 2
		{
			mouseClick "left", 929, 433
		}
		
		sleep 500
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 1194, 391
		}
		else if random_number == 2
		{
			mouseClick "left", 1206, 441
		}
		
		sleep 501
		
		; Click submit
		mouseClick "left", 1193, 712
		mouseClick "left", 1175, 727		
	}
	
	return hinge_opener
}



   
dislike(dating_app)
{
	if dating_app == "tinder"
	{		
		; Fix random profile popups
		loop 4
		{
			; Tinder arrow keys sometimes don't work on first try
			send "{down}"
			sleep 500
		}
		
		mouseClick "left", 1075, 845
	}
	else if dating_app == "bumble"
	{
		; click_and_drag(1182, 536, 724, 539, 2000)
		mouseClick "left", 1055, 975
	}
	else if dating_app == "hinge"
    {
		; Swipe down button
		mouseClick "left", 1724, 39
		sleep 1000
		
		
		; Dislike button with footer, but change from AirDroid menu coords first by clicking on an area away from the app area.
		mouseClick "left", 230, 559
		sleep 1000
		mouseClick "left", 723, 918
		
		; Click dislike with footer and compatibility caption
		;;;;;;;;;;;;;;;;;;;;
		
		sleep 3000
    }
	else if dating_app == "okcupid"
	{
		; Click dislike
		mouseClick "left", 660, 410	
	}
	else if dating_app == "2redbeans"
	{
		mouseClick "left", 828, 190
		sleep 500
		mouseClick "left", 751, 224
		
		; Wait for list to scroll up
		sleep 1000
	}
	else if dating_app == "photofeeler"
	{
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 723, 431
		}
		else if random_number == 2
		{
			mouseClick "left", 745, 481
		}
		
		sleep 499
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 929, 433
		}
		else if random_number == 2
		{
			mouseClick "left", 960, 471
		}
		
		sleep 500
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 1206, 441
		}
		else if random_number == 2
		{
			mouseClick "left", 1145, 488
		}
		
		sleep 501
		
		; Click submit
		mouseClick "left", 1177, 730
	}
}
