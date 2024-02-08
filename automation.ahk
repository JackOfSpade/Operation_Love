; Automation Module: This module handles all interactions with the
; dating app. It navigates to the appropriate tab in the browser,
; goes to user profiles, and likes or dislikes. This module uses
; Selenium to automate these tasks.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

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
	else if dating_app == "photofeeler"
    {
		try 
		{
			winactivate "Vote"
		}
		catch as e 
		{
			winactivate "www.photofeeler.com"
		}
		
		sleep 500
    }
}



remaining_super_likes(dating_app)
{
	remainingSuperLikes := 0
	if dating_app == "tinder"
    {
		; Click profile
		MouseClick "left", 41, 159
		sleep 5000
			
		remainingSuperLikes := ocr(208, 394, 223, 411, 500)
		
		; Go back to discover
		MouseClick "left", 41, 159
		sleep 5000
	}
	else if dating_app == "bumble"
	{		
		mouseMove 1139, 955
		sleep 1000
		remainingSuperLikes := ocr(1139, 955, 1205, 1015, 100)
		
		if !IsNumber(remainingSuperLikes)
		{
			remainingSuperLikes := 0
		}
	}
	else if dating_app == "hinge"
    {
		; Do it manually in standouts since it doesn't replenish (only 1 free rose/week)
    }
	
	return remainingSuperLikes
	
}
		   
super_like(dating_app, root_directory)
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
		
		mouseClick "left", 860, 624
		sleep 1000
		
		; click away comment recommendation
		mouseClick "left", 865, 639
	}
	else if dating_app == "bumble"
	{
		mouseClick "left", 1208, 947
	}
	else if dating_app == "hinge"
    {
		like(dating_app, root_directory)
    }
	else if dating_app == "photofeeler"
	{
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 746, 347
		}
		else if random_number == 2
		{
			mouseClick "left", 746, 387
		}
		
		sleep 499
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 946, 343
		}
		else if random_number == 2
		{
			mouseClick "left", 938, 382
		}
		
		sleep 500
		
		
		random_number := Random(1, 2)
		
		if random_number == 1
		{
			mouseClick "left", 1079, 344
		}
		else if random_number == 2
		{
			mouseClick "left", 1194, 391
		}
		
		sleep 501
		
		; Click submit
		mouseClick "left", 1177, 730
	}
}

   
like(dating_app, root_directory)
{

	start_of_like_function_label:
	
	hinge_opener := ""
	
	if dating_app == "tinder"
	{		
		; Fix random profile popups
		loop 4
		{
			; Tinder arrow keys sometimes don't work on first try
			send "{down}"
			sleep 500
		}
		
		; Click like
		mouseClick "left", 932, 619
		
		sleep 3000
		
		; Click away super like upgrade popup
		mouseClick "left", 638, 597
	}
	else if dating_app == "bumble"
	{
		send "{right}"
	}
	else if dating_app == "hinge"
    {		
		; The 33 character limit for auto-type on hinge is crippling and makes nonsensical responses.
		; Saving this for future platforms that allows more characters for auto-type.
		; RunWait("bulk_image_ocr.exe")				
		; 
		; RunWait("chatgpt.exe")	
		; 
		; hinge_opener := FileRead("chatgpt_response.txt")
		
		hinge_opener := "Hey there, are you a fan of winter?"
		
		; Scroll up 
		; loop 21
		; {
		; 	click_and_drag(665, 211, 673, 644, 500)
		; }
		
		; Scroll down instead of scroll up (faster)
		Loop 2
		{
			loop 4
			{
				click_and_drag(669, 776, 665, 319, 1500)
			}
		}
		
		sleep 3000
		
		; Click like with footer
		mouseClick "left", 1191, 841
		sleep 1000
		; Click like without footer
		mouseClick "left", 1192, 780
		sleep 4000
		
		
		if !InStr(ocr(910, 963, 1006, 996, 500), "Cancel", 0)
		{
			; Click like with compatibility
			mouseClick "left", 1189, 895
			sleep 4000
		}	
		
		
		; Click "Add a comment"
		mouseClick "left", 812, 737
		
		sleep 3000
		
		send hinge_opener
		
		sleep 6000
		
		; Send like		
		mouseClick "left", 1018, 848
		sleep 1000
		
		; Click away send a rose instead
		mouseClick "left", 937, 971
		sleep 5000
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
		
		send "{^ down}l"
		send "{^ up}"
		send "https://www.photofeeler.com/vote/dating"
		send "{Enter}"
		
		
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
		
		mouseClick "left", 786, 618
	}
	else if dating_app == "bumble"
	{
		send "{left}"
	}
	else if dating_app == "hinge"
    {
		; Dislike button with footer
		mouseClick "left", 727, 922
		sleep 1000
		; Click dislike without footer
		mouseClick "left", 729, 982
		sleep 4000
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
