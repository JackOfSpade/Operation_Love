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
}



remaining_super_likes(dating_app)
{
	remainingSuperLikes := 0
	if dating_app == "tinder"
    {
		; Click profile
		MouseClick "left", 41, 159
		sleep 5000
			
		remainingSuperLikes := ocr(209, 381, 223, 397, 100)

		
		; Go back to discover
		MouseClick "left", 41, 159
		sleep 5000
	}
	else if dating_app == "bumble"
	{		
		mouseMove 1139, 955
		sleep 1000
		remainingSuperLikes := ocr(1139, 955, 1198, 1011, 100)
		
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
		
		if InStr(ocr(390, 138, 1094, 570, 1000), "Upgrade", 0)
		{
			; Make super_like upgrade notice for popular profiles dissapear 
			send "{esc}"
			
			sleep 1000
			
			; Click like
			mouseClick "left", 932, 619
		}		
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
		
		hinge_opener := "Hi, what are you up to right now?"
		
		; Scroll up
		loop 16
		{
			click_and_drag(660, 521, 669, 828, 500)
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
		sleep 2000
		
		; Click away send a rose instead
		mouseClick "left", 938, 972
		sleep 4000
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
		mouseClick "left", 731, 915
		sleep 1000
		; Click dislike without footer
		mouseClick "left", 729, 982
		sleep 4000
		
    }
}
