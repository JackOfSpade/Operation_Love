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
    }
	else if dating_app == "bumble"
    {
		winactivate "Bumble"
    }
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
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
		; Do it manually in standouts since it doesn't replenish (only 1 free rose/week
    }
	
	return remainingSuperLikes
	
}
	
	
	   
super_like(dating_app, root_directory)
{
	if dating_app == "tinder"
	{
		winactivate "Tinder"
		send "{up}"
		sleep 1000
		send "{down}"
		sleep 1000
		send "{enter}"
		sleep 1000
		; click away comment recommendation
		send "{esc}"
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
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
		winactivate "Tinder"
		send "{up}"
		sleep 1000
		send "{down}"
		sleep 1000
		send "{right}"
		
		sleep 1000
		; Click away super_like upgrade notice for popular profiles
		send "{esc}"
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
		send "{right}"
	}
	else if dating_app == "hinge"
    {
		RunWait("bulk_image_ocr.exe")		
		bulk_images_ocr_text := FileRead("bulk_images_ocr_text.txt")
		; Remove all symbols that cannot be typed on a keyboard
		bulk_images_ocr_text := RegExReplace(bulk_images_ocr_text, "[^ -~\r\n]", "")		
		
		winactivate "hinge"
		sleep 1000
		mouseClick "left", 860, 968
		
		sleep 500
		
		; Cannot remove coding work from response
		A_Clipboard := "The following text is from a dating profile for a woman: " . bulk_images_ocr_text . " ====== End of dating profile ======= Give a download link for a message that is a funny commentary on one specific detail from her profile. The message should not be more than 33 characters long. Do not include an invitation to do anything; If the generated message has an ending period, remove it. Avoid word play. Avoid messages that uses exclamation marks. It shows too much enthusiasm for an initial message. Do not answer prompt questions that the girl has written on her profile without at least subtly referring back to it because she won't know what prompt we're referring to." 
		
		Send "^v"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 500
		
		; Submit it
		mouseClick "left", 1458, 967
		
		sleep 4000				
			
		mouseClick "left", 826, 968
		
		sleep 500
				
		; For some reason ChatGPT refuses to give the download link the first time sometimes
		A_Clipboard := "Make sure you're processing the most recent profile given. The message should not be more than 33 characters long. Reassess whether the message is just an reiteration of what she has already said in her profile. If so, rewrite. Give me the download link only without any other explanations, meta responses or any other text"		
		Send "^v"
		
		sleep 1000
		
		while not InStr(ocr(756, 955, 797, 974, 100), "Send")
        {				
		
			; Check if limit is reached
			get_text(779, 801, 827, 801, 1000)
			
			if InStr(A_Clipboard, "default")
			{
				send "{F5}"		
				sleep 5000
				goto start_of_like_function_label
			}	

			; Click regenerate
			mouseClick "left", 1086, 956
			
			; click textbox
			mouseClick "left", 826, 968
		
			sleep 500
			send "^a"
					
			; For some reason ChatGPT refuses to give the download link the first time sometimes
			A_Clipboard := "Make sure you're processing the most recent profile given. Reassess whether the message is just an reiteration of what she has already said in her profile. If so, rewrite. Give me the download link only without any other explanations, meta responses or any other text"		
			Send "^v"
			
			sleep 1000
		
			; Click submit
			mouseClick "left", 1458, 967	
        }  
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 6000
		
		; Wait for download
		while not InStr(A_Clipboard, "Download") and not InStr(A_Clipboard, "text") and not InStr(A_Clipboard, "txt")
		{	
			mouseClick "left", 716, 808
			
			loop 10
			{
				send "{WheelDown}"
				sleep 100
			}
			
			sleep 1000
			
			; Check if limit is reached
			get_text(779, 801, 827, 801, 1000)
			
			if InStr(A_Clipboard, "default")
			{
				send "{F5}"		
				sleep 5000
				goto start_of_like_function_label
			}
		} 
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 2000
		
		; Click download link
		mouseClick "left", 770, 805
		; Alternative spot
		mouseClick "left", 1164, 807
		
		sleep 5000
		
		send "^l"
        send "^a"
		A_Clipboard := root_directory
        send "^v"
		send "{Enter}"
		sleep 1000
		
		loop 4
		{
			send "{Tab}"
			sleep 500
		}
		
		send "!n"
		A_Clipboard := "hinge_opener"
		send "^v"
		
		send "{Enter}"
		sleep 500
		; Replace existing
		send "{left}"
		sleep 500
		send "{Enter}"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 3000
		
		hinge_opener := FileRead("hinge_opener.txt")
		; Remove all symbols that cannot be typed on a keyboard
		hinge_opener := RegExReplace(hinge_opener, "[^ -~]", "")
		
		; msgBox(hinge_opener)
		
		winactivate "AirDroid"
		
		loop 10
		{
			click_and_drag(668, 480, 667, 833, 500)
		}
		
		sleep 3000
		
		; Click like with footer
		mouseClick "left", 1191, 841
		sleep 1000
		; Click like without footer
		mouseClick "left", 1192, 780
		sleep 2000
		
		; Click "Add a comment"
		mouseClick "left", 813, 778
		
		sleep 1000
		
		SendInput hinge_opener
		
		sleep 6000
		
		; Send like		
		mouseClick "left", 1013, 875
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
		winactivate "Tinder"
		send "{up}"
		sleep 1000
		send "{down}"
		sleep 1000
		send "{left}"
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
		send "{left}"
	}
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		
		; Dislike button with footer
		mouseClick "left", 727, 920
		sleep 1000
		; Click dislike without footer
		mouseClick "left", 729, 982
		sleep 2000
		
    }
}
