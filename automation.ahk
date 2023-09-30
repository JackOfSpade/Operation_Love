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
		; Do it manually in standouts since it doesn't replenish (only 1 free rose/week
    }
	
	return remainingSuperLikes
	
}
	
	
	   
super_like(dating_app, root_directory)
{
	if dating_app == "tinder"
	{
		winactivate "Tinder"
		sleep 1000
		
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
		sleep 1000
		
		; Fix random profile popups
		loop 4
		{
			; Tinder arrow keys sometimes don't work on first try
			send "{down}"
			sleep 500
		}
		
		mouseClick "left", 932, 619
		
		sleep 3000
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
		
		sleep 1000
		
		; Cannot remove coding work from response
		A_Clipboard := "The following text is from a dating profile for a woman: " . bulk_images_ocr_text . " ====== End of dating profile ======= Generate an opening message based on one specific detail from her profile. The message should not be more than 33 characters long. Do not include an invitation to do anything; If the generated message has an ending period, remove it. Avoid word play. Avoid messages that uses exclamation marks. Do not answer prompt questions that the girl has written on her profile without at least subtly referring back to it because she won't know what prompt we're referring to." 
		
		Send "^v"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 500
		
		; Submit it
		mouseClick "left", 1450, 989
		
		sleep 10000				
			
		; Some reason you gotta click multiple times if ChatGPT is still processing
		mouseClick "left", 824, 967		
		sleep 500		
		mouseClick "left", 824, 967		
		sleep 500
		mouseClick "left", 824, 967		
		sleep 500
		
		; For some reason ChatGPT refuses to give the download link the first time sometimes
		A_Clipboard := "Put the message in a download link without explanations. Make sure the message is based on the most recent profile given. If the message is more than 33 characters long, rewrite so that it's less than or equal to 33 characters long."
		
		Send "^v"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 500
		
		; Submit it
		mouseClick "left", 1450, 989
		
		sleep 10000
		
		while not InStr(ocr(755, 971, 798, 997, 100), "Send")
        {		

			mouseMove 558, 753
			loop 10
			{
				send "{WheelDown}"
				sleep 100
			}
			
			; Check if limit is reached
			get_text(754, 806, 827, 801, 1000)
			
			if InStr(A_Clipboard, "default")
			{
				send "{F5}"		
				sleep 5000
				goto start_of_like_function_label
			}	
			
			sleep 100
			A_Clipboard := ""
			sleep 100

			; Click regenerate
			mouseClick "left", 1086, 956
			
			; click textbox
			mouseClick "left", 826, 968
			
			; Submit it
			mouseClick "left", 1450, 989
			
			sleep 1000
        }  
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 3000
		; Click submit again, random glitch that doesn't submit the first time and the entered text pops up again
		mouseClick "left", 1450, 989
		sleep 5000
		
		; Wait for download
		while not InStr(A_Clipboard, "Download") or InStr(A_Clipboard, "sandbox")
		{		
			mouseMove 558, 753
			loop 10
			{				
				send "{WheelDown}"
				sleep 100
			}
			
			; Check if limit is reached
			get_text(754, 806, 827, 801, 1000)
			
			if InStr(A_Clipboard, "default")
			{
				send "{F5}"		
				sleep 5000
				goto start_of_like_function_label
			}
			
			sleep 100
			A_Clipboard := ""
			sleep 100
			
			; Reset highlights
			mouseClick "left", 479, 807
			
			; Check for download link
			get_text(761, 807, 1500, 807, 1000)
			
			if InStr(A_Clipboard, "sandbox") and not InStr(A_Clipboard, "Download")
			{
				; click textbox
				mouseClick "left", 826, 968
			
				sleep 500
				send "^a"
						
				; When ChatGPT doesn't give a proper download link
				A_Clipboard := "That's not a download link"
				
				Send "^v"
		
				sleep 100
				A_Clipboard := ""
				sleep 100
				
				sleep 500
				
				; Submit it
				mouseClick "left", 1450, 989
		
				sleep 1000
			}
		} 
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 2000
		
		
		; Click download link (alternative spot)
		mouseClick "left", 1164, 807
		sleep 500
		; Click download link (alternative spot)
		mouseClick "left", 957, 807
		sleep 500
		; Click download link (alternative spot)
		mouseClick "left", 892, 807
		sleep 500
		; Click download link
		mouseClick "left", 770, 807
		
		
		sleep 7000
		
		send "!n"
		A_Clipboard := "hinge_opener"
		send "^v"
		
		sleep 500
		
		send "^l"
        send "^a"
		A_Clipboard := root_directory
        send "^v"
		send "{Enter}"
		sleep 500
		send "{Enter}"
		sleep 1000
		; Replace existing
		send "{left}"
		sleep 500
		send "{Enter}"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 3000
		
		hinge_opener := FileRead("hinge_opener.txt")
		
		; msgBox(hinge_opener)
		
		winactivate "AirDroid"
		
		sleep 500
		
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
		sleep 4000
		
		; Click "Add a comment"
		mouseClick "left", 813, 778
		
		sleep 3000
		
		send hinge_opener
		
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
		sleep 1000
		
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
		winactivate "Bumble"
		send "{left}"
	}
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		
		sleep 500
		
		; Dislike button with footer
		mouseClick "left", 731, 915
		sleep 1000
		; Click dislike without footer
		mouseClick "left", 729, 982
		sleep 4000
		
    }
}
