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
	if dating_app == "tinder"
    {
		; Click profile
		MouseClick "left", 41, 159
		sleep 1000
			
		remainingSuperLikes := ocr(209, 381, 223, 397, 100)

		
		; Go back to discover
		MouseClick "left", 41, 159
		
		return remainingSuperLikes
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
				
		return remainingSuperLikes
	}
	else if dating_app == "hinge"
    {
		; Do it manually in standouts since it doesn't replenish (only 1 free rose/week
    }
	
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
		winactivate "AirDroid"
		
		loop 14
		{
			send "{up}"
		}
		
		mouseClick "left", 1149, 350
		
		winactivate "hinge"
		mouseClick "left", 805, 966
		
		; Cannot remove coding work from response
		A_Clipboard := "Do not answer with meta responses, only display the direct result. Use OCR to extract the text from all the attached image files. These text are from a dating profile for a woman. Give a clickable download link to download a text message that contains a short, witty text message (without emojis) to send that focuses on one topic in her profile. The message should not include an invitation to do anything."
		
		Send "^v"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		mouseClick "left", 739, 964
		sleep 1500
		
		; The folder should already be the correct one after decision.ahk runs
		send "+{Tab}"
		send "{LShift down}"
		
		; Select all profile pics
		Loop 10
		{
			send "{right}"
		}
		
		send "{LShift up}"		
		send "{Enter}"
		sleep 5000
		send "{Tab}"
		send "{Enter}"
		
		sleep 5000
		
		get_text(759, 807, 1488, 807, 100)
		
		; Wait for download
		while not InStr(A_Clipboard, "Download") and not InStr(A_Clipboard, "text") and not InStr(A_Clipboard, "txt")
		{			
			mouseClick "left", 716, 808
			get_text(759, 807, 1488, 807, 100)
			
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
		
		; Click download
		mouseClick "left", 850, 807
		sleep 5000
		
		send "^l"
        send "^a"
		A_Clipboard := root_directory
        send "^v"
		send "{Enter}"
		sleep 1000
		
		loop 3
		{
			send "{Tab}"
			sleep 500
		}
		
		send "!n"
		A_Clipboard := "hinge_opener"
		send "^v"
		
		send "{Enter}"
		
		; Replace existing
		send "{left}"
		send "{Enter}"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		sleep 2000
		
		hinge_opener := FileRead("hinge_opener.txt")
		; Remove all symbols that cannot be typed on a keyboard
		hinge_opener := RegExReplace(hinge_opener, "[^ -~]", "")
		
		MsgBox(hinge_opener)
		
		loop 12
		{
			send "{up}"
		}
		
		sleep 2000
		
		; Click like
		mouseClick "left", 1149, 442
		sleep 4000
		
		; Click "Add a comment"
		mouseClick "left", 843, 778
		sleep 5000
		
		send hinge_opener
		
		sleep 5000
		
		mouseClick "left", 755, 1004	
		mouseClick "left", 755, 1004

		sleep 3000
		
		mouseClick "left", 1050, 942
		
		sleep 3000
		
		mouseClick "left", 1050, 942
		
		sleep 3000
		
		mouseClick "left", 1172, 535
		
		sleep 3000
		
		; Click Done
		mouseClick "left", 1172, 535
		
		sleep 3000
		
		; Send like		
		mouseClick "left", 1037, 860
		
		sleep 3000
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
		mouseClick "left", 767, 906
    }
}
