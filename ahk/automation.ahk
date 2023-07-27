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
}



remaining_super_likes(dating_app)
{
	if dating_app == "tinder"
    {
		; Click profile
		MouseClick "left", 86, 181
		sleep 1000
			
		remainingSuperLikes := ocr(283, 425, 309, 449, 100)

		
		; Go back to discover
		MouseClick "left", 53, 180
		
		return remainingSuperLikes
	}
	else if dating_app == "bumble"
	{		
		remainingSuperLikes := ocr(1162, 902, 1251, 980, 100)
		
		if !IsNumber(remainingSuperLikes)
		{
			remainingSuperLikes := 0
		}
				
		return remainingSuperLikes
	}

	
}
	
	
	   
super_like(dating_app)
{
	if dating_app == "tinder"
	{
		winactivate "Tinder"
		send "{up}"
		sleep 1000
		send "{down}"
		sleep 1000
		send "{enter}"
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
		mouseClick "left", 1208, 947
	}
}
   
like(dating_app)
{
	
	
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
		mouseClick "left", 960, 778
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
		send "{right}"
	}
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
}
