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
}



remaining_super_likes(dating_app)
{
	if dating_app == "tinder"
    {
		; Click profile
		MouseClick "left", 156, 362
		sleep 1000
			
		remainingSuperLikes := ocr(510, 851, 557, 898)
		
		; Go back to discover
		MouseClick "left", 96, 360
		
		return remainingSuperLikes
	}

	
}
	
	
   
like(dating_app)
{
	winactivate "Tinder"
	
	if dating_app == "tinder"
	{
		send "{up}"
		sleep 1000
		send "{right}"
	}
}


   
super_like(dating_app)
{
	winactivate "Tinder"
	
	if dating_app == "tinder"
	{
		send "{up}"
		sleep 1000
		send "{enter}"
	}
}

   
dislike(dating_app)
{
	winactivate "Tinder"
	
	if dating_app == "tinder"
	{
		send "{up}"
		sleep 1000
		send "{left}"
	}
}
