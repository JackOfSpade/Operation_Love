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
		MouseClick "left", 50, 162
		sleep 1000
			
		remainingSuperLikes := ocr(227, 381, 250, 399, 100)

		
		; Go back to discover
		MouseClick "left", 50, 162
		
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
	else if dating_app == "hinge"
    {
		
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
		; Check location in profile ---> super like dialog
		mouseClick "left", 964, 841
	}
	else if dating_app == "bumble"
	{
		winactivate "Bumble"
		send "{right}"
	}
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		mouseClick "left", 1151, 646
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
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		mouseClick "left", 767, 906
    }
}
