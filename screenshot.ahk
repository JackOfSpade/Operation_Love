#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

take_screenshot(dating_app)    
{   
    if dating_app == "tinder"
    {
       winactivate "Tinder"
       
       loop 6
       {
           print_screen(675, 126, 1032, 555)
           sleep 500
           send "{space}"
           sleep 500
       }        
    }
    else if dating_app == "bumble"
    {
       winactivate "Bumble"
       
       print_screen(580, 200, 1155, 927)
       sleep 750
       send "{down}"
       send "{down}"
       sleep 500
       
       loop 5
       {
           print_screen(580, 200, 1155, 927)
           sleep 750
           send "{down}"
           sleep 500
       }
    }      
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		 
		send "{up}"
		send "{up}"
		send "{up}"
		send "{down}"
		send "{down}"
		sleep 2000	
		 
		Loop 10
		{
			print_screen(710, 0, 1208, 969)
			send "{down}"
			sleep 2000
		}
	}
}

clear_root_directory(path)
{
    send "<#r"
    sleep 500
    send "^a"
    send path
    send "{enter}"
    sleep 1000
    send "^a"
    send "{delete}"
    sleep 500
    WinClose "Screenshots"    
        
    ; This deactivates the dating website, make sure you re-activate them in other functions.
}
