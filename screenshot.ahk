#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"



take_screenshot(dating_app)    
{   
	profile_text := ""
	
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
		send "{down}"
		Loop 8
		{
			print_screen(657, 106, 1260, 889)
			; MouseClickDrag "left", 956, 516, 956, 420, 100
			
			loop 4
			{
				click_and_drag(669, 576, 665, 119, 1000)
			}
			
			
			sleep 4000
		}
	}
}

clear_screenshot_directory(root_directory)
{
    send "<#r"
    sleep 500
	winActivate "Run"
	sleep 500
    send "^a"	
	A_Clipboard := root_directory . "/Screenshots"
    send "^v"
    send "{enter}"
    sleep 2000
	winActivate "Screenshots"
	sleep 500
    send "^a"
    send "{delete}"
    sleep 500
    winClose "Screenshots"   

	sleep 100
    A_Clipboard := ""
    sleep 100	
        
    ; This deactivates the dating website, make sure you re-activate them in other functions.
}
