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
           print_screen(675, 126, 1032, 504)
           sleep 500
		   ; Space stops working for some reason sometimes
           
		   mouseClick "left", 1036, 356
           sleep 1000
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
		Loop 8
		{
			print_screen(657, 106, 1260, 889)
			
			loop 4
			{
				click_and_drag(669, 776, 665, 319, 1500)
			}
			
			
			sleep 4000
		}
	}
	else if dating_app == "photofeeler"
    {
		winactivate "Vote"
       
		print_screen(105, 320, 555, 725)		
	}
}

clear_screenshot_directory(root_directory, dating_app)
{
    send "<#r"
    sleep 500
	
	if dating_app == "tinder"
	{
		sleep 3000
	}
	
	winActivate "Run"
	sleep 1000
	A_Clipboard := root_directory . "/Screenshots"
	sleep 100
    send "^v"
	sleep 1000
    send "{enter}"
    sleep 3000
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
