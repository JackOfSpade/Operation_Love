#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

take_screenshot(dating_app, screenshot_directory)    
{
     if dating_app == "tinder"
     {
        winactivate "Tinder"
        
        loop 6
        {
            print_screen(962, 220, 1330, 753, screenshot_directory)
            sleep 500
            send "{space}"
            sleep 500
        }        
     }
     else if dating_app == "bumble"
     {
        winactivate "Bumble"
        
        print_screen(580, 200, 1155, 927, screenshot_directory)
        sleep 750
        send "{down}"
        send "{down}"
        sleep 500
        
        loop 5
        {
            print_screen(580, 200, 1155, 927, screenshot_directory)
            sleep 750
            send "{down}"
            sleep 500
        }
     }       
    if dating_app == "tinder"
    {
       winactivate "Tinder"
       
       loop 6
       {
           print_screen(962, 220, 1330, 753, screenshot_directory)
           sleep 500
           send "{space}"
           sleep 500
       }        
    }
    else if dating_app == "bumble"
    {
       winactivate "Bumble"
       
       print_screen(580, 200, 1155, 927, screenshot_directory)
       sleep 750
       send "{down}"
       send "{down}"
       sleep 500
       
       loop 5
       {
           print_screen(580, 200, 1155, 927, screenshot_directory)
           sleep 750
           send "{down}"
           sleep 500
       }
    }      
	else if dating_app == "hinge"
    {
		 print_screen(729, 231, 1189, 689, screenshot_directory)
		 sleep 750
		 send "{down}"
		 sleep 500
		 ; text
		 print_screen(736, 485, 1189, 796, screenshot_directory)
		 send "{down}"
		 send "{down}"
		 send "{up}"
		 send "{up}"
		 print_screen(730, 428, 1187, 885, screenshot_directory)
		 send "{down}"
		 send "{down}"
		 ; text
		 print_screen(732, 377, 1188, 740, screenshot_directory)
		 send "{down}"
		 print_screen(729, 350, 1187, 807, screenshot_directory)
		 send "{down}"
		 ; text
		 print_screen(678, 337, 1186, 792, screenshot_directory)
		 send "{down}"
		 print_screen(731, 349, 1188, 805, screenshot_directory)
		 send "{down}"
		 print_screen(731, 337, 1189, 794, screenshot_directory)
		 
		 
	}
}

clear_screenshot_directory(path)
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
