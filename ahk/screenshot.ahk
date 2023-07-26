

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


take_screenshot(dating_app, windows_version)    
{
	
     if dating_app == "tinder"
	 {
		loop 6
		{
			print_screen(1732, 364, 2572, 1578, windows_version)
			winactivate "Tinder"
			sleep 500
			send "{space}"
			sleep 500
		}		
	 }
         
}

clear_screenshot_directory(path)
{
	send "<#r"
	sleep 500
	send "^a"
	send path
	send "{enter}"
	sleep 500
	send "^a"
	send "{delete}"
	sleep 500
	WinClose "Screenshots"
}
