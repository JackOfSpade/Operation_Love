

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


take_screenshot(dating_app, windows_version, screenshot_directory)    
{
	
     if dating_app == "tinder"
	 {
		loop 6
		{
			print_screen(1732, 364, 2572, 1578, windows_version, screenshot_directory)
			winactivate "Tinder"
			sleep 500
			send "{space}"
			sleep 500
		}		
	 }
	 else if dating_app == "bumble"
	 {
		print_screen(1035, 425, 2157, 1826, windows_version, screenshot_directory)
		winactivate "Bumble"
		sleep 500
		send "{down}"
		send "{down}"
		sleep 500
		
		loop 5
		{
			print_screen(1035, 425, 2157, 1826, windows_version, screenshot_directory)
			winactivate "Bumble"
			sleep 500
			send "{down}"
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
