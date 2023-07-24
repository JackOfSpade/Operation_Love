

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


take_screenshot(dating_app)    
{
     if dating_app == "tinder"
	 {
		loop 6
		{
			print_screen(1744, 352, 2585, 1215)		
			send "{space}"
		}		
	 }
         
}

a::
{
	take_screenshot("tinder") 
}