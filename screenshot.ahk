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
       
	   ; For doing multi-photo iterations
       ; loop 6
       ; {
       ;     print_screen(962, 215, 1339, 718)
       ;     sleep 500
		 ;   ; Space stops working for some reason sometimes
       ;     
		 ;   ; Go to next picture
		 ;   mouseClick "left", 1318, 509
       ;     sleep 1000
       ; }        
	   
	   ; For doing one-photo iterations
	   loop 1
       {
           print_screen(962, 215, 1339, 718)
           sleep 500
       }    
    }
    else if dating_app == "bumble"
    {
       winactivate "AirDroid"
       
       count := 0
       
	   ; For doing multi-photo iterations
       ; loop 5
       ; {
       ;     print_screen(492, 193, 853, 628)
       ;     sleep 500
       ;     send "{down}"
		 ;   
		 ;   if count == 0
		 ;   {
		 ;		send "{down}"
		 ;		count++
		 ;   }
		 ;   
       ;     sleep 500
       ; }
	   
	   ; For doing one-photo iterations
	   loop 1
       {
           print_screen(658, 108, 1258, 999)
           sleep 500
       }
    }      
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		
		; For doing multi-photo iterations
		; loop 7
		; {
		; 	print_screen(657, 106, 1260, 889)
		; 	
		; 	loop 4
		; 	{
		; 		click_and_drag(669, 776, 665, 319, 1500)
		; 	}
		; }
		
		; For doing one-photo iterations
		loop 1
		{
			print_screen(657, 106, 1260, 889)
		}
	}
	else if dating_app == "okcupid"
	{
		winactivate "OkCupid"
		
		; For doing multi-photo iterations
		; loop 6
		; {
	
		; }
		
		; For doing one-photo iterations
		loop 1
		{
			print_screen(232, 364, 1088, 640)
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
