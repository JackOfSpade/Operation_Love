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
		print_screen(711, 143, 1208, 975)
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
			print_screen(225, 431, 522, 712)
		}
		
	}
	else if dating_app == "photofeeler"
    {
		; don't winactivate "Vote" because continous refresh make loading time inconsistent
       
		print_screen(105, 320, 555, 725)		
	}
}

move_screenshot(root_directory, how_much_like, new_file_name) 
{
    ; Define source and destination folders
    screenshot_file_path := root_directory . "/Screenshots/screenshot.png" 
    processed_profiles_folder_path := root_directory . "/processed_profiles"

    super_liked_folder_path := processed_profiles_folder_path . "/super_liked" 
    liked_folder_path := processed_profiles_folder_path . "/liked" 
    disliked_folder_path := processed_profiles_folder_path . "/disliked"   

    if how_much_like == "super_like"
    {
        destination_file_path := super_liked_folder_path . "/" . new_file_name . ".png"			  
    }
    else if how_much_like == "like"
    {
        destination_file_path := liked_folder_path . "/" . new_file_name . ".png"
    }
    else
    {
        destination_file_path := disliked_folder_path . "/" . new_file_name . ".png"
    }
	
    ; Move the file to the appropriate folder
    FileMove screenshot_file_path, destination_file_path
}
