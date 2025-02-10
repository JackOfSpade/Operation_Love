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
	   
	   print_screen(960, 220, 1340, 723)
       sleep 500  
    }
    else if dating_app == "bumble"
    {
       winactivate "Bumble"
       
       count := 0
	   
	   print_screen(570, 195, 1162, 930)
       sleep 500
    }      
	else if dating_app == "hinge"
    {
		winactivate "AirDroid"
		
		print_screen(711, 143, 1208, 975)
	}
	else if dating_app == "okcupid"
	{
		winactivate "OkCupid"
		
		print_screen(225, 431, 522, 712)
		
	}
	else if dating_app == "2redbeans"
	{
		winactivate "2RedBeans"
		
		print_screen(173, 178, 279, 276)
	}
	else if dating_app == "photofeeler"
    {
		; don't winactivate "Vote" because continous refresh make loading time inconsistent
       
		print_screen(105, 320, 555, 725)		
	}
}

move_screenshot(root_directory) 
{
    ; Define source and destination folders
    screenshot_folder_path := root_directory . "/Screenshots" 
    processed_profiles_folder_path := root_directory . "/processed_profiles"
    
	DirMove screenshot_folder_path, processed_profiles_folder_path, 2
	DirCreate screenshot_folder_path
}
