#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

take_screenshot(dating_app, windows_version, screenshot_directory)    
{
     if dating_app == "tinder"
     {
        winactivate "Tinder"
        
        loop 0
        {
            print_screen(962, 220, 1330, 753, windows_version, screenshot_directory)
            sleep 500
            send "{space}"
            sleep 500
        }        
     }
     else if dating_app == "bumble"
     {
        winactivate "Bumble"
        
        print_screen(580, 200, 1155, 927, windows_version, screenshot_directory)
        sleep 750
        send "{down}"
        send "{down}"
        sleep 500
        
        loop 5
        {
            print_screen(580, 200, 1155, 927, windows_version, screenshot_directory)
            sleep 750
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
    sleep 1000
    send "^a"
    send "{delete}"
    sleep 500
    WinClose "Screenshots"    
        
    ; This deactivates the dating website, make sure you re-activate them in other functions.
}
