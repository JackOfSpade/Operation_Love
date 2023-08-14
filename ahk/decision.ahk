; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

upload_profile_pics(screenshot_directory, resolution)
{
	; Display locations
	if resolution == "1920x1080"
	{
		pixel_location := [[700, 841, 951, 840], [805, 851], [1212, 537], [1014, 569, 1040, 568]]
	}
	else if resolution == "1366x768"
	{
		pixel_location := [[470, 586, 629, 606], [543, 598], [548, 413], [732, 414, 768, 413]]
	}


    ; https://hotchat3000.com/
    winactivate "Hot Chat"
    
    
    scores := []
    
    Loop 6  
    {
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		while not InStr(A_Clipboard, "PHOTO")
		{
			; weird pop-up glitch on company PC win 10
			if InStr(A_Clipboard, "STAR")
			{
				send "{tab}"
				send "{tab}"
				send "{tab}"
				send "{enter}"				
			}
			
			; "Choose Photo" Button scan
			get_text(pixel_location[1][1], pixel_location[1][2], pixel_location[1][3], pixel_location[1][4], 100)
		}
		
		sleep 100
		A_Clipboard := ""
		sleep 100
		
		; Click "Choose Photo" Button
        mouseClick "left", pixel_location[2][1], pixel_location[2][2]
		
        sleep 1500
		
        send "^l"
        send "^a"
        send screenshot_directory
        send "{enter}"
		
		sleep 1000
		
		send "!n"		
		send "+{Tab}"
        
        send "{right}"
        send "{left}"
        
        loop A_Index - 1
        {
            send "{right}"
        }
        
        sleep 500
        
        send "{enter}"
        
        winactivate "Hot Chat"
		
		sleep 100
        A_Clipboard := ""
        sleep 100
		
        while not InStr(A_Clipboard, "ERROR") and not InStr(A_Clipboard, "CHATTING")
        {	
			send "^a"
			sleep 500
			send "^c"
			clipwait(1, 1)
			
        }  
		
        ; click away highlights
		mouseClick "left", pixel_location[3][1], pixel_location[3][2]
		sleep 500
		
		if InStr(A_Clipboard, "CHATTING")
		{
			
			sleep 100
			A_Clipboard := ""
			sleep 100
			
			; Get score
			get_text(pixel_location[4][1], pixel_location[4][2], pixel_location[4][3], pixel_location[4][4], 60*60*24)
			
			A_Clipboard := StrReplace(A_Clipboard, A_Space, "")
			
			if IsNumber(A_Clipboard)
			{
				scores.push(A_Clipboard)
			}
			
			; Go to upload page
			send "{tab}"
			send "{tab}"
			send "{tab}"
			send "{enter}"
		}
		else if InStr(A_Clipboard, "ERROR")
		{
			; Go to upload page
			send "{tab}"
			send "{enter}"
			
		}
        
        sleep 100
        A_Clipboard := ""
        sleep 100
    }

    score := 0

    ; Iterate over each score in the array
    Loop scores.Length 
    {
        ; Get the current score
        score := Max(score, scores[A_Index])
    }

    ; Calculate combined metric
    combined_metric := score

    ; String representation of array
    scores_string := ""

    if scores.Length > 0
    {
        Loop scores.Length
        {
            scores_string .= scores[A_Index] . ", "
        }

        ; Remove trailing comma and space
        scores_string := SubStr(scores_string, 1, -2)
    }

    FileAppend "scores: " . scores_string . "`ncombined_metric: " . combined_metric . "`n", ".\log.txt"

    return combined_metric

}



make_decision(screenshot_directory, resolution)
{
    combined_metric := upload_profile_pics(screenshot_directory, resolution)
    
    if combined_metric >= 8.5
    {
        decision := "super_like"
    }
    else if combined_metric >= 6.5
    {
        decision := "like"
    }
    else
    {
        decision := "dislike"
    }
    
    FileAppend "decision: " . decision . "`n", ".\log.txt"
    
    return decision    
}
