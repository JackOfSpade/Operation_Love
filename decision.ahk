; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

upload_profile_pics(root_directory, resolution, dating_app)
{
	; Display locations
	if resolution == "1920x1080"
	{
		pixel_location := [[700, 841, 951, 840], [805, 851], [1212, 537], [1014, 569, 1040, 568], [1125, 740]]
	}
	else if resolution == "1366x768"
	{
		pixel_location := [[470, 586, 629, 606], [543, 598], [548, 413], [732, 414, 768, 413], [821, 558]]
	}


    ; https://hotchat3000.com/
    winactivate "Hot Chat"
    
    scores := []
    
	if dating_app != "hinge"
	{
		loop_count := 6
	}
	else
	{
		loop_count := 8
	}
	
    Loop loop_count  
    {
		start_of_loop:
		sleep 100
        A_Clipboard := ""
        sleep 100
		
		get_text(pixel_location[1][1], pixel_location[1][2], pixel_location[1][3], pixel_location[1][4], 100)
		
		while not InStr(A_Clipboard, "PHOTO")
		{		
			; clear highlights
			mouseClick "left", pixel_location[3][1], pixel_location[3][2]
			sleep 500
			
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
		
		sleep 2000
		
		; Click "Choose Photo" Button
        mouseClick "left", pixel_location[2][1], pixel_location[2][2]
		
        sleep 10000
		
		if dating_app == "tinder"
		{
			sleep 5000
		}
		
        send "^l"
        send "^a"
		A_Clipboard := root_directory . "/Screenshots"
        send "^v"
        send "{enter}"
		
		sleep 100
        A_Clipboard := ""
        sleep 100		
		
		sleep 1000
		
		if dating_app == "tinder"
		{
			sleep 3000
		}		
		
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
		
		sleep 1000
        
		; For some reason, hotchat3000 minimizes sometimes after this
        winactivate "Hot Chat"
		
		count := 0
		
        while not InStr(A_Clipboard, "ERROR") and not InStr(A_Clipboard, "CHATTING")
        {	
			send "^a"
			sleep 500
			send "^c"
			clipwait(1, 1)
			
			
			if count > 120
			{
				send "{F5}"
				sleep 62000
				mouseClick "left", pixel_location[5][1], pixel_location[5][2]
				sleep 2000
				goto start_of_loop
			}
			
			
			count++
			
        }  
		
        ; clear highlights
		mouseClick "left", pixel_location[3][1], pixel_location[3][2]
		sleep 500
		
		if InStr(A_Clipboard, "CHATTING")
		{
			
			sleep 100
			A_Clipboard := ""
			sleep 100
			
			sleep 2000
			
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
		
		; TEST
		sleep 3000
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



make_decision(root_directory, resolution, dating_app)
{
    combined_metric := upload_profile_pics(root_directory, resolution, dating_app)
    
    if (dating_app == "tinder" and combined_metric >= 8.5) or (dating_app == "bumble" and combined_metric >= 8)
    {
        decision := "super_like"
    }
    else if combined_metric >= 6
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
