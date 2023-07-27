; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

upload_profile_pics(screenshot_directory)
{
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
			get_text(700, 841, 951, 840, 100)
		}
		
		sleep 100
		A_Clipboard := ""
		sleep 100
		
		; Click "Choose Photo" Button
        mouseClick "left", 805, 851
        sleep 1500
        send "^l"
        send "^a"
        send screenshot_directory
        send "{enter}"
        n := 3 + A_Index
        
        loop 4
        {      
            send "{Tab}"
			sleep 500
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
		mouseClick "left", 1212, 537
		sleep 500
		
		if InStr(A_Clipboard, "CHATTING")
		{
			
			sleep 100
			A_Clipboard := ""
			sleep 100
			
			get_text(1014, 569, 1040, 568, 60*60*24)
			
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



make_decision(screenshot_directory)
{
    combined_metric := upload_profile_pics(screenshot_directory)
    
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
