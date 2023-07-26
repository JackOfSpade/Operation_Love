; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.



#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


upload_profile_pics(screenshot_directory, upload_time_in_ms)
{
	; https://hotchat3000.com/
	winactivate "Hot Chat"
	send "{home}"
	sleep 500
	
	scores := []
	
	Loop 6  
	{
		mouseClick "left", 1446, 1683
		sleep 1500
		send "^l"
		send "^a"
		send screenshot_directory
		send "{enter}"
		
		n := 3 + A_Index
		
		loop 4
		{
			sleep 500
			send "{Tab}"
		}
		
		sleep 500
		send "{right}"
		sleep 500
		send "{left}"
		
		loop A_Index - 1
		{
			send "{right}"
		}
		
		sleep 500
		
		send "{enter}"
		
		; This sleep depends on the upload time of the internet.
		sleep upload_time_in_ms
		
		; Check for error text
		mouseMove 1591, 748
		send "{LButton down}"
		sleep 100
		mouseMove 1852, 745
		send "{LButton up}"
		sleep 100
		send "^c"
		
		clipwait(1)
		
		
		if not InStr(A_Clipboard, "ERROR")
		{
			A_Clipboard := ""
			
			; Wait for score to load
			sleep 3500 
			
			mouseMove 1841, 1139
			send "{LButton down}"
			sleep 100
			mouseMove 1915, 1143
			send "{LButton up}"
			sleep 100
			send "^c"
			
			clipwait(1)
			
			if IsNumber(A_Clipboard)
			{
				scores.push(A_Clipboard)
			}
			
			mouseClick "left", 1708, 1682
			
		}
		else
		{
			mouseClick "left", 1721, 1680
		}		
		
		
		A_Clipboard := ""
		sleep 500
	}
	
	
	
	
	
	
	
	
	
	
	
	

	; https://photo-ranker.com/
	;winactivate "Photo Ranker"
	;send "{home}"
	;sleep 500
	;mouseClick "left", 1715, 1440 
	;sleep 500
	;send "^l"
	;send "^a"
	;send screenshot_directory 
	;
	;loop 4
	;{
	;	sleep 500
	;	send "{tab}"
	;}
	;
	;send "^a"
	;sleep 500
	;send "{enter}"
	;sleep 500
	;mouseClick "left", 1730, 2000
	;
	;loop 20
	;{
	;	send "{down}"
	;}
	;
	;
	;progress := ""
	;
	;while not InStr(progress, "00")
	;{
	;	progress := ocr(1683, 322, 1762, 349)
	;}
	;
	;; Click show score
	;mouseClick "left", 1720, 460
	;
	;scores := []
	;
	;ocr(838, 637, 965, 689)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	;
	;ocr(1546, 639, 1675, 691)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	;
	;ocr(2255, 641, 2378, 686)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	;
	;ocr(2971, 636, 3098, 686)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	;
	;ocr(1548, 1665, 1679, 1718)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	;
	;ocr(2257, 1667, 2387, 1719)
	;
	;if IsNumber(A_Clipboard)
	;{
	;	scores.push(A_Clipboard)
	;}
	
	
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



make_decision(screenshot_directory, upload_time_in_ms)
{
	combined_metric := upload_profile_pics(screenshot_directory, upload_time_in_ms)
	
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