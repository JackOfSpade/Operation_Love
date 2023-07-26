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
	
	while not InStr(A_Clipboard, "UPlOAD")
	{
		get_text(1369, 740, 2139, 743, 0.5)
	}
	
	
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
		
		send "{right}"
		send "{left}"
		
		loop A_Index - 1
		{
			send "{right}"
		}
		
		sleep 500
		
		send "{enter}"
		
		winactivate "Hot Chat"
	
		
		while not InStr(A_Clipboard, "RERANK")
		{
			; Check for error text
			get_text(1105, 1656, 2376, 1653, 0.5)
		}
		
		sleep 100
		A_Clipboard := ""
		sleep 100
		
		; Check for error text
		get_text(1591, 748, 1852, 745, 0.5)
		
		
		if not InStr(A_Clipboard, "ERROR")
		{
			sleep 100
			A_Clipboard := ""
			sleep 100
			
			get_text(1841, 1139, 1915, 1143, 60*60*24)
			
			A_Clipboard := StrReplace(A_Clipboard, A_Space, "")
			
			if IsNumber(A_Clipboard)
			{
				scores.push(A_Clipboard)
			}
			
			mouseClick "left", 1741, 1668
			
		}
		else
		{
			mouseClick "left", 1721, 1680
		}		
		
		
		sleep 100
		A_Clipboard := ""
		sleep 100
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