; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.



#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


upload_profile_pics()
{
	winactivate "Photo Ranker"
	send "{home}"
	sleep 500
	mouseClick "left", 1715, 1440 
	sleep 500
	send "^l"
	send "^a"
	send "C:\Users\Shadow\Pictures\Screenshots" 
	
	loop 4
	{
		sleep 500
		send "{tab}"
	}
	
	send "^a"
	sleep 500
	send "{enter}"
	sleep 500
	mouseClick "left", 1730, 2000
	
	loop 20
	{
		send "{down}"
	}
	
	
	progress := ""
	
	while not InStr(progress, "00")
	{
		progress := ocr(1683, 322, 1762, 349)
	}
	
	; Click show score
	mouseClick "left", 1720, 460
	
	scores := []
	
	ocr(838, 637, 965, 689)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	ocr(1546, 639, 1675, 691)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	ocr(2255, 641, 2378, 686)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	ocr(2971, 636, 3098, 686)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	ocr(1548, 1665, 1679, 1718)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	ocr(2257, 1667, 2387, 1719)
	
	if IsNumber(A_Clipboard)
	{
		scores.push(A_Clipboard)
	}
	
	; Initialize variables for sum, sum of squares and other stats
	sum := 0
	sum_of_squares := 0

	; Iterate over each score in the array
	Loop scores.Length 
	{
		; Get the current score
		score := scores[A_Index]

		; Compute sum and sum of squares for standard deviation calculation
		sum := sum + score
		sum_of_squares := sum_of_squares + (score * score)
	}

	; Calculate the average
	if scores.Length > 0
	{
		average := sum / scores.Length
	}
	else
	{
		average := 0
	}
	

	; Calculate standard deviation
	n := scores.Length
	if (n > 1)
	{
		variance := (sum_of_squares - ((sum * sum) / n)) / (n - 1)
		std_deviation := sqrt(variance)
	}
	else
	{
		std_deviation := 0
	}

	; Define k (how heavily the standard deviation will penalize the average score)
	k := 0.5

	; Calculate combined metric
	combined_metric := average - (k * std_deviation)

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

	FileAppend "scores: " . scores_string . "`naverage: " . average . "`nstd_deviation: " . std_deviation . "`ncombined_metric: " . combined_metric . "`n", ".\log.txt"

	return combined_metric

}



make_decision()
{
	combined_metric := upload_profile_pics()
	
	if combined_metric >= 8.5
	{
		decision := "super_like"
	}
	else if combined_metric >= 7
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