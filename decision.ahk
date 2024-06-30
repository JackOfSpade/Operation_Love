; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"



make_decision()
{	
	RunWait('powershell.exe -Command ".\run_open_ai_clip.ps1"')
	
	open_ai_clip_result := FileRead("open_ai_clip_result.txt")

	firstLine := ""
	; Extract first line string
	matchResult := RegExMatch(open_ai_clip_result, "^(.+)", &firstLine)
	adjective := firstLine[1]

	beautyMatch := ""
	; Extract beauty probability
	matchResult := RegExMatch(open_ai_clip_result, "Beautiful probability: (\d+\.\d+)", &beautyMatch)
	beautiful_probability := beautyMatch[1]

	uglyMatch := ""
	; Extract ugly probability
	RegExMatch(open_ai_clip_result, "Ugly probability: (\d+\.\d+)", &uglyMatch)
	ugly_probability := uglyMatch[1]	
    
    if adjective == "beautiful" and beautiful_probability >= 0.95
    {
        decision := "super_like"
    }
    else if adjective == "beautiful" or adjective == "neutral"
    {
        decision := "like"
    }
    else if adjective == "ugly"
    {
        decision := "dislike"
    }
	else
	{
		msgbox('Error: adjective is not "beautiful", "ugly" or "neutral".')
	}
	
	file_name := "decision: " . decision . "adjective: " . adjective . "beautiful_probability: " . beautiful_probability . "ugly_probability: " . ugly_probability
	
    return [decision, adjective, file_name]    
}

