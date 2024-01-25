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

	; Extract first line string
	RegExMatch(open_ai_clip_result, "^([^\r\n]+)", &firstLine)
	adjective := firstLine[1]

	; Extract number from the line about the beautiful girl
	RegExMatch(open_ai_clip_result, "Average probability of beautiful girl:\s*(\d+\.\d+)", &beautyMatch)
	beauty_probability := beautyMatch[1]

	; Extract number from the line about the ugly girl
	RegExMatch(open_ai_clip_result, "Average probability of ugly girl:\s*(\d+\.\d+)", &uglyMatch)
	ugly_probability := uglyMatch[1]		
    
    if adjective == "beautiful" and beauty_probability >= 0.8
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
	
    return decision    
}

