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

	; Extract number from the second line
	RegExMatch(open_ai_clip_result, "Average probability of .*?:\s*(\d+\.\d+)", &probabilityMatch)
	beauty_probability := probabilityMatch[1]
	ugly_probability := probabilityMatch[2]
		
    
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
    
    FileAppend "Analysis: " . open_ai_clip_result . "`ndecision: " . decision . "`n", ".\log.txt"
    
    return decision    
}

