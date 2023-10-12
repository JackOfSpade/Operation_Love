; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"



make_decision()
{	
	RunWait('powershell.exe -Command "Set-ExecutionPolicy Bypass -Scope Process; .\run_open_ai_clip.ps1"')
	
	bulk_images_ocr_text := FileRead("open_ai_clip_result.txt")

	; Extract first line string
	RegExMatch(bulk_images_ocr_text, "^([^\r\n]+)", &firstLine)
	adjective := firstLine[1]

	; Extract number from the second line
	RegExMatch(bulk_images_ocr_text, "Average probability of .*?:\s*(\d+\.\d+)", &probabilityMatch)
	probability := probabilityMatch[1]
		
    
    if adjective == "beautiful" and probability >= 0.8
    {
        decision := "super_like"
    }
    else if adjective == "beautiful"
    {
        decision := "like"
    }
    else if adjective == "ugly"
    {
        decision := "dislike"
    }
	else
	{
		msgbox('Error: adjective is not "beautiful" or "ugly".')
	}
    
    FileAppend "Analysis: " . bulk_images_ocr_text . "`ndecision: " . decision . "`n", ".\log.txt"
    
    return decision    
}

