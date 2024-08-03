; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"



make_decision()
{	
	RunWait('powershell.exe -Command ".\run_beauty_and_BMI_analysis.ps1"')
	
	text := FileRead("./beauty_and_BMI_analysis/beauty_and_BMI_analysis_result.txt")
	line_array := StrSplit(text, "`n", "`r")
	firstLine := line_array[1]
	
    
    if firstLine != "super_like" 
	and firstLine != "like" 
	and firstLine != "dislike" 
	and firstLine != "No face detected"
    {       
		msgbox('Error: firstLine is not "super_like", "like", "dislike" or "No face detected".')
	}
	
    return firstLine  
}

