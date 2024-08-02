; Decision Module: This module generates an attractiveness score
; and makes a decision of like or dislike based on that.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"



make_decision()
{	
	RunWait('powershell.exe -Command ".\run_beauty_and_BMI_analysis.ps1"')
	
	beauty_and_BMI_analysis := FileRead("beauty_and_BMI_analysis_result.txt")

	firstLine := ""
	; Extract first line string
	matchResult := RegExMatch(beauty_and_BMI_analysis, "^(.+)", &firstLine)
	
    
    if firstLine != "super-like" and firstLine != "like" and firstLine != "dislike"
    {       
		msgbox('Error: firstLine is not "super-like", "like" or "dislike".')
	}
	
    return firstLine  
}

