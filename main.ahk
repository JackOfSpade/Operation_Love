; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; 100% zoom in resolution settings
; Laptop must be plugged in or else the save/file dialog will lag.
; Greenshot: set output location to screenshot folder
; 			 set capture region to f11
; Set-ExecutionPolicy Unrestricted -Scope LocalMachine
; Unblock-File -Path "C:\Users\Dell\Desktop\GitHub\Operation_Love\open_ai_clip\venv\Scripts\activate.ps1"


; Warnings:
; log.txt will not log if you have it open in notepad++.
; Running this while having another one running inside Shadow PC will cause the one outside to crash due to clipboard conflicts.
; If AirDroid, tap become long presses where the context menu pops up, click "Hoykeys" in its menu and click "Switch input method"


#include automation.ahk
#include screenshot.ahk
#include decision.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


main(dating_app, root_directory, resolution)
{
	; Clear the log
	file := FileOpen(".\log.txt", "w")
	file.close()
	
	; test
	; hinge_opener := like(dating_app, root_directory)
	
	super_likes := 0
	
	navigate_to_discover(dating_app)
	super_likes := remaining_super_likes(dating_app)
	
	if super_likes == "O"
	{
		super_likes := 0
	}
	
	sleep 500
	
	while true
	{	
		start:
		
		if dating_app == "tinder"
		{
			; send "{f5}"
			; sleep 12000
		}
		else if dating_app == "bumble"
		{			
			if InStr(ocr(1025, 476, 1096, 503, 500), "Want", 0) or InStr(ocr(1049, 514, 1100, 542, 500), "You", 0) or !WinActive("Bumble")
			{
				send "{f5}"
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 7000
				
				goto("start")
			}	
			else if InStr(ocr(1124, 706, 1173, 728, 500), "Open", 0)
			{
				sleep 100
				A_Clipboard := ""
				sleep 100
				mouseClick "left", 1160, 770
			}
		}
		else if dating_app == "hinge"
		{
			if InStr(ocr(910, 775, 1000, 808, 500), "skipped", 0)
			{
				mouseClick "left", 837, 1027
				sleep 3000
				mouseClick "left", 714, 1028
				sleep 3000
			}
		}
	
		FileAppend "`n`nsuper_likes: " . super_likes . "`n", ".\log.txt"
	
		take_screenshot(dating_app)  
		
		decision := make_decision()
		
		if decision == "super_like" && super_likes > 0
		{
			super_like(dating_app, root_directory)
			super_likes -= 1
			FileAppend "actual decision: super_like" . "`n", ".\log.txt"
		}
		else if decision == "super_like" || decision == "like"
		{
			hinge_opener := like(dating_app, root_directory)
			
			FileAppend "actual decision: like" . "`n", ".\log.txt"
			FileAppend "hinge_opener: '" . hinge_opener . "'`n", ".\log.txt"
		}
		else
		{
			dislike(dating_app)
			
			FileAppend "actual decision: dislike" . "`n", ".\log.txt"
		}		
		
		if dating_app == "tinder"
		{
			sleep 6000
		}
		else if dating_app == "hinge"
		{
			sleep 3000
			; Below cause errors when notification of someone matching with you occurs at the same time as we are clicking off prompt poll popup, it will click the match popup and go to messages page.
			; Click off add a prompt poll popup
			; mouseClick "left", 723, 161
			; sleep 3000
			; mouseClick "left", 828, 567
			; sleep 3000
			
			; If no prompt poll, need to remove photo description as a consequence of clicking on it
			mouseClick "left", 957, 634
		}
		
		clear_screenshot_directory(root_directory, dating_app)
		
		if dating_app == "tinder"
		{
			sleep 5000
		}
	}
	
}

; "tinder", "bumble", "okcupid", "match", "eharmony", "hinge"

; main("tinder", "C:\Users\Dell\Desktop\GitHub\Operation_Love", "1366x768")

; main("bumble", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love", "1920x1080")

main("hinge", "C:\Users\Bull\Desktop\Github\Operation_Love", "1920x1080")



f12::
{
	exitApp
}