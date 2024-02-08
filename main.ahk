; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; 100% zoom in resolution settings
; Laptop must be plugged in or else the save/file dialog will lag.
; Greenshot: set output location to Screenshots folder
; 			 set capture region to f11
;            Capture --> turn "Show notifications" off
; Capture2Text: unbind Win + R so we can open run dialog
;				turn off show popup window
; In powershell (run as admin):
; 	Set-ExecutionPolicy Unrestricted -Scope LocalMachine
; 	Unblock-File -Path "...\Desktop\GitHub\Operation_Love\open_ai_clip\venv\Scripts\activate.ps1"


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
	
	if super_likes == "o" or super_likes == "O"
	{
		super_likes := 0
	}
	
	if dating_app == "photofeeler"
	{
		super_likes := 2000000000
	}
	
	sleep 500
	
	while true
	{	
		start:
		
		; Refresh Page Logics
		if dating_app == "tinder"
		{
			; Second condition is for when tinder pops up the card "It's a match!"
			if InStr(ocr(811, 534, 845, 555, 100), "Go", 0) or InStr(ocr(964, 534, 1015, 555, 100), "SEND", 0) or InStr(ocr(386, 336, 425, 364, 100), "Aw", 0)
			{
				send "{f5}"
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 15000
				
				goto("start")
			}	
		}
		else if dating_app == "bumble"
		{			
			if InStr(ocr(1025, 476, 1096, 510, 100), "Want", 0) or InStr(ocr(1049, 514, 1100, 542, 100), "You", 0) or InStr(ocr(659, 441, 701, 467, 500), "Aw", 0) or !WinActive("Bumble")
			{
				send "{f5}"
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 7000
				
				goto("start")
			}	
			; Click away BOOM match popup
			else if InStr(ocr(1119, 721, 1173, 751, 500), "Open", 0)
			{
				sleep 100
				A_Clipboard := ""
				sleep 100
				mouseClick "left", 1183, 781
				sleep 10000
			}
			
			; Click away matched notification (need to redo, this makes it skip first pic and scroll down)
			; mouseClick "left", 1167, 786
		}
		else if dating_app == "hinge"
		{
			; search for "skipped", then refreshes
			if InStr(ocr(910, 775, 1000, 808, 500), "sk", 0)
			{
				; Go into match preferences
				mouseClick "left", 943, 688
				sleep 5000
				; Go back
				mouseClick "left", 1229, 69
				sleep 7000
				goto("start")
			}
			
			; search for "skipped" text or "Try" in "Try Again", then refresh matches
			if InStr(ocr(900, 656, 946, 690, 500), "Try", 0)
			{
				; Click "Try Again"
				mouseClick "left", 957, 670
				sleep 500
				mouseClick "left", 957, 670
				sleep 7000
				goto("start")
			}
		}
		else if dating_app == "photofeeler" 
		{			
			; Test open_ai_clip
			; if InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			
			if InStr(ocr(769, 145, 925, 245, 100), "Max", 0) or InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			{
				; F5 for refresh stops working after repeated uses
				mouseClick "left", 103, 69
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 10000
				
				goto("start")
			}	
		}
		
		take_screenshot(dating_app)  
		
		decision := make_decision()
		
			
		
		; Liking Logic
		if decision == "super_like" && super_likes > 0
		{
			super_like(dating_app, root_directory)
			super_likes -= 1
		}
		else if decision == "super_like" || decision == "like"
		{
			hinge_opener := like(dating_app, root_directory)
		}
		else
		{
			dislike(dating_app)
		}		
		
		if dating_app == "tinder"
		{
			sleep 6000
		}
		else if dating_app == "bumble"
		{
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
			; mouseClick "left", 957, 634
		}
		else if dating_app == "photofeeler"
		{
		}
		
		clear_screenshot_directory(root_directory, dating_app)
		
		if dating_app == "tinder"
		{
			sleep 5000
		}
	}
	
}

; "tinder", "bumble", "okcupid", "match", "eharmony", "hinge"

main("photofeeler", "C:\Users\LENOVO\Desktop\GitHub\Operation_Love", "1366x768")

; main("tinder", "C:\Users\Dell\Desktop\GitHub\Operation_Love", "1366x768")

; main("bumble", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love", "1920x1080")

; main("hinge", "C:\Users\Bull\Desktop\Github\Operation_Love", "1920x1080")


; TESTING
; main("bumble", "C:\Users\super\Desktop\GitHub\Operation_Love", "1920x1080")



f12::
{
	exitApp
}