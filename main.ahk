; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; 100% zoom on resolution
; Open 2 separate chrome maximized window, one with the dating site and one with the attractiveness eval site
; Laptop must be plugged in or else the save/file dialog will lag.
; Greenshot: set output location to screenshot folder
; 			 set capture region to f11
; ChatGPT tab name should be "hinge" with upload file capability.

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
		if dating_app == "tinder"
		{
			send "{f5}"
			sleep 12000
		}
		else if dating_app == "hinge"
		{
			send "{f5}"
			sleep 6000
		}
	
		FileAppend "`n`nsuper_likes: " . super_likes . "`n", ".\log.txt"
	
		take_screenshot(dating_app)  
		
		decision := make_decision(root_directory, resolution, dating_app)
		
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
			sleep 3000
		}
		else if dating_app == "hinge"
		{
			sleep 3000
			; Click off add a prompt poll popup
			mouseClick "left", 723, 161
			sleep 3000
			mouseClick "left", 828, 567
			sleep 3000
		}
		
		clear_screenshot_directory(root_directory)
		
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