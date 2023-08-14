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


main(dating_app, screenshot_directory, resolution)
{
	; Clear the log
	file := FileOpen(".\log.txt", "w")
	file.close()
	
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
		FileAppend "`n`nsuper_likes: " . super_likes . "`n", ".\log.txt"
	
		take_screenshot(dating_app, screenshot_directory)  
		
		decision := make_decision(screenshot_directory, resolution)
		
		if decision == "super_like" && super_likes > 0
		{
			super_like(dating_app)
			super_likes -= 1
			FileAppend "actual decision: super_like" . "`n", ".\log.txt"
		}
		else if decision == "super_like" || decision == "like"
		{
			like(dating_app)
			
			FileAppend "actual decision: like" . "`n", ".\log.txt"
		}
		else
		{
			dislike(dating_app)
			
			FileAppend "actual decision: dislike" . "`n", ".\log.txt"
		}
		
		clear_screenshot_directory(screenshot_directory)
	}
	
	
	
}

; "tinder", "bumble", "okcupid", "match", "eharmony"

main("tinder", "C:\Users\Dell\Desktop\GitHub\Operation_Love\ahk\Screenshots"", "1366x768")

; main("bumble", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love\ahk\Screenshots", "1920x1080")

; main("hinge", "C:\Users\Bull\Desktop\Github\Operation_Love\ahk\Screenshots, "1920x1080")


f12::
{
	exitApp
}