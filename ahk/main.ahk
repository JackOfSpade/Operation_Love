; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; Download windows snipping tool and set its setting to auto-save, link prtsc to that.
; Open 2 separate chrome maximized window (3456x2160), one with the dating site and one with https://photo-ranker.com/'
; If using windows <= 11, configure greenshot.
; Open capture2text but unmap Win+R hotkey on it because we need it to open Run command.


#include automation.ahk
#include screenshot.ahk
#include decision.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"


main(dating_app, windows_version)
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
	
	while true
	{
		FileAppend "`n`nsuper_likes: " . super_likes . "`n", ".\log.txt"
	
		take_screenshot(dating_app, windows_version)   
		decision := make_decision()
		
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
		
		clear_screenshot_directory("C:\Users\Shadow\Pictures\Screenshots")
	}
	
	
	
}

dating_app_list := ["tinder", "bumble", "okcupid", "match", "eharmony"]


main(dating_app_list[1], 10)


	
f12::
{
	exitApp
}