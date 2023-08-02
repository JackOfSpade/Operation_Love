#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

; Unreliable, only use if copy-paste is not allowed
ocr(x1, y1, x2, y2, delay)
{
	sleep 100
	A_Clipboard := ""
	sleep 100
	
	MouseMove x1, y1
	Send "{LWin down}"
	Send "{q down}"
	Send "{q up}"
	Send "{LWin up}"
	sleep 100
	MouseMove x2, y2
	sleep delay
	send "{LButton}"	
	; Bumble super_likes button text can be numeric or symbol
	clipwait(1, 1)
	return A_Clipboard	
}

get_text(x1, y1, x2, y2, clipwait_time)
{
	mouseMove x1, y1
	send "{LButton down}"
	sleep 100
	mouseMove x2, y2
	send "{LButton up}"
	sleep 100
	send "^c"
	clipwait(clipwait_time, 0)
}

print_screen(x1, y1, x2, y2, screenshot_directory)
{		
	send "{f11}"
	sleep 1000

	MouseMove x1, y1
	Send "{LButton down}"	
	sleep 500	
	MouseMove x2, y2
	sleep 500
	Send "{LButton up}"
	clipwait(1, 1)
	

	send "{down}"
	send "{down}"
	sleep 500
	send "{enter}"
	
	sleep 100
	A_Clipboard := ""
	sleep 100
}
