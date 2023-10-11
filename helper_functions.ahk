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

print_screen(x1, y1, x2, y2)
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
	
	sleep 500
	
	MouseMove 0, 0

	send "{down}"
	send "{down}"
	sleep 500
	send "{enter}"
	
	sleep 100
	A_Clipboard := ""
	sleep 100
}

click_and_drag(x1, y1, x2, y2, delay)
{
	MouseMove x1, y1, 100
	Send "{LButton down}"	
	sleep delay	
	MouseMove x2, y2, 100
	sleep delay
	Send "{LButton up}"
}