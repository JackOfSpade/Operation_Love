#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

ocr(x1, y1, x2, y2)
{
	MouseMove x1, y1
	Send "{LWin down}"
	Send "{q down}"
	Send "{q up}"
	Send "{LWin up}"
	MouseMove x2, y2
	sleep 100
	send "{LButton}"	
	return A_Clipboard	
}

print_screen(x1, y1, x2, y2)
{
	send "<#+s"
	sleep 1500
	MouseMove x1, y1
	Send "{LButton down}"	
	sleep 100
	MouseMove x2, y2
	Send "{LButton up}"
	sleep 1000
}