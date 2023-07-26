#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

ocr(x1, y1, x2, y2)
{
	A_Clipboard	:= ""
	sleep 50
	MouseMove x1, y1
	Send "{LWin down}"
	Send "{q down}"
	Send "{q up}"
	Send "{LWin up}"
	sleep 100
	MouseMove x2, y2
	sleep 100
	send "{LButton}"	
	clipwait(1)
	return A_Clipboard	
}

print_screen(x1, y1, x2, y2, windows_version, screenshot_directory)
{	
	A_Clipboard	 := ""
	
	send "<#+s"
	sleep 1500

	MouseMove x1, y1
	Send "{LButton down}"	
	sleep 100	
	MouseMove x2, y2
	sleep 100
	Send "{LButton up}"
	sleep 1000
	
	if windows_version <= 10
	{
		send "<#r"
		sleep 500
		send "^a"
		send "MSPaint"
		send "{enter}"
		sleep 500
		send "^v"
		send "^s"
		sleep 1000
		send "^l"
		send "^a"
		send screenshot_directory
		loop 5
		{
			sleep 500
			send "{tab}"
		}
		
		sleep 500
		send Random(0, 9223372036854775807)
		sleep 500
		send "{enter}"
		sleep 500
		WinClose "Paint"		
	}
	
	A_Clipboard	 := ""
}
