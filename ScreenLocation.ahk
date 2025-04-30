; `: set location
; 1: test location
#SingleInstance force

CoordMode "Mouse", "Window"

x := 0
y := 0


check()
{
	global x
	global y
	mouseMove x, y
	
}

`::
{
	global x
	global y
	MouseGetPos &x, &y
	A_Clipboard := x . ", " . y
	
	
	soundBeep
}

1::
{
	check()
	soundBeep
}

f12::
{
	ExitApp
}
