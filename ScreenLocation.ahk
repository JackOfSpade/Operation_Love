; `: set location
; 1: test location
#SingleInstance force

CoordMode "Mouse", "Window"

SetTimer Check, 50
x := 0
y := 0


Check()
{
	global x
	global y
	MouseGetPos &x, &y
}

`::
{
	global x
	global y
	soundBeep
	A_Clipboard := x . ", " . y
}


f12::
{
	ExitApp
}
