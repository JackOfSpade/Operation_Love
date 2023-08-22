
#SingleInstance force

CoordMode "Mouse", "Window"

SetTimer Check, 20
xx := ""
yy := ""


Check()
{
	global xx
	global yy
	MouseGetPos &xx, &yy
}

`::
{
	soundBeep
	A_Clipboard := xx . ", " . yy
}


f12::
{
	ExitApp
}
