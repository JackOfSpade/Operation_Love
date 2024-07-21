f11::
{
	loop 604800
	{
		send "print(abc)"
		sleep 1000
		send "^a"
		send "{backspace}"
	}
}

f12::
{
	soundBeep 
	exitApp
}