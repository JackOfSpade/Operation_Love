#SingleInstance

RunWait('powershell.exe -noexit -Command "Set-ExecutionPolicy Bypass -Scope Process; .\run_open_ai_clip.ps1"', "", "hide")
