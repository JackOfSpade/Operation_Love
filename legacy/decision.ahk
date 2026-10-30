; Decision Module: consume only a freshly successful analysis result.

#include helper_functions.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

make_decision()
{
    resultPath := A_ScriptDir "\beauty_and_BMI_analysis\beauty_and_BMI_analysis_result.txt"
    if FileExist(resultPath)
        FileDelete(resultPath)

    exitCode := RunWait('powershell.exe -NoProfile -File "' A_ScriptDir '\run_beauty_and_BMI_analysis.ps1"')
    if exitCode != 0 or !FileExist(resultPath)
        throw Error("Profile analysis failed; automation stopped without a decision.")

    firstLine := Trim(FileRead(resultPath))
    if firstLine != "super_like"
    and firstLine != "like"
    and firstLine != "dislike"
    and firstLine != "No face detected"
        throw Error("Profile analysis returned an invalid decision; automation stopped.")

    return firstLine
}
