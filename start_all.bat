@echo off
title SeaArrayCalib
powershell.exe -NoProfile -File "%~dp0start_web.ps1"
if errorlevel 1 pause
