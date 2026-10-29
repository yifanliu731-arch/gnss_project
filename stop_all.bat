@echo off
title Stop SeaArrayCalib
powershell.exe -NoProfile -File "%~dp0stop_web.ps1"
if errorlevel 1 pause
