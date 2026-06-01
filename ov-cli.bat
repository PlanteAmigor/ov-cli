@echo off
REM ov-cli: Windows 入口脚本
REM 自动发现 .venv 并进入虚拟环境运行

set DIR=%~dp0
set VENV=%DIR%.venv

if not exist "%VENV%\Scripts\python.exe" (
    echo 错误: 找不到虚拟环境，请先运行: ov-cli setup
    exit /b 1
)

"%VENV%\Scripts\python.exe" -m ov_cli %*
