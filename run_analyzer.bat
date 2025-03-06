@echo off
echo Running Net Zero Analyzer through WSL...

REM Connect to WSL and run the analyzer
wsl -d Ubuntu bash -c "cd /home/charl/net_zero_analyzer && python3 net_zero_analyzer.py"

pause