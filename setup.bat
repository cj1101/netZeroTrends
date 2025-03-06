@echo off 
echo Setting up Net Zero Analyzer... 
 
python -m venv nz_env 
call nz_env\Scripts\activate.bat 
pip install pandas numpy matplotlib scikit-learn 
 
echo @echo off > run_analyzer.bat 
echo call nz_env\Scripts\activate.bat >> run_analyzer.bat 
echo python net_zero_analyzer.py >> run_analyzer.bat 
echo pause >> run_analyzer.bat 
 
echo Setup complete! Double-click run_analyzer.bat to start the program. 
pause 
