# Net Zero Trends Analyzer

## Project Overview
The Net Zero Trends Analyzer is a Python application designed to help analyze and visualize greenhouse gas emissions data, supporting organizations in tracking their progress towards net zero targets.

## Prerequisites
Before you begin, ensure you have the following installed:
- Windows Subsystem for Linux (WSL)
- Ubuntu on WSL
- Python 3.8+ 
- pip (Python package manager)

## Setup Instructions

### 1. Install Required Software
1. Open Windows PowerShell as Administrator
2. Install WSL:
   ```
   wsl --install
   ```
3. Restart your computer
4. Complete Ubuntu installation when prompted

### 2. Prepare Python Environment
1. Open Ubuntu terminal
2. Update package lists:
   ```
   sudo apt update
   sudo apt upgrade
   ```
3. Install Python and pip:
   ```
   sudo apt install python3 python3-pip python3-venv
   ```

### 3. Clone the Repository
1. In Ubuntu terminal, navigate to your desired directory:
   ```
   cd ~/
   mkdir net_zero_analyzer
   cd net_zero_analyzer
   ```
2. Clone the repository:
   ```
   git clone https://github.com/yourusername/netZeroTrends.git .
   ```

### 4. Create Virtual Environment
```
python3 -m venv nz_env
source nz_env/bin/activate
```

### 5. Install Dependencies
```
pip install pandas numpy matplotlib scikit-learn
```

### 6. Run the Application
```
python3 net_zero_analyzer.py
```

## Using the Net Zero Analyzer
1. Enter company name
2. Set net zero target year
3. Input historical emissions data
   - Manually enter data points
   - Import from CSV
4. Click "Run Analysis"
5. View graph and detailed report

## Troubleshooting
- Ensure Python 3.8+ is installed
- Check all dependencies are installed
- Verify CSV format (two columns: year, emissions)

## Contributing
Contributions welcome! Please open an issue or submit a pull request.

## License
[Specify your license here]

## Contact
[Your contact information]
