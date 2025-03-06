# Net Zero Trends Analyzer

## Project Overview
The Net Zero Trends Analyzer is a Python application for analyzing and visualizing greenhouse gas emissions data, helping organizations track their progress towards net zero targets.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/netZeroTrends.git
cd netZeroTrends
```

### 2. Create Virtual Environment
```bash
python -m venv nz_env
# On Windows
nz_env\Scripts\activate
# On macOS/Linux
source nz_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Run the Application
```bash
python net_zero_analyzer.py
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
