import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
import os
import csv
import datetime as dt

class NetZeroAnalyzer:
    """Analysis engine for emissions data and net zero progress."""
    
    def __init__(self):
        self.emissions_data = None
        self.company_name = None
        self.target_year = None
        self.years = None
        self.emissions = None
        self.model = None
        self.slope = None
        self.intercept = None
        self.r_squared = None
        self.required_slope = None
        self.projected_emissions = None
        self.required_reduction_rate = None
        self.current_reduction_rate = None
        
    def load_data(self, data):
        """Load emissions data from a pandas DataFrame."""
        self.emissions_data = data
        self.years = data['year'].values.reshape(-1, 1)
        self.emissions = data['emissions'].values
        
    def set_company_name(self, name):
        """Set company name for reporting."""
        self.company_name = name
        
    def set_target_year(self, year):
        """Set the net zero target year."""
        try:
            self.target_year = int(year)
        except ValueError:
            raise ValueError("Target year must be an integer")
        
    def analyze(self):
        """Perform regression analysis on emissions data."""
        if self.emissions_data is None:
            raise ValueError("Data must be loaded before analysis")
        if self.target_year is None:
            raise ValueError("Target year must be set before analysis")
            
        # Linear regression on historical data
        self.model = LinearRegression().fit(self.years, self.emissions)
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        
        # Calculate R-squared
        y_pred = self.model.predict(self.years)
        y_mean = np.mean(self.emissions)
        ss_total = np.sum((self.emissions - y_mean) ** 2)
        ss_residual = np.sum((self.emissions - y_pred) ** 2)
        self.r_squared = 1 - (ss_residual / ss_total)
        
        # Calculate required slope to reach net zero
        last_year = int(self.years[-1])
        last_emission = self.emissions[-1]
        years_to_target = self.target_year - last_year
        self.required_slope = -last_emission / years_to_target if years_to_target > 0 else 0
        
        # Calculate reduction rates as percentages
        self.current_reduction_rate = -self.slope / self.emissions[0] * 100
        total_required_reduction = self.emissions[-1]
        years_remaining = self.target_year - self.years[-1][0]
        self.required_reduction_rate = (total_required_reduction / self.emissions[-1]) / years_remaining * 100 if years_remaining > 0 else float('inf')
        
        # Project future emissions based on current trajectory
        future_years = np.arange(int(self.years[0]), self.target_year + 1).reshape(-1, 1)
        self.projected_emissions = self.model.predict(future_years)
        
        # Determine expected emissions at target year based on current trajectory
        target_year_projection = self.intercept + self.slope * self.target_year
        
        return {
            'current_slope': self.slope,
            'r_squared': self.r_squared,
            'required_slope': self.required_slope,
            'current_reduction_rate': self.current_reduction_rate,
            'required_reduction_rate': self.required_reduction_rate,
            'target_year_projection': target_year_projection,
            'on_track': target_year_projection <= 0
        }
    
    def create_figure(self):
        """Create matplotlib figure for visualization."""
        if self.model is None:
            raise ValueError("Must run analyze() before visualization")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data
        ax.scatter(self.years, self.emissions, color='blue', label='Historical Emissions', s=50)
        
        # Plot regression line for current trajectory
        future_years = np.arange(int(self.years[0]), self.target_year + 1).reshape(-1, 1)
        future_emissions = self.model.predict(future_years)
        ax.plot(future_years, future_emissions, 'b--', linewidth=2, label=f'Current Trajectory (R² = {self.r_squared:.2f})')
        
        # Plot required reduction line to reach net zero
        last_year = int(self.years[-1])
        last_emission = self.emissions[-1]
        target_years = np.array([last_year, self.target_year]).reshape(-1, 1)
        target_emissions = np.array([last_emission, 0])
        ax.plot(target_years, target_emissions, 'r-', linewidth=2, label='Required Net Zero Path')
        
        # Mark the net zero target point
        ax.scatter([self.target_year], [0], color='red', s=100, marker='*', label='Net Zero Target')
        
        # Set labels and title
        company_label = f" for {self.company_name}" if self.company_name else ""
        ax.set_title(f'Net Zero Progress Analysis{company_label}', fontsize=15)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Emissions (CO₂ equivalent)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate gap at target year
        target_year_projection = self.intercept + self.slope * self.target_year
        
        # Add text about the verdict
        if target_year_projection <= 0:
            verdict = f"ON TRACK to reach net zero before {self.target_year}"
            verdict_color = 'green'
        else:
            years_beyond_target = -target_year_projection / self.slope if self.slope < 0 else float('inf')
            if years_beyond_target != float('inf'):
                actual_zero_year = self.target_year + years_beyond_target
                verdict = f"NOT ON TRACK. At current rates, will reach net zero by {int(actual_zero_year)}"
            else:
                verdict = f"NOT ON TRACK. Emissions are increasing, net zero unreachable"
            verdict_color = 'red'
            
        plt.figtext(0.15, 0.01, verdict, fontsize=12, weight='bold', color=verdict_color)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig
    
    def generate_report(self):
        """Generate a text report summarizing the analysis."""
        if self.model is None:
            raise ValueError("Must run analyze() before generating report")
        
        company = self.company_name if self.company_name else "The company"
        target_year_projection = self.intercept + self.slope * self.target_year
        
        report = [
            f"NET ZERO PROGRESS ANALYSIS{'  FOR ' + self.company_name.upper() if self.company_name else ''}",
            f"{'=' * 50}",
            f"Analysis Date: {dt.datetime.now().strftime('%Y-%m-%d')}",
            f"{'=' * 50}\n",
            f"OVERVIEW:",
            f"- Base Year: {int(self.years[0])} (Emissions: {self.emissions[0]:.1f})",
            f"- Latest Year: {int(self.years[-1])} (Emissions: {self.emissions[-1]:.1f})",
            f"- Net Zero Target Year: {self.target_year}",
            f"- Years of historical data analyzed: {len(self.years)}",
            f"- Years remaining until target: {self.target_year - int(self.years[-1])}\n",
            f"TRAJECTORY ANALYSIS:",
            f"- Current emissions trend: {'DECREASING' if self.slope < 0 else 'INCREASING'} by {abs(self.slope):.2f} per year",
            f"- R² (quality of trend line): {self.r_squared:.2f}",
            f"- Annual reduction as percentage of base year: {abs(self.current_reduction_rate):.2f}%",
            f"- Required annual reduction to reach target: {abs(self.required_reduction_rate):.2f}% of current emissions\n",
            f"TARGET YEAR PROJECTION:",
            f"- Projected emissions at target year: {target_year_projection:.1f}",
        ]
        
        # Add verdict
        if target_year_projection <= 0:
            estimated_zero_year = int(self.years[-1]) - int(self.emissions[-1] / self.slope)
            report.append(f"- VERDICT: ON TRACK to reach net zero by {estimated_zero_year} (before target)")
        else:
            if self.slope < 0:
                years_beyond_target = -target_year_projection / self.slope
                actual_zero_year = self.target_year + years_beyond_target
                report.append(f"- VERDICT: NOT ON TRACK. At current rates, will reach net zero by {int(actual_zero_year)}")
                report.append(f"- Years beyond target: {int(years_beyond_target)}")
            else:
                report.append(f"- VERDICT: NOT ON TRACK. Emissions are INCREASING, net zero unreachable")
            
            acceleration_needed = (self.required_slope / self.slope) if self.slope < 0 else float('inf')
            report.append(f"- Reduction rate must accelerate by {acceleration_needed:.1f}x to reach target")
        
        # Add greenwashing assessment
        report.extend([
            f"\nPOTENTIAL GREENWASHING ASSESSMENT:",
            f"- Is public commitment aligned with action? {'YES' if target_year_projection <= 0 else 'NO'}"
        ])
        
        if target_year_projection > 0:
            if self.slope < 0:
                report.append(f"- {company} is reducing emissions but NOT at the rate required to meet stated goals")
                report.append(f"- Current reduction rate ({abs(self.current_reduction_rate):.1f}%) is inadequate compared to required ({abs(self.required_reduction_rate):.1f}%)")
                multiplier = self.required_reduction_rate / self.current_reduction_rate if self.current_reduction_rate != 0 else float('inf')
                report.append(f"- Must accelerate reductions by {abs(multiplier):.1f}x to achieve target")
            else:
                report.append(f"- {company} emissions are INCREASING while claiming net zero target")
                report.append(f"- This represents a significant disconnect between public commitments and actual performance")
                report.append(f"- High risk of greenwashing: public statements and actual trajectory are in opposite directions")
        else:
            report.append(f"- {company} appears to be on track to meet or exceed stated net zero goals")
            report.append(f"- Current evidence does not support greenwashing concerns")
        
        return "\n".join(report)


class NetZeroAnalyzerGUI:
    """GUI application for the Net Zero Analyzer."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Net Zero Analyzer")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        self.analyzer = NetZeroAnalyzer()
        self.create_widgets()
        
        # Initialize data storage
        self.emissions_data = []
        self.data_df = None
        
    def create_widgets(self):
        """Create the GUI elements."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Company info frame
        info_frame = ttk.LabelFrame(main_frame, text="Company Information", padding=10)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Company Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.company_name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.company_name_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Net Zero Target Year:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.target_year_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.target_year_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Horizontal separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Data entry frame
        data_frame = ttk.LabelFrame(main_frame, text="Emissions Data Entry", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Manual data entry section
        entry_frame = ttk.Frame(data_frame)
        entry_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(entry_frame, text="Year:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.year_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.year_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(entry_frame, text="Emissions:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.emissions_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.emissions_var, width=15).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(entry_frame, text="Add Data Point", command=self.add_data_point).grid(row=0, column=4, padx=5, pady=5)
        
        # Data table
        table_frame = ttk.Frame(entry_frame)
        table_frame.grid(row=1, column=0, columnspan=5, sticky=tk.NSEW, pady=10)
        
        self.data_tree = ttk.Treeview(table_frame, columns=("Year", "Emissions"), show="headings")
        self.data_tree.heading("Year", text="Year")
        self.data_tree.heading("Emissions", text="Emissions")
        self.data_tree.column("Year", width=100)
        self.data_tree.column("Emissions", width=100)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Button frame for data operations
        button_frame = ttk.Frame(entry_frame)
        button_frame.grid(row=2, column=0, columnspan=5, sticky=tk.EW, pady=5)
        
        ttk.Button(button_frame, text="Import CSV", command=self.import_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All Data", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        
        # Sample data button
        ttk.Button(button_frame, text="Load Sample Data", command=self.load_sample_data).pack(side=tk.LEFT, padx=5)
        
        # Horizontal separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Analysis and results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis & Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Button to run analysis
        ttk.Button(results_frame, text="Run Analysis", command=self.run_analysis).pack(anchor=tk.W, pady=5)
        
        # Notebook for results tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Graph tab
        self.graph_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.graph_frame, text="Graph")
        
        # Report tab
        self.report_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.report_frame, text="Report")
        
        # Text widget for report display
        self.report_text = tk.Text(self.report_frame, wrap=tk.WORD, width=80, height=20)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Export buttons
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export Report", command=self.export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Graph", command=self.export_graph).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Store figure reference
        self.figure = None
        self.canvas = None
    
    def add_data_point(self):
        """Add a data point from the entry fields."""
        try:
            year = int(self.year_var.get().strip())
            emissions = float(self.emissions_var.get().strip())
            
            # Check if the year already exists
            for item in self.emissions_data:
                if item[0] == year:
                    messagebox.showwarning("Duplicate Year", f"Year {year} already exists. Please edit or delete the existing entry.")
                    return
            
            # Add to data list and table
            self.emissions_data.append((year, emissions))
            self.data_tree.insert("", tk.END, values=(year, emissions))
            
            # Clear entry fields
            self.year_var.set("")
            self.emissions_var.set("")
            
            # Update status
            self.status_var.set(f"Added data point: Year {year}, Emissions {emissions}")
            
            # Automatically sort data by year
            self.emissions_data.sort(key=lambda x: x[0])
            self.refresh_data_table()
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for year and emissions.")
    
    def refresh_data_table(self):
        """Refresh the data table with sorted data."""
        # Clear the table
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Repopulate with sorted data
        for year, emissions in self.emissions_data:
            self.data_tree.insert("", tk.END, values=(year, emissions))
    
    def import_csv(self):
        """Import emissions data from a CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Read CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header row
                
                # Check if CSV has correct columns
                if len(header) < 2:
                    messagebox.showerror("CSV Error", "CSV must have at least two columns (year, emissions).")
                    return
                
                # Clear existing data
                self.clear_data()
                
                # Process data rows
                for row in reader:
                    if len(row) >= 2:
                        try:
                            year = int(row[0].strip())
                            emissions = float(row[1].strip())
                            self.emissions_data.append((year, emissions))
                        except ValueError:
                            continue  # Skip invalid rows
            
            # Sort data by year
            self.emissions_data.sort(key=lambda x: x[0])
            
            # Refresh table
            self.refresh_data_table()
            
            # Update status
            self.status_var.set(f"Imported data from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Error importing CSV: {str(e)}")
    
    def remove_selected(self):
        """Remove selected data point from the table."""
        selected_item = self.data_tree.selection()
        if not selected_item:
            messagebox.showinfo("Selection", "Please select a data point to remove.")
            return
        
        # Get the selected year
        values = self.data_tree.item(selected_item)['values']
        year = values[0]
        
        # Remove from data list
        self.emissions_data = [item for item in self.emissions_data if item[0] != year]
        
        # Remove from tree
        self.data_tree.delete(selected_item)
        
        # Update status
        self.status_var.set(f"Removed data point for year {year}")
    
    def clear_data(self):
        """Clear all data from the table."""
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        self.emissions_data = []
        self.status_var.set("All data cleared")
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        # Clear existing data
        self.clear_data()
        
        # Sample data - approximate emissions for a hypothetical company
        sample_data = [
            (2013, 1000),
            (2014, 980),
            (2015, 950),
            (2016, 970),
            (2017, 930),
            (2018, 940),
            (2019, 910),
            (2020, 850),
            (2021, 830),
            (2022, 800)
        ]
        
        # Add to data list and table
        self.emissions_data = sample_data
        self.refresh_data_table()
        
        # Set sample company and target
        self.company_name_var.set("Sample Corp")
        self.target_year_var.set("2030")
        
        # Update status
        self.status_var.set("Loaded sample data")
    
    def run_analysis(self):
        """Run the emissions analysis and display results."""
        # Validate inputs
        if not self.emissions_data:
            messagebox.showerror("Data Error", "Please enter emissions data before running analysis.")
            return
        
        if len(self.emissions_data) < 2:
            messagebox.showerror("Data Error", "Please enter at least two data points for analysis.")
            return
        
        try:
            target_year = int(self.target_year_var.get().strip())
            if target_year <= max(year for year, _ in self.emissions_data):
                messagebox.showerror("Target Error", "Target year must be in the future.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid target year.")
            return
        
        try:
            # Prepare data for analysis
            data_df = pd.DataFrame(self.emissions_data, columns=["year", "emissions"])
            self.data_df = data_df
            
            # Configure analyzer
            self.analyzer.load_data(data_df)
            self.analyzer.set_company_name(self.company_name_var.get().strip())
            self.analyzer.set_target_year(target_year)
            
            # Run analysis
            results = self.analyzer.analyze()
            
            # Update status
            self.status_var.set("Analysis complete")
            
            # Display results
            self.display_graph()
            self.display_report()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
    
    def display_graph(self):
        """Display the analysis graph."""
        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        try:
            fig = self.analyzer.create_figure()
            self.figure = fig
            
            # Display in GUI
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            
            # Store canvas reference
            self.canvas = canvas
            
        except Exception as e:
            messagebox.showerror("Graph Error", f"Error creating graph: {str(e)}")
    
    def display_report(self):
        """Display the analysis report."""
        try:
            # Clear previous report
            self.report_text.delete(1.0, tk.END)
            
            # Generate and display report
            report = self.analyzer.generate_report()
            self.report_text.insert(tk.END, report)
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Error generating report: {str(e)}")
    
    def export_report(self):
        """Export the analysis report to a text file."""
        if not hasattr(self.analyzer, 'model') or self.analyzer.model is None:
            messagebox.showinfo("Export Error", "Please run analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as file:
                file.write(self.analyzer.generate_report())
            
            self.status_var.set(f"Report exported to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Successful", f"Report saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting report: {str(e)}")
    
    def export_graph(self):
        """Export the analysis graph to a PNG file."""
        if self.figure is None:
            messagebox.showinfo("Export Error", "Please run analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Graph",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.status_var.set(f"Graph exported to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Successful", f"Graph saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting graph: {str(e)}")


def main():
    root = tk.Tk()
    app = NetZeroAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()