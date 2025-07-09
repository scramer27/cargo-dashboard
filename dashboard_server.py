import matplotlib
matplotlib.use('Agg') # use non-interactive backend

import sys
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from flask import Flask, Response, render_template_string

# --- plotting functions ---
def create_summary_charts(summary_df):
    # generates trend charts from the master summary data
    if summary_df.shape[0] < 2:
        return {}
    charts = {}
    df = summary_df.copy()
    
    df.reset_index(drop=True, inplace=True)
    df['Day'] = pd.to_numeric(df['Day'])
    
    sns.set_theme(style="whitegrid")
    
    # chart 1: average times vs. day
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Day', y='Stow Avg (s)', marker='o', label='Stow Avg (s)')
    sns.lineplot(data=df, x='Day', y='Retrieve Avg (s)', marker='o', label='Retrieve Avg (s)')
    sns.lineplot(data=df, x='Day', y='Read Label Avg (s)', marker='o', label='Read Label Avg (s)')
    plt.title('Average Process Times Over Days', fontsize=16, weight='bold')
    plt.ylabel('Time (s)')
    plt.xlabel('Day') 
    plt.xticks(rotation=0) 
    plt.tight_layout()
    
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100); plt.close()
    charts['times_chart'] = base64.b64encode(img_data.getvalue()).decode('utf-8')

    # chart 2: throughput vs. day
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Day', y='Throughput (pkg/hr)', marker='o', color='tab:blue')
    plt.title('Throughput Over Days', fontsize=16, weight='bold')
    plt.ylabel('Throughput (pkg/hr)')
    plt.xlabel('Day')
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100); plt.close()
    charts['throughput_chart'] = base64.b64encode(img_data.getvalue()).decode('utf-8')

    # chart 3: packages stowed vs. attempts
    plt.figure(figsize=(10, 5))
    stow_data = df[['Day', 'Packages Stowed', 'Stow Attempts']]
    stow_data_melted = stow_data.melt(id_vars='Day', var_name='Metric', value_name='Count')
    sns.barplot(data=stow_data_melted, x='Day', y='Count', hue='Metric', palette=['#34a853', '#fbbc05'])
    plt.title('Packages Stowed vs. Stow Attempts', fontsize=16, weight='bold')
    plt.ylabel('Count')
    plt.xlabel('Day')
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100); plt.close()
    charts['stow_chart'] = base64.b64encode(img_data.getvalue()).decode('utf-8')

    # chart 4: packages retrieved vs. attempts
    plt.figure(figsize=(10, 5))
    retrieval_data = df[['Day', 'Packages Retrieved', 'Retrieval Attempts']]
    retrieval_data_melted = retrieval_data.melt(id_vars='Day', var_name='Metric', value_name='Count')
    
    sns.barplot(data=retrieval_data_melted, x='Day', y='Count', hue='Metric')
    
    plt.title('Packages Retrieved vs. Retrieval Attempts', fontsize=16, weight='bold')
    plt.ylabel('Count')
    plt.xlabel('Day')
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100); plt.close()
    charts['packages_chart'] = base64.b64encode(img_data.getvalue()).decode('utf-8')
    
    return charts

def create_distribution_histogram(data_series, title):
    # generates a distribution histogram for a series of data
    if data_series.empty or data_series.isnull().all(): return None
    plt.figure(figsize=(8, 4.5))
    sns.histplot(data_series.dropna(), kde=True, bins=15, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=14, weight='bold'); plt.xlabel('Time (s)'); plt.ylabel('Frequency'); plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100); plt.close()
    return base64.b64encode(img_data.getvalue()).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"> <meta http-equiv="refresh" content="30">
    <title>Cargo Robotics Performance Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; }
        .container { max-width: 1600px; margin: 0 auto; padding: 1rem; } h1 { text-align: center; }
        .tabs { display: flex; border-bottom: 2px solid #dee2e6; margin-bottom: 1.5rem; flex-wrap: wrap; }
        .tab-link { padding: 0.8rem 1.2rem; cursor: pointer; border: 2px solid transparent; margin-bottom: -2px; font-weight: 500; background: none; }
        .tab-link.active { color: #0d6efd; border-color: #dee2e6 #dee2e6 #fff; border-radius: 0.25rem 0.25rem 0 0; background-color: #fff; }
        .tab-content { display: none; animation: fadeIn 0.5s; } .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        table { border-collapse: collapse; width: auto; margin-bottom: 2rem; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.05); font-size: 0.9rem; }
        th, td { border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; } th { background-color: #e9ecef; }
        .charts-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; }
        .chart-wrapper { background-color: #fff; padding: 1rem; border-radius: 0.25rem; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
        .chart-wrapper img { max-width: 100%; } h2 { border-bottom: 1px solid #dee2e6; padding-bottom: 0.5rem; margin-top: 2rem; }
        .flex-container { display: flex; flex-wrap: wrap; gap: 2rem; align-items: flex-start; }
        .benchmark-column { flex: 1; min-width: 400px; }
        td:first-child { word-break: break-word; }
    </style>
</head>
<body>
    <div class="container"> <h1>Cargo Performance Dashboard</h1> <div class="tabs"> {{ TABS }} </div> {{ CONTENT }} </div>
    <script>
        function openTab(evt, tabName) {
            // if called by a real click, evt will exist
            if (evt) {
                evt.currentTarget.classList.add("active");
            }
            document.querySelectorAll('.tab-content').forEach(tc => tc.style.display = "none");
            document.querySelectorAll('.tab-link').forEach(tl => tl.classList.remove("active"));
            
            document.getElementById(tabName).style.display = "block";
            document.querySelector(`[data-tab-name='${tabName}']`).classList.add("active");
            localStorage.setItem('activeTab', tabName);
        }

        document.addEventListener("DOMContentLoaded", function() {
            // remember the active tab across reloads
            const tabs = document.querySelectorAll('.tab-link');
            if (tabs.length === 0) return;

            let activeTabName = localStorage.getItem('activeTab');
            let tabToActivate = document.querySelector(`[data-tab-name='${activeTabName}']`);

            // if the stored tab doesn't exist, default to the first one
            if (!tabToActivate) {
                activeTabName = tabs[0].getAttribute('data-tab-name');
            }
            
            // open the tab without needing an event object
            openTab(null, activeTabName);
        });
    </script>
</body>
</html>
"""

# --- flask web server ---
app = Flask(__name__)
EXCEL_PATH = "" # will be set from a command line argument

def generate_html_content():
    # main logic to generate the dashboard html from the excel report
    if not Path(EXCEL_PATH).exists():
        return "<h1>Error</h1><p>Report file not found at '{}'. Please run report_updater.py first.</p>".format(EXCEL_PATH)

    xls = pd.ExcelFile(EXCEL_PATH)
    tabs_html = ""
    content_html = ""

    # --- master summary tab ---
    summary_df = pd.read_excel(xls, 'Master Summary')
    tabs_html += '<button class="tab-link" data-tab-name="summary" onclick="openTab(event, \'summary\')">Master Summary</button>'
    
    # make sure relevant columns are numeric for calculations
    numeric_cols = ['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 'Min Stow Gap (s)', 'Min Retrieve Gap (s)', 'Throughput (pkg/hr)', 'Packages Stowed', 'Stow Attempts', 'Packages Retrieved', 'Retrieval Attempts', 'Total Errors', 'Stow Driver Shift Time (hr)', 'Retrieve Driver Shift Time (hr)']
    for col in summary_df.columns:
        if col in numeric_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    # --- performance summary table ---
    last_day_stats = summary_df.iloc[-1] if not summary_df.empty else None
    prev_day_stats = summary_df.iloc[-2] if len(summary_df) >= 2 else None

    def get_change(current, previous, is_time=False):
        if pd.isna(current) or pd.isna(previous): return 'N/A'
        if previous == 0: return "∞" if current > 0 else "0.00%"
        change = ((current - previous) / previous) * 100
        if is_time:
            color = "black" if change > 0 else "black"
            return f'<span style="color:{color};">{change:+.2f}%</span>'
        return f"{change:+.2f}%"

    def get_rate_change(current_rate, prev_rate):
        if pd.isna(current_rate) or pd.isna(prev_rate): return 'N/A'
        change = current_rate - prev_rate
        return f"{change:+.2f} pts"

    stow_success_rate = (summary_df['Packages Stowed'].sum() / summary_df['Stow Attempts'].sum() * 100) if summary_df['Stow Attempts'].sum() > 0 else 0
    retrieval_success_rate = (summary_df['Packages Retrieved'].sum() / summary_df['Retrieval Attempts'].sum() * 100) if summary_df['Retrieval Attempts'].sum() > 0 else 0
    
    last_stow_rate = (last_day_stats['Packages Stowed'] / last_day_stats['Stow Attempts'] * 100) if last_day_stats is not None and last_day_stats['Stow Attempts'] > 0 else 0
    last_retrieval_rate = (last_day_stats['Packages Retrieved'] / last_day_stats['Retrieval Attempts'] * 100) if last_day_stats is not None and last_day_stats['Retrieval Attempts'] > 0 else 0
    
    prev_stow_rate = (prev_day_stats['Packages Stowed'] / prev_day_stats['Stow Attempts'] * 100) if prev_day_stats is not None and prev_day_stats['Stow Attempts'] > 0 else 0
    prev_retrieval_rate = (prev_day_stats['Packages Retrieved'] / prev_day_stats['Retrieval Attempts'] * 100) if prev_day_stats is not None and prev_day_stats['Retrieval Attempts'] > 0 else 0

    perf_data = {
        'Metric': ['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 'Throughput (pkg/hr)', 'Stow Success Rate', 'Retrieval Success Rate'],
        'Overall Average': [
            f"{summary_df['Stow Avg (s)'].mean():.2f}", f"{summary_df['Retrieve Avg (s)'].mean():.2f}",
            f"{summary_df['Read Label Avg (s)'].mean():.2f}", f"{summary_df['Throughput (pkg/hr)'].mean():.2f}",
            f"{stow_success_rate:.2f}%", f"{retrieval_success_rate:.2f}%"
        ],
        'Most Recent Day': [
            f"{last_day_stats['Stow Avg (s)']:.2f}" if last_day_stats is not None else 'N/A',
            f"{last_day_stats['Retrieve Avg (s)']:.2f}" if last_day_stats is not None else 'N/A',
            f"{last_day_stats['Read Label Avg (s)']:.2f}" if last_day_stats is not None else 'N/A',
            f"{last_day_stats['Throughput (pkg/hr)']:.2f}" if last_day_stats is not None else 'N/A',
            f"{last_stow_rate:.2f}%" if last_day_stats is not None else 'N/A',
            f"{last_retrieval_rate:.2f}%" if last_day_stats is not None else 'N/A'
        ],
        'Change from Prev. Day': [
            get_change(last_day_stats['Stow Avg (s)'], prev_day_stats['Stow Avg (s)']) if last_day_stats is not None and prev_day_stats is not None else 'N/A',
            get_change(last_day_stats['Retrieve Avg (s)'], prev_day_stats['Retrieve Avg (s)']) if last_day_stats is not None and prev_day_stats is not None else 'N/A',
            get_change(last_day_stats['Read Label Avg (s)'], prev_day_stats['Read Label Avg (s)']) if last_day_stats is not None and prev_day_stats is not None else 'N/A',
            get_change(last_day_stats['Throughput (pkg/hr)'], prev_day_stats['Throughput (pkg/hr)']) if last_day_stats is not None and prev_day_stats is not None else 'N/A',
            get_rate_change(last_stow_rate, prev_stow_rate) if last_day_stats is not None and prev_day_stats is not None else 'N/A',
            get_rate_change(last_retrieval_rate, prev_retrieval_rate) if last_day_stats is not None and prev_day_stats is not None else 'N/A'
        ]
    }
    perf_df = pd.DataFrame(perf_data)
    high_level_summary_html = '<h2>KPI Tracker</h2>' + perf_df.to_html(index=False, na_rep="", classes='table', escape=False)

    # --- benchmark comparison tables ---
    benchmark_html = ""
    # check if required columns for benchmarks exist
    benchmark_cols_exist = (
        last_day_stats is not None and prev_day_stats is not None and
        'Stow Driver Shift Time (hr)' in last_day_stats and 'Retrieve Driver Shift Time (hr)' in last_day_stats and
        'Stow Driver Shift Time (hr)' in summary_df.columns and 'Retrieve Driver Shift Time (hr)' in summary_df.columns
    )

    if benchmark_cols_exist:
        # benchmark constants
        fedex_pkgs, fedex_stow_hrs, fedex_retrieve_hrs = 150, 3.0, 3.0
        amazon_pkgs, amazon_stow_hrs, amazon_retrieve_hrs = 219, 0.3, 3.9

        # helper for calculating change from benchmark
        def get_benchmark_change(our_value, benchmark_value):
            if pd.isna(our_value) or pd.isna(benchmark_value) or benchmark_value == 0: return "N/A"
            change = ((our_value - benchmark_value) / benchmark_value) * 100
            color = "black" if change > 0 else "black"
            return f'<span style="color:{color};">{change:+.2f}%</span>'

        # calculate our performance per package for different periods
        stow_time_per_pkg_last = (last_day_stats['Stow Driver Shift Time (hr)'] / last_day_stats['Packages Stowed']) if last_day_stats['Packages Stowed'] > 0 else float('nan')
        retrieve_time_per_pkg_last = (last_day_stats['Retrieve Driver Shift Time (hr)'] / last_day_stats['Packages Retrieved']) if last_day_stats['Packages Retrieved'] > 0 else float('nan')
        
        stow_time_per_pkg_prev = (prev_day_stats['Stow Driver Shift Time (hr)'] / prev_day_stats['Packages Stowed']) if prev_day_stats['Packages Stowed'] > 0 else float('nan')
        retrieve_time_per_pkg_prev = (prev_day_stats['Retrieve Driver Shift Time (hr)'] / prev_day_stats['Packages Retrieved']) if prev_day_stats['Packages Retrieved'] > 0 else float('nan')

        avg_stow_time_per_pkg = (summary_df['Stow Driver Shift Time (hr)'] / summary_df['Packages Stowed']).mean()
        avg_retrieve_time_per_pkg = (summary_df['Retrieve Driver Shift Time (hr)'] / summary_df['Packages Retrieved']).mean()

        # fedex table
        our_stow_fedex_avg = avg_stow_time_per_pkg * fedex_pkgs
        our_retrieve_fedex_avg = avg_retrieve_time_per_pkg * fedex_pkgs
        our_stow_fedex_last = stow_time_per_pkg_last * fedex_pkgs
        our_retrieve_fedex_last = retrieve_time_per_pkg_last * fedex_pkgs
        our_stow_fedex_prev = stow_time_per_pkg_prev * fedex_pkgs
        our_retrieve_fedex_prev = retrieve_time_per_pkg_prev * fedex_pkgs
        
        fedex_data = {
            'Metric': ["Stow (Loading) Time (hr)", "Retrieve (Unloading) Time (hr)"],
            'Overall Average': [f"{our_stow_fedex_avg:.2f}", f"{our_retrieve_fedex_avg:.2f}"],
            'Most Recent Day': [f"{our_stow_fedex_last:.2f}", f"{our_retrieve_fedex_last:.2f}"],
            'Change from Previous Day': [get_change(our_stow_fedex_last, our_stow_fedex_prev, is_time=True), get_change(our_retrieve_fedex_last, our_retrieve_fedex_prev, is_time=True)],
            'Benchmark': [f"{fedex_stow_hrs:.1f}", f"{fedex_retrieve_hrs:.1f}"],
            'Change from Benchmark': [get_benchmark_change(our_stow_fedex_avg, fedex_stow_hrs), get_benchmark_change(our_retrieve_fedex_avg, fedex_retrieve_hrs)]
        }
        fedex_df = pd.DataFrame(fedex_data)
        fedex_html = f'<h3>FedEx Benchmark ({fedex_pkgs} pkgs/van)</h3>' + fedex_df.to_html(index=False, na_rep="N/A", classes='table', escape=False)

        # amazon table
        our_stow_amazon_avg = avg_stow_time_per_pkg * amazon_pkgs
        our_retrieve_amazon_avg = avg_retrieve_time_per_pkg * amazon_pkgs
        our_stow_amazon_last = stow_time_per_pkg_last * amazon_pkgs
        our_retrieve_amazon_last = retrieve_time_per_pkg_last * amazon_pkgs
        our_stow_amazon_prev = stow_time_per_pkg_prev * amazon_pkgs
        our_retrieve_amazon_prev = retrieve_time_per_pkg_prev * amazon_pkgs

        amazon_data = {
            'Metric': ["Stow (Loading) Time (hr)", "Retrieve (Unloading) Time (hr)"],
            'Overall Average': [f"{our_stow_amazon_avg:.2f}" if pd.notna(our_stow_amazon_avg) else "N/A", f"{our_retrieve_amazon_avg:.2f}" if pd.notna(our_retrieve_amazon_avg) else "N/A"],
            'Most Recent Day': [f"{our_stow_amazon_last:.2f}", f"{our_retrieve_amazon_last:.2f}"],
            'Change from Previous Day': [get_change(our_stow_amazon_last, our_stow_amazon_prev, is_time=True), get_change(our_retrieve_amazon_last, our_retrieve_amazon_prev, is_time=True)],
            'Benchmark': [f"{amazon_stow_hrs:.1f}", f"{amazon_retrieve_hrs:.1f}"],
            'Change from Benchmark': [get_benchmark_change(our_stow_amazon_avg, amazon_stow_hrs), get_benchmark_change(our_retrieve_amazon_avg, amazon_retrieve_hrs)]
        }
        amazon_df = pd.DataFrame(amazon_data)
        amazon_html = f'<h3>Amazon Benchmark ({amazon_pkgs} pkgs/van)</h3>' + amazon_df.to_html(index=False, na_rep="N/A", classes='table', escape=False)

        benchmark_html = '<h2>Benchmark Comparison</h2>'
        benchmark_html += f'{fedex_html}{amazon_html}'

    # --- error analysis table ---
    error_table_html = "<h2>Error Analysis</h2><p>Not enough data for error analysis.</p>"
    if 'Error Summary' in xls.sheet_names:
        error_df = pd.read_excel(xls, 'Error Summary').fillna(0)
        # all columns except day and date are error categories
        error_categories = [col for col in error_df.columns if col not in ['Day', 'Date']]
        
        # sum up totals for each error category across all days
        total_errors_by_category = error_df[error_categories].sum()
        total_errors_overall = total_errors_by_category.sum()

        last_day_errors = error_df.iloc[-1] if not error_df.empty else None
        prev_day_errors = error_df.iloc[-2] if len(error_df) >= 2 else None
        
        last_day_total = last_day_errors[error_categories].sum() if last_day_errors is not None else 0
        prev_day_total = prev_day_errors[error_categories].sum() if prev_day_errors is not None else 0

        error_analysis_data = []
        for cat in error_categories:
            # skip if this error never occurred
            if total_errors_by_category[cat] == 0:
                continue

            overall_pct = (total_errors_by_category[cat] / total_errors_overall * 100) if total_errors_overall > 0 else 0
            last_day_pct = (last_day_errors.get(cat, 0) / last_day_total * 100) if last_day_errors is not None and last_day_total > 0 else 0
            prev_day_pct = (prev_day_errors.get(cat, 0) / prev_day_total * 100) if prev_day_errors is not None and prev_day_total > 0 else 0
            
            change_str = 'N/A'
            if last_day_errors is not None and prev_day_errors is not None:
                change_in_pct = last_day_pct - prev_day_pct
                change_str = f"{change_in_pct:+.2f} pts"

            error_analysis_data.append({
                'Error Category': cat,
                'Overall % of Total Errors': overall_pct, # store as a number for sorting
                'Most Recent Day %': last_day_pct,
                'Change from Prev. Day': change_str
            })
        
        # sort by overall percentage, descending
        sorted_error_data = sorted(error_analysis_data, key=lambda x: x['Overall % of Total Errors'], reverse=True)
        
        # group the top 7 errors and combine the rest into "others"
        final_error_data = []
        if len(sorted_error_data) > 7:
            final_error_data = sorted_error_data[:7]
            others_data = sorted_error_data[7:]
            
            others_overall_pct = sum(d['Overall % of Total Errors'] for d in others_data)
            others_recent_pct = sum(d['Most Recent Day %'] for d in others_data)
            
            # change for "others" is tricky and less meaningful, so we omit it
            final_error_data.append({
                'Error Category': 'Others',
                'Overall % of Total Errors': others_overall_pct,
                'Most Recent Day %': others_recent_pct,
                'Change from Prev. Day': 'N/A'
            })
        else:
            final_error_data = sorted_error_data

        # format percentages for display
        for item in final_error_data:
            item['Overall % of Total Errors'] = f"{item['Overall % of Total Errors']:.2f}%"
            item['Most Recent Day %'] = f"{item['Most Recent Day %']:.2f}%"

        analysis_df = pd.DataFrame(final_error_data)
        error_table_html = '<h2>Error Analysis</h2>' + analysis_df.to_html(index=False, na_rep="", classes='table')

    # --- day-by-day table ---
    day_by_day_html = '<h2>Day-by-Day Performance</h2>' + summary_df.to_html(index=False, na_rep="", classes='table')

    # --- charts ---
    summary_charts = create_summary_charts(summary_df)
    charts_html = ""
    if (summary_charts):
        charts_html += '<div class="charts-container">'
        charts_html += f'<div class="chart-wrapper"><img src="data:image/png;base64,{summary_charts["times_chart"]}"></div>'
        charts_html += f'<div class="chart-wrapper"><img src="data:image/png;base64,{summary_charts["throughput_chart"]}"></div>'
        if 'stow_chart' in summary_charts:
             charts_html += f'<div class="chart-wrapper"><img src="data:image/png;base64,{summary_charts["stow_chart"]}"></div>'
        if 'packages_chart' in summary_charts:
             charts_html += f'<div class="chart-wrapper"><img src="data:image/png;base64,{summary_charts["packages_chart"]}"></div>'
        charts_html += '</div>'

    # --- assemble the master summary tab ---
    content_html += '<div id="summary" class="tab-content">'
    content_html += '<div class="flex-container">'
    content_html += f'<div>{high_level_summary_html}</div>'
    content_html += f'<div>{error_table_html}</div>'
    content_html += '</div>'
    content_html += benchmark_html
    content_html += charts_html
    content_html += day_by_day_html
    content_html += '</div>'

    # --- daily tabs ---
    day_sheets = [s for s in xls.sheet_names if s.startswith('Day ')]
    day_sheets.sort(key=lambda x: int(x.split(' ')[1]))

    for sheet_name in day_sheets:
        # load and display daily data
        day_id = sheet_name.replace(" ", "_")
        tabs_html += f'<button class="tab-link" data-tab-name="{day_id}" onclick="openTab(event, \'{day_id}\')">{sheet_name}</button>'
        content_html += f'<div id="{day_id}" class="tab-content">'
        
        day_df = pd.read_excel(xls, sheet_name)
        
        content_html += f'<h2>Full Event Log for {sheet_name}</h2>'
        content_html += day_df.to_html(index=False, na_rep="", classes='table')

        # other daily details like plots could go here

        content_html += '</div>'

    return HTML_TEMPLATE.replace("{{ TABS }}", tabs_html).replace("{{ CONTENT }}", content_html)

@app.route("/")
def show_report():
    # generate and serve the dashboard html
    html_content = generate_html_content()
    return Response(html_content, mimetype="text/html")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dashboard_server.py <path_to_master_report.xlsx>")
        sys.exit(1)
    
    EXCEL_PATH = sys.argv[1]
    print("Starting dashboard server...")
    print(f"--> Open your browser and go to: http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)