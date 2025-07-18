import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.components.v1 import html

# --- Page Configuration ---
st.set_page_config(
    page_title="Cargo Robotics Performance Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- Theme Detection and Logo Switching ---
# Use a JS-based approach to get the theme and store it in session_state
if 'theme' not in st.session_state:
    st.session_state.theme = "light" # Default

# JavaScript to detect the theme and send it back to Streamlit
js_code = """
<script>
const observer = new MutationObserver(function(mutations) {
    for (const mutation of mutations) {
        if (mutation.attributeName === 'class') {
            const isDark = mutation.target.classList.contains('dark');
            const theme = isDark ? 'dark' : 'light';
            window.parent.Streamlit.setComponentValue(theme);
            // We can disconnect after the first detection if we want
            // observer.disconnect(); 
        }
    }
});
observer.observe(window.parent.document.body, { attributes: true });

// Also send the initial theme
const initialTheme = window.parent.document.body.classList.contains('dark') ? 'dark' : 'light';
window.parent.Streamlit.setComponentValue(initialTheme);
</script>
"""

# Execute the JS and get the theme
theme_from_js = html(js_code, height=0, width=0)

if theme_from_js:
    st.session_state.theme = theme_from_js

logo_path = "cargo_logo_dark.png" if st.session_state.theme == "dark" else "cargo_logo.png"

# --- Custom CSS for a Polished Look & Responsiveness ---
st.markdown("""
<style>
    /* Increase base font size for the entire app */
    html, body, [class*="st-"] {
        font-size: 18px;
    }
    
    /* Responsive columns for KPI and Error tables */
    @media (max-width: 1200px) {
        div[data-testid="column"] {
            flex-wrap: wrap;
            flex: 1 1 100%; /* Allow columns to take full width and wrap */
            min-width: 300px; /* Prevent columns from becoming too narrow */
        }
    }

    /* Increase font size for tables generated from pandas .to_html() */
    table {
        font-size: 1.1rem !important; /* Make table text larger */
        width: 100%;
    }
    th {
        font-size: 1.2rem !important; /* Make table headers even larger */
        text-align: left;
        font-weight: bold;
    }
    /* Increase font size for native Streamlit DataFrames and their content */
    .stDataFrame, .stDataFrame [data-testid="stTable"] {
        font-size: 1.1rem !important;
    }
    /* Increase font size for widget labels (selectbox, etc.) */
    .st-emotion-cache-1y4p8pa {
        font-size: 1.2rem !important;
    }
    /* Increase font size for expander header */
    .st-emotion-cache-pwan1i {
        font-size: 1.2rem !important;
    }
    /* Increase font size for expander content */
    .st-emotion-cache-1hver84 {
        font-size: 1.1rem !important;
    }
    /* Style for the main section dividers */
    hr {
        height: 2px !important;
        border: none !important;
        background: linear-gradient(to right, #ff4b4b, #ffa421, #f9f871) !important; /* Red -> Orange -> Yellow */
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads all sheets from an Excel file into a dictionary of DataFrames."""
    try:
        xls = pd.ExcelFile(file_path)
        data = {}
        for sheet_name in xls.sheet_names:
            # Read summary sheets normally, but read daily sheets with the first row as the header
            if sheet_name in ['Master Summary', 'Error Summary']:
                data[sheet_name] = pd.read_excel(xls, sheet_name)
            else:
                data[sheet_name] = pd.read_excel(xls, sheet_name, header=0)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please run the report updater first.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Excel file: {e}", icon="🔥")
        return None

# --- Reusable Dashboard Function ---
def create_dashboard_view(summary_df, error_summary_df, all_data, key_prefix):
    """Creates a full dashboard view for a given dataset."""
    if summary_df is None or summary_df.empty:
        st.info("No data available for this view.", icon="📊")
        return

    # --- Helper function for calculating change ---
    def get_change(current, previous, is_time=False):
        """Calculates the percentage change and formats it for st.metric."""
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return None  # Return None for st.metric to display no delta
        
        change = ((current - previous) / previous) * 100
        return f"{change:.2f}%"

    # --- Prepare DataFrames ---
    summary_df = summary_df.copy()
    error_summary_df = error_summary_df.copy() if error_summary_df is not None else pd.DataFrame()

    # Rename 'Day' to 'Log' to match user request
    if 'Day' in summary_df.columns:
        summary_df.rename(columns={'Day': 'Log'}, inplace=True)

    # Convert relevant columns to numeric, coercing errors
    numeric_cols = [
        'Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 'Throughput (pkg/hr)',
        'Packages Stowed', 'Stow Attempts', 'Packages Retrieved', 'Retrieval Attempts',
        'Total Errors', 'Stow Driver Shift Time (hr)', 'Retrieve Driver Shift Time (hr)'
    ]
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    # --- Master Summary Section (No longer in a tab) ---
    st.header("Overall Performance", divider='orange')

    # --- KPI Tracker & Error Analysis ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KPI Tracker")
        if not summary_df.empty and len(summary_df) >= 2:
            last_day_stats = summary_df.iloc[-1]
            prev_day_stats = summary_df.iloc[-2]

            # --- Helpers for formatting changes ---
            def get_change(current, previous, is_time=False):
                if pd.isna(current) or pd.isna(previous) or previous == 0: return "N/A"
                change = ((current - previous) / previous) * 100
                color = "green" if (change < 0 if is_time else change > 0) else "red"
                return f'<span style="color:{color};">{change:+.2f}%</span>'

            def get_rate_change(current, previous):
                if pd.isna(current) or pd.isna(previous): return "N/A"
                change = current - previous
                color = "green" if change >= 0 else "red"
                return f'<span style="color:{color};">{change:+.2f} pts</span>'

            # --- Calculate Overall Averages & Rates ---
            stow_rate_overall = (summary_df['Packages Stowed'].sum() / summary_df['Stow Attempts'].sum() * 100) if summary_df['Stow Attempts'].sum() > 0 else 0
            retrieve_rate_overall = (summary_df['Packages Retrieved'].sum() / summary_df['Retrieval Attempts'].sum() * 100) if summary_df['Retrieval Attempts'].sum() > 0 else 0

            # --- Calculate Last & Previous Day Rates ---
            last_stow_rate = (last_day_stats['Packages Stowed'] / last_day_stats['Stow Attempts'] * 100) if last_day_stats['Stow Attempts'] > 0 else 0
            last_retrieval_rate = (last_day_stats['Packages Retrieved'] / last_day_stats['Retrieval Attempts'] * 100) if last_day_stats['Retrieval Attempts'] > 0 else 0
            prev_stow_rate = (prev_day_stats['Packages Stowed'] / prev_day_stats['Stow Attempts'] * 100) if prev_day_stats['Stow Attempts'] > 0 else 0
            prev_retrieval_rate = (prev_day_stats['Packages Retrieved'] / prev_day_stats['Retrieval Attempts'] * 100) if prev_day_stats['Retrieval Attempts'] > 0 else 0

            kpi_data = {
                'Metric': ['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 'Throughput (pkg/hr)', 'Stow Success Rate', 'Retrieval Success Rate'],
                'Overall Average': [
                    f"{summary_df['Stow Avg (s)'].mean():.2f}", f"{summary_df['Retrieve Avg (s)'].mean():.2f}",
                    f"{summary_df['Read Label Avg (s)'].mean():.2f}", f"{summary_df['Throughput (pkg/hr)'].mean():.2f}",
                    f"{stow_rate_overall:.2f}%", f"{retrieve_rate_overall:.2f}%"
                ],
                'Most Recent Day': [
                    f"{last_day_stats['Stow Avg (s)']:.2f}", f"{last_day_stats['Retrieve Avg (s)']:.2f}",
                    f"{last_day_stats['Read Label Avg (s)']:.2f}", f"{last_day_stats['Throughput (pkg/hr)']:.2f}",
                    f"{last_stow_rate:.2f}%", f"{last_retrieval_rate:.2f}%"
                ],
                'Change from Prev. Day': [
                    get_change(last_day_stats['Stow Avg (s)'], prev_day_stats['Stow Avg (s)'], is_time=True),
                    get_change(last_day_stats['Retrieve Avg (s)'], prev_day_stats['Retrieve Avg (s)'], is_time=True),
                    get_change(last_day_stats['Read Label Avg (s)'], prev_day_stats['Read Label Avg (s)'], is_time=True),
                    get_change(last_day_stats['Throughput (pkg/hr)'], prev_day_stats['Throughput (pkg/hr)']),
                    get_rate_change(last_stow_rate, prev_stow_rate),
                    get_rate_change(last_retrieval_rate, prev_retrieval_rate)
                ]
            }
            kpi_df = pd.DataFrame(kpi_data)
            st.markdown(kpi_df.to_html(escape=False, index=False, na_rep="N/A"), unsafe_allow_html=True)

        elif not summary_df.empty: # Handle case with only one day of data
            last_day_stats = summary_df.iloc[-1]
            stow_rate = (last_day_stats['Packages Stowed'] / last_day_stats['Stow Attempts'] * 100) if last_day_stats['Stow Attempts'] > 0 else 0
            retrieval_rate = (last_day_stats['Packages Retrieved'] / last_day_stats['Retrieval Attempts'] * 100) if last_day_stats['Retrieval Attempts'] > 0 else 0
            
            # For a single day, Overall Average is the same as Most Recent Day
            most_recent_values = [
                f"{last_day_stats['Stow Avg (s)']:.2f}" if pd.notna(last_day_stats['Stow Avg (s)']) else "N/A",
                f"{last_day_stats['Retrieve Avg (s)']:.2f}" if pd.notna(last_day_stats['Retrieve Avg (s)']) else "N/A",
                f"{last_day_stats['Read Label Avg (s)']:.2f}" if pd.notna(last_day_stats['Read Label Avg (s)']) else "N/A",
                f"{last_day_stats['Throughput (pkg/hr)']:.2f}" if pd.notna(last_day_stats['Throughput (pkg/hr)']) else "N/A",
                f"{stow_rate:.2f}%",
                f"{retrieval_rate:.2f}%"
            ]
            
            kpi_data = {
                'Metric': ['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 'Throughput (pkg/hr)', 'Stow Success Rate', 'Retrieval Success Rate'],
                'Overall Average': most_recent_values,
                'Most Recent Day': most_recent_values
            }
            kpi_df = pd.DataFrame(kpi_data)
            st.markdown(kpi_df.to_html(escape=False, index=False, na_rep="N/A"), unsafe_allow_html=True)
        else:
            st.info("No KPI data available.")

    with col2:
        st.subheader("Error Analysis")
        if not error_summary_df.empty and len(summary_df) >= 2:
            error_categories = [col for col in error_summary_df.columns if col not in ['Day', 'Date']]
            # Ensure error counts are numeric, filling any non-numeric with 0
            for col in error_categories:
                error_summary_df[col] = pd.to_numeric(error_summary_df[col], errors='coerce').fillna(0)

            total_errors_by_category = error_summary_df[error_categories].sum()
            total_errors_overall = total_errors_by_category.sum()

            last_day_errors = error_summary_df.iloc[-1]
            prev_day_errors = error_summary_df.iloc[-2]
            
            last_day_total = last_day_errors[error_categories].sum()
            prev_day_total = prev_day_errors[error_categories].sum()

            error_analysis_data = []
            for cat in error_categories:
                if total_errors_by_category[cat] == 0: continue
                
                overall_pct = (total_errors_by_category[cat] / total_errors_overall * 100) if total_errors_overall > 0 else 0
                last_day_pct = (last_day_errors.get(cat, 0) / last_day_total * 100) if last_day_total > 0 else 0
                prev_day_pct = (prev_day_errors.get(cat, 0) / prev_day_total * 100) if prev_day_total > 0 else 0
                
                change_in_pct = last_day_pct - prev_day_pct
                color = "red" if change_in_pct > 0 else "green"
                change_str = f'<span style="color:{color};">{change_in_pct:+.2f} pts</span>'

                error_analysis_data.append({
                    'Error Category': cat, 'Overall % of Total Errors': overall_pct,
                    'Most Recent Day %': last_day_pct, 'Change from Prev. Day': change_str
                })
            
            sorted_error_data = sorted(error_analysis_data, key=lambda x: x['Overall % of Total Errors'], reverse=True)
            final_error_data = sorted_error_data[:7]
            if len(sorted_error_data) > 7:
                others_data = sorted_error_data[7:]
                final_error_data.append({
                    'Error Category': 'Others',
                    'Overall % of Total Errors': sum(d['Overall % of Total Errors'] for d in others_data),
                    'Most Recent Day %': sum(d['Most Recent Day %'] for d in others_data),
                    'Change from Prev. Day': 'N/A'
                })
            
            for item in final_error_data:
                item['Overall % of Total Errors'] = f"{item['Overall % of Total Errors']:.2f}%"
                item['Most Recent Day %'] = f"{item['Most Recent Day %']:.2f}%"
            
            error_df = pd.DataFrame(final_error_data)
            st.markdown(error_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        elif not error_summary_df.empty: # Handle case with only one day of error data
            error_categories = [col for col in error_summary_df.columns if col not in ['Day', 'Date']]
            for col in error_categories:
                error_summary_df[col] = pd.to_numeric(error_summary_df[col], errors='coerce').fillna(0)

            last_day_errors = error_summary_df.iloc[-1]
            total_errors_last_day = last_day_errors[error_categories].sum()

            error_analysis_data = []
            if total_errors_last_day > 0:
                for cat in error_categories:
                    count = last_day_errors.get(cat, 0)
                    if count > 0:
                        error_analysis_data.append({
                            'Error Category': cat,
                            '% of Total Errors': (count / total_errors_last_day * 100)
                        })
                
                sorted_error_data = sorted(error_analysis_data, key=lambda x: x['% of Total Errors'], reverse=True)
                
                for item in sorted_error_data:
                    item['% of Total Errors'] = f"{item['% of Total Errors']:.2f}%"
                
                error_df = pd.DataFrame(sorted_error_data)
                st.markdown(error_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No errors recorded for this day.")
        else:
            st.info("Not enough data for error analysis.")

    # --- Carrier Benchmark Comparison ---
    st.header("Carrier Benchmark Comparison", divider='orange')
    benchmark_cols_exist = (
        'Stow Driver Shift Time (hr)' in summary_df.columns and 
        'Retrieve Driver Shift Time (hr)' in summary_df.columns and
        len(summary_df) >= 2
    )

    if benchmark_cols_exist:
        last_day_stats = summary_df.iloc[-1]
        prev_day_stats = summary_df.iloc[-2]

        def get_benchmark_change(our_value, benchmark_value):
            if pd.isna(our_value) or pd.isna(benchmark_value) or benchmark_value == 0: return "N/A"
            change = ((our_value - benchmark_value) / benchmark_value) * 100
            color = "red" if change > 0 else "green"
            return f'<span style="color:{color};">{change:+.2f}%</span>'

        stow_time_per_pkg_last = (last_day_stats['Stow Driver Shift Time (hr)'] / last_day_stats['Packages Stowed']) if last_day_stats['Packages Stowed'] > 0 else float('nan')
        retrieve_time_per_pkg_last = (last_day_stats['Retrieve Driver Shift Time (hr)'] / last_day_stats['Packages Retrieved']) if last_day_stats['Packages Retrieved'] > 0 else float('nan')
        
        stow_time_per_pkg_prev = (prev_day_stats['Stow Driver Shift Time (hr)'] / prev_day_stats['Packages Stowed']) if prev_day_stats['Packages Stowed'] > 0 else float('nan')
        retrieve_time_per_pkg_prev = (prev_day_stats['Retrieve Driver Shift Time (hr)'] / prev_day_stats['Packages Retrieved']) if prev_day_stats['Packages Retrieved'] > 0 else float('nan')

        avg_stow_time_per_pkg = (summary_df['Stow Driver Shift Time (hr)'] / summary_df['Packages Stowed']).mean()
        avg_retrieve_time_per_pkg = (summary_df['Retrieve Driver Shift Time (hr)'] / summary_df['Packages Retrieved']).mean()

        # --- Benchmark Selection ---
        benchmark_choice = st.selectbox(
            "Select a benchmark to view:",
            options=['FedEx', 'Amazon'],
            index=0,  # Default to FedEx
            key=f"{key_prefix}_benchmark_select" # FIX: Add a unique key
        )

        if benchmark_choice == 'FedEx':
            # --- FedEx Table ---
            fedex_pkgs, fedex_stow_hrs, fedex_retrieve_hrs = 150, 3.0, 3.0
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
            st.markdown(f"<h4>FedEx Benchmark ({fedex_pkgs} pkgs/van)</h4>", unsafe_allow_html=True)
            st.markdown(pd.DataFrame(fedex_data).to_html(escape=False, index=False), unsafe_allow_html=True)

        elif benchmark_choice == 'Amazon':
            # --- Amazon Table ---
            amazon_pkgs, amazon_stow_hrs, amazon_retrieve_hrs = 219, 0.3, 3.9
            our_stow_amazon_avg = avg_stow_time_per_pkg * amazon_pkgs
            our_retrieve_amazon_avg = avg_retrieve_time_per_pkg * amazon_pkgs
            our_stow_amazon_last = stow_time_per_pkg_last * amazon_pkgs
            our_retrieve_amazon_last = retrieve_time_per_pkg_last * amazon_pkgs
            our_stow_amazon_prev = stow_time_per_pkg_prev * amazon_pkgs
            our_retrieve_amazon_prev = retrieve_time_per_pkg_prev * amazon_pkgs

            amazon_data = {
                'Metric': ["Stow (Loading) Time (hr)", "Retrieve (Unloading) Time (hr)"],
                'Overall Average': [f"{our_stow_amazon_avg:.2f}", f"{our_retrieve_amazon_avg:.2f}"],
                'Most Recent Day': [f"{our_stow_amazon_last:.2f}", f"{our_retrieve_amazon_last:.2f}"],
                'Change from Previous Day': [get_change(our_stow_amazon_last, our_stow_amazon_prev, is_time=True), get_change(our_retrieve_amazon_last, our_retrieve_amazon_prev, is_time=True)],
                'Benchmark': [f"{amazon_stow_hrs:.1f}", f"{amazon_retrieve_hrs:.1f}"],
                'Change from Benchmark': [get_benchmark_change(our_stow_amazon_avg, amazon_stow_hrs), get_benchmark_change(our_retrieve_amazon_avg, amazon_retrieve_hrs)]
            }
            st.markdown(f"<h4>Amazon Benchmark ({amazon_pkgs} pkgs/van)</h4>", unsafe_allow_html=True)
            st.markdown(pd.DataFrame(amazon_data).to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("Not enough data for benchmark comparison (requires at least 2 days of data and driver shift times).")

    # --- Trend Charts ---
    st.header("Performance Trends", divider='orange')
    if len(summary_df) > 1:
        # Ensure Log column is integer for axis ticks
        if 'Log' in summary_df.columns:
            summary_df['Log'] = summary_df['Log'].astype(int)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>Average Process Times</h4>", unsafe_allow_html=True)
            fig = px.line(summary_df, x='Log', y=['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)'], markers=True)
            fig.update_xaxes(dtick=1) # Set x-axis ticks to integers
            fig.update_yaxes(title_text='Time (s)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<h4>Packages Stowed vs. Attempts</h4>", unsafe_allow_html=True)
            fig2 = go.Figure(data=[
                go.Bar(name='Packages Stowed', x=summary_df['Log'], y=summary_df['Packages Stowed']),
                go.Bar(name='Stow Attempts', x=summary_df['Log'], y=summary_df['Stow Attempts'])
            ])
            fig2.update_layout(barmode='group', yaxis_title='Count')
            fig2.update_xaxes(dtick=1, title_text='Log')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("<h4>Throughput Over Time</h4>", unsafe_allow_html=True)
            fig3 = px.line(summary_df, x='Log', y='Throughput (pkg/hr)', markers=True)
            fig3.update_xaxes(dtick=1)
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("<h4>Packages Retrieved vs. Attempts</h4>", unsafe_allow_html=True)
            fig4 = go.Figure(data=[
                go.Bar(name='Packages Retrieved', x=summary_df['Log'], y=summary_df['Packages Retrieved']),
                go.Bar(name='Retrieval Attempts', x=summary_df['Log'], y=summary_df['Retrieval Attempts'])
            ])
            fig4.update_layout(barmode='group', yaxis_title='Count')
            fig4.update_xaxes(dtick=1, title_text='Log')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Need at least two days of data to show trend charts.")

    # --- Day-by-Day Table ---
    st.header("Daily Performance", divider='orange')
    st.subheader("Day-by-Day Performance")

    # Create a display-ready version of the dataframe to ensure 'N/A' is shown correctly.
    display_df = summary_df.copy()

    # Define which columns need specific decimal formatting.
    cols_to_format = [
        'Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 
        'Throughput (pkg/hr)', 'Stow Driver Shift Time (hr)', 
        'Retrieve Driver Shift Time (hr)'
    ]

    # Apply formatting: 2 decimal places for numbers, and 'N/A' for any missing values.
    for col in cols_to_format:
        if col in display_df.columns:
            # The summary_df was already converted to numeric, so we can safely format.
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

    # Display the fully formatted dataframe.
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Daily Detail Section ---
    st.subheader("Daily Detailed Report")
    day_sheets = sorted([s for s in all_data.keys() if s.startswith('Day ')], key=lambda x: int(x.split(' ')[1]))

    if day_sheets:
        selected_day = st.selectbox(
            "Select a day to inspect:",
            options=day_sheets,
            key=f"{key_prefix}_day_select" # Use the key_prefix
        )

        if selected_day:
            day_df = all_data[selected_day]

            # --- Find the end of the main event logs (before summary stats) ---
            summary_start_indices = day_df.index[day_df.iloc[:, 0].astype(str).str.contains('Average', na=False)]
            event_log_end_row = summary_start_indices.min() - 2 if not summary_start_indices.empty else len(day_df)

            # --- Extract Event Logs and find the last row for positioning the error log ---
            stow_events = day_df.iloc[:event_log_end_row, 0:3].dropna(how='all')
            retrieve_events = day_df.iloc[:event_log_end_row, 3:6].dropna(how='all')
            read_label_events = day_df.iloc[:event_log_end_row, 8:11].dropna(how='all')

            # --- Standardize Column Headers ---
            clean_headers = ['Timestamp', 'Event Label', 'Time (s)']
            
            if stow_events.empty:
                stow_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                stow_events.columns = clean_headers

            if retrieve_events.empty:
                retrieve_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                retrieve_events.columns = clean_headers

            if read_label_events.empty:
                read_label_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                if read_label_events.shape[1] == 3:
                    read_label_events.columns = clean_headers
                else:
                    read_label_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)

            last_stow_idx = stow_events.index[-1] if not stow_events.empty else -1
            last_retrieve_idx = retrieve_events.index[-1] if not retrieve_events.empty else -1
            last_read_label_idx = read_label_events.index[-1] if not read_label_events.empty else -1
            
            furthest_down_row = max(last_stow_idx, last_retrieve_idx, last_read_label_idx)

            # --- Display Summaries and Event Logs in Columns ---
            st.subheader("Event Logs & Summaries")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("<h4>Stow Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    stow_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 0:2].dropna(how='all', axis=1).reset_index(drop=True)
                    if stow_summary_df.shape[1] == 2:
                        stow_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        stow_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    stow_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(stow_summary_df, use_container_width=True, hide_index=True)
                
                st.markdown("<h4>Stow Events</h4>", unsafe_allow_html=True)
                stow_df_display = stow_events.reset_index(drop=True)
                stow_df_display.index = stow_df_display.index + 1
                stow_df_display.index.name = '#'
                st.dataframe(stow_df_display, use_container_width=True)

            with col2:
                st.markdown("<h4>Retrieve Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    retrieve_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 3:5].dropna(how='all', axis=1).reset_index(drop=True)
                    if retrieve_summary_df.shape[1] == 2:
                        retrieve_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        retrieve_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    retrieve_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(retrieve_summary_df, use_container_width=True, hide_index=True)

                st.markdown("<h4>Retrieve Events</h4>", unsafe_allow_html=True)
                retrieve_df_display = retrieve_events.reset_index(drop=True)
                retrieve_df_display.index = retrieve_df_display.index + 1
                retrieve_df_display.index.name = '#'
                st.dataframe(retrieve_df_display, use_container_width=True)

            with col3:
                st.markdown("<h4>Read Label Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    read_label_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 8:10].dropna(how='all', axis=1).reset_index(drop=True)
                    if read_label_summary_df.shape[1] == 2:
                        read_label_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        read_label_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    read_label_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(read_label_summary_df, use_container_width=True, hide_index=True)

                st.markdown("<h4>Read Label Events</h4>", unsafe_allow_html=True)
                read_label_df_display = read_label_events.reset_index(drop=True)
                read_label_df_display.index = read_label_df_display.index + 1
                read_label_df_display.index.name = '#'
                st.dataframe(read_label_df_display, use_container_width=True)

            # --- Display Failures Section (Full Width) ---
            st.subheader("Failures")
            st.markdown("<h4>Failures Logged</h4>", unsafe_allow_html=True)
            if furthest_down_row != -1:
                error_start_row = furthest_down_row + 7 # Start 7 rows down
                
                # Extract the block of potential error data
                error_block = day_df.iloc[error_start_row:]
                
                # Find the first completely blank row to mark the end of the table
                end_row_indices = error_block.index[error_block.isnull().all(axis=1)]
                error_end_row = end_row_indices.min() if not end_row_indices.empty else len(day_df)
                
                # Dynamically find the start of the error table instead of assuming it's at column 0
                potential_error_df = day_df.iloc[error_start_row:error_end_row]
                
                # Find the first column that is not entirely empty
                first_valid_col = 0
                for col in range(potential_error_df.shape[1]):
                    if not potential_error_df.iloc[:, col].isnull().all():
                        first_valid_col = col
                        break
                
                # Select the 3 columns of the error table from the dynamic start
                error_df = potential_error_df.iloc[:, first_valid_col:first_valid_col+3].dropna(how='all')

                if not error_df.empty and error_df.shape[1] == 3:
                    # Check if the first row looks like headers before assigning
                    if not pd.api.types.is_numeric_dtype(error_df.iloc[0, 0]) and isinstance(error_df.iloc[0, 0], str):
                        error_df.columns = error_df.iloc[0].astype(str)
                        error_df = error_df[1:].reset_index(drop=True)
                    
                    error_df.index = error_df.index + 1
                    error_df.index.name = '#'
                    st.dataframe(error_df, use_container_width=True)
                else:
                    st.write("No failure data found in expected location.")
            else:
                st.write("Could not determine where to look for failure data.")
    else:
        st.info("No daily detail sheets found in the report to display.")

# --- Main Application ---
# Load data from BOTH report files
road_test_data = load_data('master_report.xlsx')
benchmark_data = load_data('benchmark_report.xlsx')

# Display the logo and the title underneath
st.image(logo_path, width=400)
st.title("Performance Dashboard")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["Road Tests", "In-House Benchmarks"])

with tab1:
    if road_test_data:
        summary_df = road_test_data.get('Master Summary')
        error_summary_df = road_test_data.get('Error Summary')
        create_dashboard_view(summary_df, error_summary_df, road_test_data, "road_test")
    else:
        st.info("No Road Test data found. Please process logs to generate the `master_report.xlsx` file.", icon="🛣️")

with tab2:
    if benchmark_data:
        benchmark_summary_df = benchmark_data.get('Master Summary')
        # Benchmarks will have their own error summary within their own report
        error_summary_df = benchmark_data.get('Error Summary') 
        create_dashboard_view(benchmark_summary_df, error_summary_df, benchmark_data, "benchmark")
    else:
        st.info("No Benchmark data found. Please process logs to generate the `benchmark_report.xlsx` file.", icon="🔬")
