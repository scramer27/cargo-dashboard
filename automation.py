import os
import subprocess
import shutil
from datetime import datetime
import json

# --- Constants ---
# Road Test Paths
LOGS_FOLDER = "logs"
ARCHIVE_FOLDER = "archive"
PROCESSED_LOGS_FILE = os.path.join(ARCHIVE_FOLDER, "processed_logs.txt")
MASTER_REPORT = "master_report.xlsx" # Report for Road Tests
ROAD_TEST_SHEET = "Master Summary"

# Benchmark Paths
LOGS_BENCHMARK_FOLDER = "logs_benchmark"
ARCHIVE_BENCHMARK_FOLDER = "archive_benchmark"
PROCESSED_LOGS_BENCHMARK_FILE = os.path.join(ARCHIVE_BENCHMARK_FOLDER, "processed_logs_benchmark.txt")
BENCHMARK_REPORT = "benchmark_report.xlsx" # A separate report for Benchmarks
BENCHMARK_SHEET = "Master Summary" # The main sheet inside the benchmark report

# Common Paths
REPORT_UPDATER = "report_updater.py"


def get_first_timestamp(log_path):
    """Extracts the first timestamp from a JSONL log file."""
    try:
        with open(log_path, 'r') as f:
            first_line = f.readline()
            if first_line:
                log_entry = json.loads(first_line)
                return log_entry.get('timestamp')
    except (IOError, json.JSONDecodeError, IndexError):
        pass
    return None

def load_processed_logs(processed_file_path):
    """Load the list of already processed log files from a given path."""
    if not os.path.exists(processed_file_path):
        return set()
    with open(processed_file_path, "r") as f:
        return set(line.strip() for line in f)

def save_processed_log(log_file, processed_file_path):
    """Add a log file to a given processed logs list."""
    with open(processed_file_path, "a") as f:
        f.write(log_file + "\n")

def rename_new_logs(logs_dir):
    """Rename new logs.json files in a given directory."""
    if not os.path.exists(logs_dir):
        return
    for file in os.listdir(logs_dir):
        if file == "logs.json":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"logs_{timestamp}.json"
            os.rename(os.path.join(logs_dir, file), os.path.join(logs_dir, new_name))
            print(f"Renamed {file} in {logs_dir} to {new_name}")

def process_logs(logs_folder, archive_folder, processed_logs_file, report_file, target_sheet):
    """Generic function to process log files and update a target report file."""
    print(f"\n--- Processing logs for report: '{report_file}' ---")
    # Ensure source and archive folders exist
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(archive_folder, exist_ok=True)

    # Load already processed logs
    processed_logs = load_processed_logs(processed_logs_file)

    # Find all log files in the specified logs folder
    log_files_to_process = []
    for f in os.listdir(logs_folder):
        if f.endswith(".json") and f not in processed_logs:
            log_path = os.path.join(logs_folder, f)
            timestamp = get_first_timestamp(log_path)
            if timestamp:
                log_files_to_process.append((timestamp, f))

    # Sort files chronologically
    log_files_to_process.sort()

    log_files = [f for timestamp, f in log_files_to_process]

    if not log_files:
        print(f"No new log files found in '{logs_folder}'.")
        return

    # Process each log file
    for log_file in log_files:
        log_path = os.path.join(logs_folder, log_file)
        print(f"Processing log file: {log_file}...")

        # Run report_updater.py with the log file, target report, and target sheet
        try:
            subprocess.run(
                ["python3", REPORT_UPDATER, report_file, log_path, target_sheet],
                check=True
            )
            print(f"Successfully processed {log_file} for report '{report_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {log_file}: {e}")
            continue

        # Move the processed log file to its archive folder
        shutil.move(log_path, os.path.join(archive_folder, log_file))
        print(f"Archived {log_file} to '{archive_folder}'.")

        # Mark the log file as processed
        save_processed_log(log_file, processed_logs_file)

def push_to_github():
    """Push the updated reports and archived logs to GitHub."""
    # Stage the updated reports and both archive folders
    subprocess.run(["git", "add", MASTER_REPORT, BENCHMARK_REPORT, ARCHIVE_FOLDER, ARCHIVE_BENCHMARK_FOLDER], check=True)

    # Check if there are any changes to commit
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        print("No changes to commit.")
        return

    # Commit the changes with a more descriptive message
    subprocess.run(
        ["git", "commit", "-m", "feat: Process new log data and update reports"],
        check=True
    )

    # Push to GitHub
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Pushed changes to GitHub.")

def reset_system():
    """Reset the system by deleting reports, clearing processed logs, and moving archived logs back."""
    # Delete both master reports
    if os.path.exists(MASTER_REPORT):
        os.remove(MASTER_REPORT)
        print(f"Deleted {MASTER_REPORT}.")
    if os.path.exists(BENCHMARK_REPORT):
        os.remove(BENCHMARK_REPORT)
        print(f"Deleted {BENCHMARK_REPORT}.")

    # --- Reset Road Test Logs ---
    print("\nResetting Road Test environment...")
    if os.path.exists(PROCESSED_LOGS_FILE):
        with open(PROCESSED_LOGS_FILE, "w") as f:
            pass
        print(f"Cleared {PROCESSED_LOGS_FILE}.")
    if os.path.exists(ARCHIVE_FOLDER):
        for file_name in os.listdir(ARCHIVE_FOLDER):
            file_path = os.path.join(ARCHIVE_FOLDER, file_name)
            if os.path.isfile(file_path) and file_name != "processed_logs.txt":
                shutil.move(file_path, os.path.join(LOGS_FOLDER, file_name))
        print(f"Moved logs from {ARCHIVE_FOLDER} to {LOGS_FOLDER}.")

    # --- Reset Benchmark Logs ---
    print("\nResetting Benchmark environment...")
    os.makedirs(LOGS_BENCHMARK_FOLDER, exist_ok=True) # Ensure folder exists
    if os.path.exists(PROCESSED_LOGS_BENCHMARK_FILE):
        with open(PROCESSED_LOGS_BENCHMARK_FILE, "w") as f:
            pass
        print(f"Cleared {PROCESSED_LOGS_BENCHMARK_FILE}.")
    if os.path.exists(ARCHIVE_BENCHMARK_FOLDER):
        for file_name in os.listdir(ARCHIVE_BENCHMARK_FOLDER):
            file_path = os.path.join(ARCHIVE_BENCHMARK_FOLDER, file_name)
            if os.path.isfile(file_path) and file_name != "processed_logs_benchmark.txt":
                shutil.move(file_path, os.path.join(LOGS_BENCHMARK_FOLDER, file_name))
        print(f"Moved logs from {ARCHIVE_BENCHMARK_FOLDER} to {LOGS_BENCHMARK_FOLDER}.")

    print("\nSystem reset complete.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_system()
    else:
        # Rename new logs in both directories
        rename_new_logs(LOGS_FOLDER)
        rename_new_logs(LOGS_BENCHMARK_FOLDER)

        # Process new logs for both types, writing to their respective reports
        process_logs(LOGS_FOLDER, ARCHIVE_FOLDER, PROCESSED_LOGS_FILE, MASTER_REPORT, ROAD_TEST_SHEET)
        process_logs(LOGS_BENCHMARK_FOLDER, ARCHIVE_BENCHMARK_FOLDER, PROCESSED_LOGS_BENCHMARK_FILE, BENCHMARK_REPORT, BENCHMARK_SHEET)

        # Push changes to GitHub
        #push_to_github()