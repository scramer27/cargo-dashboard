import os
import subprocess
import shutil
from datetime import datetime
import json

# Paths
LOGS_FOLDER = "logs"
ARCHIVE_FOLDER = "archive"
MASTER_REPORT = "master_report.xlsx"
REPORT_UPDATER = "report_updater.py"
PROCESSED_LOGS_FILE = os.path.join(ARCHIVE_FOLDER, "processed_logs.txt")

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

def load_processed_logs():
    """Load the list of already processed log files."""
    if not os.path.exists(PROCESSED_LOGS_FILE):
        return set()
    with open(PROCESSED_LOGS_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_processed_log(log_file):
    """Add a log file to the processed logs list."""
    with open(PROCESSED_LOGS_FILE, "a") as f:
        f.write(log_file + "\n")

def rename_new_logs():
    """Rename new logs.json files with a timestamp to ensure uniqueness."""
    for file in os.listdir(LOGS_FOLDER):
        if file == "logs.json":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"logs_{timestamp}.json"
            os.rename(os.path.join(LOGS_FOLDER, file), os.path.join(LOGS_FOLDER, new_name))
            print(f"Renamed {file} to {new_name}")

def process_new_logs():
    """Process new log files and update master_report.xlsx."""
    # Ensure archive folder exists
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

    # Load already processed logs
    processed_logs = load_processed_logs()

    # Find all log files in the logs folder
    log_files_to_process = []
    for f in os.listdir(LOGS_FOLDER):
        if f.endswith(".json") and f not in processed_logs:
            log_path = os.path.join(LOGS_FOLDER, f)
            timestamp = get_first_timestamp(log_path)
            if timestamp:
                log_files_to_process.append((timestamp, f))

    # Sort files chronologically
    log_files_to_process.sort()

    log_files = [f for timestamp, f in log_files_to_process]

    if not log_files:
        print("No new log files found to process.")
        return

    # Process each log file
    for log_file in log_files:
        log_path = os.path.join(LOGS_FOLDER, log_file)
        print(f"Processing log file: {log_file}...")

        # Run report_updater.py with the log file
        try:
            subprocess.run(
                ["python3", REPORT_UPDATER, MASTER_REPORT, log_path],
                check=True
            )
            print(f"Successfully processed {log_file}.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {log_file}: {e}")
            continue

        # Move the processed log file to the archive folder
        shutil.move(log_path, os.path.join(ARCHIVE_FOLDER, log_file))
        print(f"Archived {log_file}.")

        # Mark the log file as processed
        save_processed_log(log_file)

def push_to_github():
    """Push the updated master_report.xlsx to GitHub."""
    # Stage the updated master_report.xlsx
    subprocess.run(["git", "add", MASTER_REPORT], check=True)

    # Check if there are any changes to commit
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        print("No changes to commit.")
        return

    # Commit the changes
    subprocess.run(
        ["git", "commit", "-m", "chore: Update master_report.xlsx with new log data"],
        check=True
    )

    # Push to GitHub
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Pushed changes to GitHub.")

def reset_system():
    """Reset the system by deleting master_report.xlsx, clearing processed_logs.txt, and moving archived logs back to logs."""
    # Delete master_report.xlsx
    if os.path.exists(MASTER_REPORT):
        os.remove(MASTER_REPORT)
        print(f"Deleted {MASTER_REPORT}.")
    else:
        print(f"{MASTER_REPORT} does not exist.")

    # Clear processed_logs.txt (but keep it in the archive folder)
    if os.path.exists(PROCESSED_LOGS_FILE):
        with open(PROCESSED_LOGS_FILE, "w") as f:
            pass  # Overwrite with an empty file
        print(f"Cleared {PROCESSED_LOGS_FILE}.")
    else:
        print(f"{PROCESSED_LOGS_FILE} does not exist.")

    # Move all files from archive back to logs (except processed_logs.txt)
    if os.path.exists(ARCHIVE_FOLDER):
        for file_name in os.listdir(ARCHIVE_FOLDER):
            file_path = os.path.join(ARCHIVE_FOLDER, file_name)
            if os.path.isfile(file_path) and file_name != "processed_logs.txt":  # Skip processed_logs.txt
                shutil.move(file_path, os.path.join(LOGS_FOLDER, file_name))
                print(f"Moved {file_name} from archive to logs.")
    else:
        print(f"{ARCHIVE_FOLDER} does not exist.")

    print("System reset complete.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        # Reset the system if the "reset" command is passed
        reset_system()
    else:
        # Rename new logs.json files
        rename_new_logs()

        # Process new logs
        process_new_logs()

        # Push changes to GitHub
        push_to_github()