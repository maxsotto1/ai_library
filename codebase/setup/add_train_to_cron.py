import os
import sys
import subprocess

def add_to_cron():
    # 1. Gather current execution context
    python_path = sys.executable  # Path to current Python virtualenv/environment
    working_dir = os.getcwd()      # Directory where this script is run
    
    # 2. Construct the full command
    # Runs python -m codebase.setup.train and appends logs to cron.log
    log_file = os.path.join(working_dir, "cron.log")
    command = f"cd {working_dir} && {python_path} -m codebase.setup.train >> {log_file} 2>&1"
    
    # 3. Define schedule: Every 30 minutes (*/30 * * * *)
    cron_schedule = "*/30 * * * *"
    cron_entry = f"{cron_schedule} {command}"
    
    # 4. Fetch existing crontab
    try:
        current_cron = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        current_cron = ""  # No crontab exists for this user yet
        
    # 5. Avoid adding duplicate entries
    if command in current_cron:
        print("Job already exists in crontab. No changes made.")
        return

    # 6. Append new job and update crontab
    new_cron = current_cron + cron_entry + "\n"
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(input=new_cron)
    
    if process.returncode == 0:
        print("Successfully added job to cron!")
        print(f"Scheduled entry:\n{cron_entry}")
    else:
        print("Failed to update crontab.", file=sys.stderr)

if __name__ == "__main__":
    add_to_cron()