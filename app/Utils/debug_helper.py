# Utils/debug_helper.py

def write_debug_log(message: str):
    """Appends a debug message to a log file."""
    with open("/app/debug_log.txt", "a") as log_file:  # Adjust the path if needed
        log_file.write(f"{message}\n")