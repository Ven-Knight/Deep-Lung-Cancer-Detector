import os
import sys
import logging

# ────────────────────────────────────────────────────────────────────────────────────────
# Define Logging Format String
# ────────────────────────────────────────────────────────────────────────────────────────
# Format includes timestamp, log level, module name, and message for better traceability
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# ────────────────────────────────────────────────────────────────────────────────────────
# Log Directory and File Setup
# ────────────────────────────────────────────────────────────────────────────────────────
# Create a directory named 'logs' to store log files
log_dir      = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Ensure the log directory exists (no error if already present)
os.makedirs(log_dir, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────────────
# Configure Logging Handlers
# ────────────────────────────────────────────────────────────────────────────────────────
# Logs will be written both to a file and streamed to stdout (console) 
logging.basicConfig(
                        level    = logging.INFO,                         # Set logging level to INFO
                        format   = logging_str,                          # Use the defined format string
                        handlers = [
                                    logging.FileHandler  (log_filepath), # Write logs to file
                                    logging.StreamHandler(sys.stdout)    # Stream logs to console
                                   ]
                   )

# ────────────────────────────────────────────────────────────────────────────────────────
# Create a Named Logger Instance
# ────────────────────────────────────────────────────────────────────────────────────────
# This logger can be imported and reused across modules for consistent logging
logger        = logging.getLogger("cnnClassifierLogger")