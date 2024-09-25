import logging
import os

# script_name = os.path.splitext(os.path.basename(__file__))[0]
script_name = 'mtst'
error_log_file = f"{script_name}_error.log"
info_log_file = f"{script_name}_info.log"

# Check if the log files exist and delete them if they do
# if os.path.exists(error_log_file):
#     os.remove(error_log_file)
# if os.path.exists(info_log_file):
#     os.remove(info_log_file)

# Create FileHandlers for different log files (info and error)
error_file_handler = logging.FileHandler(error_log_file)
error_file_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_file_handler.setFormatter(error_formatter)

info_file_handler = logging.FileHandler(info_log_file)
info_file_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_file_handler.setFormatter(info_formatter)

logging.basicConfig(
    format="%(levelname)s - %(message)s",
    # level=logging.INFO,  # Minimum level for basicConfig
    level=logging.NOTSET,  # Set to NOTSET to allow handlers to determine levels
    # handlers=[file_handler, logging.StreamHandler()]
    handlers=[error_file_handler, info_file_handler]  # Both FileHandlers
)

logger = logging.getLogger(__name__)


#     logger.setLevel(logging.DEBUG)

#     # Create file handler which logs messages
#     file_handler = logging.FileHandler(log_file_name)
#     file_handler.setLevel(logging.DEBUG)

#     # Create a formatter and set the format for log messages
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)

#     # Add the file handler to the logger
#     logger.addHandler(file_handler)

#     return logger

# logger = logging.getLogger()
# logger.setLevel(logging.NOTSET)  # Set the lowest level

# # Add the FileHandlers to the logger
# logger.addHandler(error_file_handler)
# logger.addHandler(info_file_handler)

# Set up coloredlogs for improved console logging
# coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')