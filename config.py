# config.py

# Default configuration values; these can be overridden by metadata sent via the API.
DEFAULT_VIDEO_PATH = "data/video/sun.mp4"  # Default video used for avatar creation.
DEFAULT_BBOX_SHIFT = 5                     # Default bbox shift (can be auto‐handled later).
DEFAULT_BATCH_SIZE = 4
DEFAULT_FPS = 25
SKIP_SAVE_IMAGES = True                    # Set to True for faster processing (won't save intermediate images).
TEMP_DIR = "temp"                          # Directory for temporary files.
RESULTS_DIR = "results/avatars"            # Directory where avatar data (and generated videos) are stored.
