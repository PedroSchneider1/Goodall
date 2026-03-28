# settings.py
import os

from enum import Enum

class Files(Enum):
    """
    Enum for indexing the file paths in the settings.
    """
    OUTPUT_DIR = 0
    FILES_FOLDER_PATH = 1
    EVENTS_CSV = 2
    TEMPLATES_CACHE_FILE = 3
    TRAIN_CACHE_FILE = 4
    TEST_CACHE_FILE = 5
    SIMILARITIES_CACHE_FILE = 6

def init():
    """
    Initialize file and folder paths for the algorithm settings.
    """
    global filesList
    
    OUTPUT_DIR = "./caches/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    FILES_FOLDER_PATH = "./dataset/"
    os.makedirs(FILES_FOLDER_PATH, exist_ok=True)
    
    EVENTS_CSV = "./dataset/csv/strikes_events.csv"
    
    TEMPLATES_CACHE_FILE = os.path.join(OUTPUT_DIR, "templates.npz")
    TRAIN_CACHE_FILE = os.path.join(OUTPUT_DIR, "train.npz")
    TEST_CACHE_FILE = os.path.join(OUTPUT_DIR, "test.npz")
    SIMILARITIES_CACHE_FILE = os.path.join(OUTPUT_DIR, 'similarities.npz')
    
    filesList = [OUTPUT_DIR, FILES_FOLDER_PATH, EVENTS_CSV,
            TEMPLATES_CACHE_FILE, TRAIN_CACHE_FILE,
            TEST_CACHE_FILE, SIMILARITIES_CACHE_FILE]