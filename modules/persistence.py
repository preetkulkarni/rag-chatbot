import os
import pickle
import faiss
from modules import config

CACHE_DIR = config.CACHED_DIR

def save_data(filename, data, serializer):
    # saves data using specified serializer
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    full_path = os.path.join(CACHE_DIR, filename)

    try:
        if serializer == 'pickle':
            with open(full_path, "wb") as f:
                pickle.dump(data, f)
        elif serializer == 'faiss':
            faiss.write_index(data, full_path)
        else:
            raise ValueError(f"Unknown serializer: {serializer}")
    except (IOError, PermissionError) as e:
        print(f"❌ Error: Could not write to {full_path}. Reason: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while saving {filename}: {e}")

def load_data(filename, serializer):
    # loads data from specified serializer
    
    full_path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(full_path):
        print(f"Cache file not found: {full_path}")
        return None

    try:
        if serializer == 'pickle':
            with open(full_path, "rb") as f:
                return pickle.load(f)
        elif serializer == 'faiss':
            return faiss.read_index(full_path)
        else:
            raise ValueError(f"Unknown serializer: {serializer}")
    except (pickle.UnpicklingError, faiss.FaissException) as e:
        print(f"⚠️ Warning: Could not load corrupted cache file '{filename}'. Reason: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading {filename}: {e}")
        return None

def cache_exists(filenames):
    # check if cached files exist
    
    return all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in filenames)