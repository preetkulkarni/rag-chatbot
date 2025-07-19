import os
import pickle
import faiss
from typing import List, Any
from . import config

def get_cache_path(file_path: str) -> str:
    # create safe directory path based on input file

    base_name = os.path.basename(file_path)
    file_name_safe = "".join(c for c in base_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
    cache_dir_name = f"{file_name_safe}_cache"
    return os.path.join(config.CACHED_DIR, cache_dir_name)

def save_data(cache_path: str, file_name: str, data: Any, serializer: str = 'pickle'):
    # saves data using specified serializer in a specified dir
    
    os.makedirs(cache_path, exist_ok=True)
    full_path = os.path.join(cache_path, file_name)

    print(f"💾 Saving {file_name} to {full_path}...")
    try:
        if serializer == 'pickle':
            with open(full_path, 'wb') as f:
                pickle.dump(data, f)
        elif serializer == 'faiss':
            faiss.write_index(data, full_path)
        else:
            raise ValueError(f"Unknown serializer: {serializer}")
    except (IOError, PermissionError) as e:
        print(f"❌ Error: Could not write to {full_path}. Reason: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while saving {file_name}: {e}")
    print("✅ Save complete.\n")

def load_data(cache_path: str, file_name: str, serializer: str = 'pickle') -> Any:
    # loads data from specified serializer within a given dir
    
    full_path = os.path.join(cache_path, file_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Cache file not found: {full_path}")

    print(f"\n📂 Loading {file_name} from {full_path}...")
    try:
        if serializer == 'pickle':
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        elif serializer == 'faiss':
            return faiss.read_index(full_path)
        else:
            raise ValueError(f"Unknown serializer: {serializer}")
    except (pickle.UnpicklingError, faiss.FaissException) as e:
        print(f"⚠️ Warning: Could not load corrupted cache file '{file_name}'. Reason: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading {file_name}: {e}")
        return None

def cache_exists(cache_path: str, files: List[str]) -> bool:
    # check if cached files exist
    
    for file_name in files:
        if not os.path.exists(os.path.join(cache_path, file_name)):
            return False
    return True