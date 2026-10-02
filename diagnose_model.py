import sys
import builtins
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

def get_joblib_references(filepath):
    print(f"\n==========================================")
    print(f"TRACKING MODULE IMPORTS FOR: {filepath.name}")
    print(f"==========================================")
    
    referenced_modules = set()
    original_import = builtins.__import__
    
    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        referenced_modules.add(name)
        return original_import(name, globals, locals, fromlist, level)
    
    builtins.__import__ = tracking_import
    obj = None
    try:
        obj = joblib.load(filepath)
        print(f"SUCCESS: Loaded {filepath.name}")
        print(f"Root object type: {type(obj).__module__}.{type(obj).__qualname__}")
    except Exception as e:
        print(f"FAILURE loading {filepath.name}: {e}")
    finally:
        builtins.__import__ = original_import

    print("\nModules Imported During Unpickling:")
    for mod in sorted(referenced_modules):
        print(f"  - {mod}")
        
    return obj, referenced_modules

def check_local_env():
    print(f"\n==========================================")
    print(f"LOCAL PYTHON ENVIRONMENT DETAILS")
    print(f"==========================================")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Full Sys Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    packages = [
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("joblib", "joblib"),
        ("pandas", "pandas"),
        ("nltk", "nltk"),
        ("pymongo", "pymongo"),
        ("python-dotenv", "dotenv"),
        ("streamlit", "streamlit"),
    ]
    
    for label, mod_name in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  - {label} ({mod_name}): {ver}")
        except ImportError:
            print(f"  - {label} ({mod_name}): NOT INSTALLED")

def inspect_object_details(obj, name):
    print(f"\n------------------------------------------")
    print(f"OBJECT DETAILS: {name}")
    print(f"------------------------------------------")
    print(f"Class: {type(obj).__module__}.{type(obj).__qualname__}")
    if hasattr(obj, "__dict__"):
        print("Attributes:")
        for k, v in obj.__dict__.items():
            print(f"  - {k}: {type(v).__module__}.{type(v).__qualname__}")

if __name__ == "__main__":
    check_local_env()
    model, mod_m = get_joblib_references(MODEL_PATH)
    tfidf, mod_t = get_joblib_references(TFIDF_PATH)
    if model:
        inspect_object_details(model, "model.pkl")
    if tfidf:
        inspect_object_details(tfidf, "tfidf.pkl")
