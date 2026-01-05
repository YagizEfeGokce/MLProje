import importlib.util
import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"MISSING: {name}")
    else:
        print(f"FOUND: {name}")

check_package("pypdf")
check_package("python-docx", "docx")
