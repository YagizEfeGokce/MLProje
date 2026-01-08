import json
import os

def run_notebook_code(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Verifying {notebook_path}...")
    
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    
    full_code = ""
    for i, cell in enumerate(code_cells):
        source = "".join(cell['source'])
        # Skip magic commands
        source_lines = [line for line in source.split('\n') if not line.strip().startswith('%')]
        clean_source = "\n".join(source_lines)
        full_code += clean_source + "\n\n"
        
    try:
        exec(full_code, {'__name__': '__main__'})
        print("\nSUCCESS: Notebook code executed without errors.")
    except Exception as e:
        print(f"\nFAILURE: Notebook execution failed.\nError: {e}")

if __name__ == "__main__":
    run_notebook_code('ML_Project_Notebook.ipynb')
