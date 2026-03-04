import os
import re
import shutil

# Target src directory
src_dir = "src"
nested_dir = os.path.join(src_dir, "document_intelligence_refinery")

if not os.path.exists(nested_dir):
    print("Nested directory not found. Already flattened?")
    exit(0)

# Move all items from nested to src
for item in os.listdir(nested_dir):
    if item == "__pycache__":
        continue
        
    s = os.path.join(nested_dir, item)
    d = os.path.join(src_dir, item)
    
    if os.path.exists(d):
        if os.path.isdir(d):
            # Merge contents
            for sub_item in os.listdir(s):
                if sub_item == "__pycache__": continue
                sub_s = os.path.join(s, sub_item)
                sub_d = os.path.join(d, sub_item)
                shutil.move(sub_s, sub_d)
                print(f"Merged {sub_s} -> {sub_d}")
            shutil.rmtree(s)
        else:
            print(f"Conflict: {d} already exists!")
    else:
        shutil.move(s, d)
        print(f"Moved {s} -> {d}")

# Clean up nested dir
try:
    shutil.rmtree(nested_dir)
except Exception as e:
    print(f"Could not remove {nested_dir}: {e}")

# Regex to find and replace module imports
# X -> X
import_pattern = r'document_intelligence_refinery\.'

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
        
    orig_content = content
    
    # Replace absolute imports
    content = re.sub(r'from\s+document_intelligence_refinery\.', 'from ', content)
    content = re.sub(r'import\s+document_intelligence_refinery\.', 'import ', content)
    content = re.sub(r'\bdocument_intelligence_refinery\.', '', content)

    if content != orig_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Updated imports in {filepath}")

# Walk through the whole project to update imports
for root, dirs, files in os.walk("."):
    if ".git" in root or ".venv" in root or "__pycache__" in root or ".refinery" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            process_file(os.path.join(root, file))

