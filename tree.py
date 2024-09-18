import os

def generate_tree(startpath, exclude_dirs=['venv', '__pycache__', '.git']):
    tree = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        tree.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            tree.append(f"{'│   ' * level}├── {f}")
    return '\n'.join(tree)

# Use the current directory as the starting point
print(generate_tree('.'))