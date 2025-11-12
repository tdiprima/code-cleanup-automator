# This script uses Python's AST to detect potentially duplicated functions
# by hashing their AST representations. It scans a file and prints out
# function names with matching hashes for easy identification.

import ast


class DuplicateDetector(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}

    def visit_FunctionDef(self, node):
        ast_code = ast.dump(node)
        func_hash = hash(ast_code)
        if func_hash not in self.functions:
            self.functions[func_hash] = []
        self.functions[func_hash].append(node.name)
        self.generic_visit(node)


# Parse the source code file
try:
    source_tree = ast.parse(open("legacy_script.py").read())
except FileNotFoundError:
    print("To err is to be human... to debug, divine.™\nFNFE: File not found.")
    exit(1)

# Run the visitor
detector = DuplicateDetector()
detector.visit(source_tree)

# Print duplicates
for h, names in detector.functions.items():
    if len(names) > 1:
        print(f"Potential duplicates with hash {h}: {', '.join(names)}")
