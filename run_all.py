import os
import sys

# Priority 1: High-Efficiency Primitive Search
print("Starting Primitive Search...")
os.system("python3 search_solvers.py")

# Priority 2: Tiny Conv Search (Catch remaining local tasks)
print("Starting TinyConv Search...")
os.system("python3 tiny_conv_solver.py")

# Priority 3: Composite Conv Search (Catch complex local tasks)
print("Starting CompositeConv Search...")
os.system("python3 composite_conv_solver.py")

print("Integrated search complete.")
