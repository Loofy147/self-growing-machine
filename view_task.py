import json, sys
def print_grid(grid):
    for row in grid: print("".join(str(c) for c in row))
task_id = int(sys.argv[1])
with open(f"data/task{task_id:03d}.json") as f: data = json.load(f)
print(f"Task {task_id}:")
for i, p in enumerate(data['train']):
    print(f"Pair {i} Input:")
    print_grid(p['input'])
    print(f"Pair {i} Output:")
    print_grid(p['output'])
