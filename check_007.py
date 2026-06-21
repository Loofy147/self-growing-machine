import json
with open("data/task007.json") as f: data = json.load(f)
print(f"Input size: {len(data['train'][0]['input'])}x{len(data['train'][0]['input'][0])}")
print(f"Output size: {len(data['train'][0]['output'])}x{len(data['train'][0]['output'][0])}")
