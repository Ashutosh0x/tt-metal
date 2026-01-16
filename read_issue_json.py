import json
import sys

with open('deepseek_bug_details.json', 'r', encoding='utf-16') as f:
    data = json.load(f)

print("--- ISSUE BODY ---")
print(data.get('body', ''))
print("\n--- COMMENTS ---")
for i, comment in enumerate(data.get('comments', [])):
    print(f"\n[Comment {i+1}] Author: {comment.get('author', {}).get('login', 'unknown')}")
    print(comment.get('body', ''))
