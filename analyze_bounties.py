import json
from datetime import datetime

def analyze_bounties(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        with open(file_path, 'r', encoding='utf-16') as f:
            data = json.load(f)
    
    untouched = []
    for issue in data:
        # Skip if assigned
        if issue['assignees']:
            continue
            
        num_comments = len(issue['comments'])
        created_at = datetime.strptime(issue['createdAt'], '%Y-%m-%dT%H:%M:%SZ')
        
        untouched.append({
            'number': issue['number'],
            'title': issue['title'],
            'comments': num_comments,
            'created': created_at,
            'url': f"https://github.com/tenstorrent/tt-metal/issues/{issue['number']}"
        })
    
    # Sort by comment count (ascending) then by creation date (descending)
    untouched.sort(key=lambda x: (x['comments'], -x['created'].timestamp()))
    
    print(f"{'#':<6} | {'Cmds':<4} | {'Created':<10} | {'Title'}")
    print("-" * 80)
    for i in untouched[:20]:
        print(f"{i['number']:<6} | {i['comments']:<4} | {i['created'].strftime('%Y-%m-%d'):<10} | {i['title']}")

if __name__ == "__main__":
    analyze_bounties('all_bounties.json')
