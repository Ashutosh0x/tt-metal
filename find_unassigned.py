import json

def analyze_all_bounties(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        with open(file_path, 'r', encoding='utf-16') as f:
            data = json.load(f)
    
    unassigned = []
    for issue in data:
        if not issue['assignees']:
            unassigned.append({
                'number': issue['number'],
                'title': issue['title'],
                'comments': len(issue['comments']),
            })
    
    # Sort by comment count (ascending)
    unassigned.sort(key=lambda x: x['comments'])
    
    print(f"Found {len(unassigned)} unassigned bounty issues:\n")
    print(f"{'#':<6} | {'Cmts':<4} | {'Title'}")
    print("-" * 90)
    for i in unassigned:
        print(f"{i['number']:<6} | {i['comments']:<4} | {i['title'][:70]}")

if __name__ == "__main__":
    analyze_all_bounties('all_open_bounties.json')
