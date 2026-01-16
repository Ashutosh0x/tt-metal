import json
import subprocess

def get_recent_bounties():
    # Sort by created-desc to find new ones
    cmd = ["gh", "issue", "list", "--repo", "tenstorrent/tt-metal", "--label", "bounty", "-S", "sort:created-desc", "--limit", "100", "--json", "number,title,assignees,comments,createdAt"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)

def main():
    data = get_recent_bounties()
    # Filter for unassigned
    untouched = [i for i in data if not i['assignees']]
    
    # Sort by comment count
    untouched.sort(key=lambda x: len(x['comments']))
    
    print(f"{'Number':<8} | {'Comments':<8} | {'Created':<12} | {'Title'}")
    print("-" * 100)
    for i in untouched[:15]:
        date = i['createdAt'][:10]
        print(f"{i['number']:<8} | {len(i['comments']):<8} | {date:<12} | {i['title']}")

if __name__ == '__main__':
    main()
