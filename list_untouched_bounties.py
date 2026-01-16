import json
import subprocess

def get_bounties():
    cmd = ["gh", "issue", "list", "--repo", "tenstorrent/tt-metal", "--label", "bounty", "--state", "open", "--limit", "100", "--json", "number,title,assignees,comments"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return []
    return json.loads(result.stdout)

def main():
    data = get_bounties()
    untouched = [i for i in data if not i['assignees']]
    
    # Sort by comment count
    untouched.sort(key=lambda x: len(x['comments']))
    
    print(f"{'Number':<8} | {'Comments':<8} | {'Title'}")
    print("-" * 100)
    for i in untouched:
        print(f"{i['number']:<8} | {len(i['comments']):<8} | {i['title']}")

if __name__ == '__main__':
    main()
