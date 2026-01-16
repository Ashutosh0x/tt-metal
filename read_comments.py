import json

def print_recent_comments(filename, count=10):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
        
        # Sort by date if possible
        comments.sort(key=lambda x: x.get("created_at", ""))
        
        print(f"--- Recent 10 comments from {filename} ---")
        for comment in comments[-count:]:
            author = comment.get("user", {}).get("login", "Unknown")
            body = comment.get("body", "")
            created_at = comment.get("created_at", "")
            path = comment.get("path", "N/A")
            line = comment.get("line", "N/A")
            
            print(f"Author: {author} ({created_at})")
            if path != "N/A":
                print(f"Path: {path} (Line: {line})")
            print(f"Comment: {body}")
            print("=" * 40)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    print_recent_comments("depth_anything_v2_comments.json")
    print_recent_comments("deepseek_v3_comments.json")
