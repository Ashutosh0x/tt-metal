import requests
import json
import os

def fetch_comments(repo, issue_number, is_pr=False):
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    # Generic issue comments
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    comments = response.json()
    
    if is_pr:
        # PR review comments
        url_review = f"https://api.github.com/repos/{repo}/pulls/{issue_number}/comments"
        response_review = requests.get(url_review, headers=headers)
        review_comments = response_review.json()
        comments.extend(review_comments)
        
    return comments

if __name__ == "__main__":
    repo = "tenstorrent/tt-metal"
    # Depth Anything V2 PR
    pr_comments = fetch_comments(repo, 35565, is_pr=True)
    with open("depth_anything_v2_comments.json", "w", encoding="utf-8") as f:
        json.dump(pr_comments, f, indent=4)
    
    # DeepSeek issue comments
    issue_comments = fetch_comments(repo, 35509)
    with open("deepseek_v3_comments.json", "w", encoding="utf-8") as f:
        json.dump(issue_comments, f, indent=4)
    
    print("Comments fetched successfully.")
