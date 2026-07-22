import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
# Find all __init__ methods
pattern = r'def __init__.*?(?=\n    def|\nclass|\Z)'
matches = re.findall(pattern, content, re.DOTALL)

for i, match in enumerate(matches, 1):
    if 'return ' in match and 'return None' not in match:
        print(f"⚠️ Found return in __init__ #{i}")
        print(match[:200])  # Show first 200 chars
        print("-" * 50)