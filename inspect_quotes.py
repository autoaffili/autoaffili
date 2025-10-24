import re
from pathlib import Path
text = Path('backend/server.py').read_text()
for m in re.finditer(r"strip\(([^)]*)\)", text):
    s = m.group(1)
    if '"' in s and "'" in s:
        print(repr(s))
