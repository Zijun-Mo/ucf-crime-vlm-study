"""Scan the public snapshot for common credentials and personal resource metadata.

Reports filenames and categories only, never matching secrets. This is a bounded
publication check, not a proof that every possible form of private data is absent.
"""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'personal_home_path': r'/home/[A-Za-z0-9_.-]+/',
    'messaging_identity': r'wxid_[A-Za-z0-9]+|xwechat_files',
    'credential': r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ya29\.[A-Za-z0-9_.-]{20,}|AIza[A-Za-z0-9_-]{25,}|sk-[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'jwt': r'eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    'private_drive_link': r'https://drive\.google\.com/(?:file|drive)/',
    'hosting_identity': r'appg(?:prj|dep|ver)_[A-Za-z0-9]{20,}',
    'signed_secret_query': r'[?&](?:access_token|refresh_token|X-Amz-Credential)=',
    'email': r'\b[\w.+-]+@(?!users\.noreply\.github\.com)[\w.-]+\.[A-Za-z]{2,}\b',
}

def main():
    hits=[];count=0
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(part in {'.git','.venv','.pytest_cache','__pycache__','downloads'} for part in p.relative_to(ROOT).parts):continue
        if p.resolve()==Path(__file__).resolve():continue
        if p.is_symlink():raise RuntimeError('Unexpected symlink: '+str(p.relative_to(ROOT)))
        try:text=p.read_text()
        except UnicodeDecodeError:continue
        count+=1
        for name,pattern in PATTERNS.items():
            if re.search(pattern,text):hits.append(dict(file=str(p.relative_to(ROOT)),category=name))
    print(json.dumps(dict(text_files_scanned=count,findings=hits),ensure_ascii=False,indent=2))
    return bool(hits)

if __name__=='__main__':sys.exit(main())
