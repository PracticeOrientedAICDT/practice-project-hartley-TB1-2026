"""Portable run folders. Codes identify local/shared data, not cloud storage."""
import json,re,secrets
from pathlib import Path
from datetime import datetime,timezone

def create_run(root='runs',code=None,**metadata):
    code=code or 'pilot-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',code):
        raise ValueError('Use a short code containing letters, numbers, hyphens or underscores')
    path=Path(root)/code
    path.mkdir(parents=True,exist_ok=False)
    save_metadata(path,dict(code=code,created_utc=datetime.now(timezone.utc).isoformat(),status='running',**metadata))
    return path

def save_metadata(path,metadata):
    (Path(path)/'run.json').write_text(json.dumps(metadata,indent=2))

def finish_run(path,**updates):
    path=Path(path);meta=json.loads((path/'run.json').read_text());meta.update(status='complete',**updates)
    save_metadata(path,meta)

def list_runs(root):
    return sorted([p for p in Path(root).glob('*') if (p/'run.json').is_file()],reverse=True)
