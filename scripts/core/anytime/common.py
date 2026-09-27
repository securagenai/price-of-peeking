import hashlib
import json
import os
import tempfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
