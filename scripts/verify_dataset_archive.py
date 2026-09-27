"""Verify a separately obtained external archive; never download or extract it."""
import argparse,hashlib
from pathlib import Path
EXPECTED='4eed0b61b028f91b0d2568b04baabcca6a4a3dbb450cd3613e2fc01f3fd20143'
SIZE=13494522554
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    if a.archive.stat().st_size!=SIZE:raise SystemExit('FAIL: archive size differs')
    h=hashlib.sha256()
    with a.archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=EXPECTED:raise SystemExit('FAIL: archive SHA-256 differs')
    print('PASS: matches recorded local archive identity; not independent authentication or ZIP integrity verification')
if __name__=='__main__':main()
