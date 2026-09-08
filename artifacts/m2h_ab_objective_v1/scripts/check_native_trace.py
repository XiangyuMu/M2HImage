"""Audit successful absolute dataset opens in an existing strace log.

Only observed execution is checked; no proof of future behavior is claimed.
"""
import argparse
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    audit=json.loads(a.audit.read_text())
    allowed=set(audit['allowed_dataset_paths'])
    # The checkpoint loader opens its declared directory to enumerate weights.
    # Permit the exact declared directory, not its arbitrary descendants.
    if 'checkpoint' in audit:
        allowed.add(str(Path(audit['checkpoint']).resolve()))
    # This old-val membership check intentionally occurs before the read hook.
    allowed.add(str(a.root/'splits/val.txt'))
    observed=[]
    unresolved=[]
    for line in a.trace.read_text().splitlines():
        # Unfinished system calls are retained as uncertainty if they contain
        # a dataset pathname; their resumed result cannot be guessed.
        if str(a.root) not in line:
            continue
        match=re.search(r'"([^"\n]*'+re.escape(str(a.root))+r'[^"\n]*)"',line)
        if not match:
            unresolved.append(line)
            continue
        path=match.group(1)
        if '<unfinished ...>' in line:
            unresolved.append(line)
        elif re.search(r'= \d+\s*$',line):
            observed.append(path)
    unexpected=sorted(set(observed)-allowed)
    report={'observed_successful_dataset_opens':len(observed),
            'unique_dataset_paths':sorted(set(observed)),
            'unexpected_dataset_paths':unexpected,'unresolved_dataset_lines':unresolved,
            'trace_scope':'strace -f open/openat, one old-val inference, absolute dataset pathnames',
            'status':'observed_trace_pass' if not unexpected and not unresolved else 'requires_review',
            'formal_gate_complete':False,
            'remaining':'I_j cache lineage still code-traced; no full regenerated identity input cache or clean split.'}
    a.out.open('x').write(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
