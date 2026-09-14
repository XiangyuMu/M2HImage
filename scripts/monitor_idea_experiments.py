#!/usr/bin/env python3
"""Monitor A/B/C runs; intended for a 2-hour cron/heartbeat invocation."""
import argparse,json,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args(); out=[]
 for name in a.runs:
  d=Path(a.root)/name; ck=list(d.rglob('checkpoints/*/READY')) if d.exists() else []
  logs=list(d.rglob('*.log')) if d.exists() else []
  latest=max([x.stat().st_mtime for x in logs],default=0); out.append({'run':name,'exists':d.exists(),'ready_checkpoints':len(ck),'log_count':len(logs),'latest_log_age_s':None if not latest else time.time()-latest,'status':'healthy' if d.exists() and logs else 'missing_or_no_logs'})
 Path(a.out).write_text(json.dumps({'created_at':time.time(),'runs':out},indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
