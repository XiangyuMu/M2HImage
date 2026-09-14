import csv,json,hashlib,shutil,os
from pathlib import Path
ROOT=Path('/data/muxiangyu/datasets/M2HImage/M2H_Final_v2')
OUT=Path('/data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1')
REP=OUT/'dataset_cleaning_report'
ART=Path('/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/full_duplicates')
cands=json.load(open(ART/'dhash_equal_candidates.json'))
edges=list(csv.DictReader(open(REP/'dhash_candidate_audit.csv')))
# index edges by unordered ids
emap={tuple(sorted((r['left_id'],r['right_id']))):r for r in edges}
rows=[]
for i,c in enumerate(cands):
    mem=c.get('members',[]); ids=sorted({m['id'] for m in mem}); key='|'.join((c.get('role',''),c.get('dhash',''),','.join(ids))); gid=f'cand_{i:04d}_{hashlib.sha1(key.encode()).hexdigest()[:10]}'
    es=[emap[k] for a in ids for b in ids if a<b and (k:=tuple(sorted((a,b)))) in emap]
    statuses=[e['status'] for e in es]; splits=sorted({m.get('split','') for m in mem}); sha_equal=any(e['sha256_equal']=='true' for e in es)
    if sha_equal or 'confirmed_duplicate' in statuses: status='confirmed_duplicate'
    elif any(s=='near_duplicate_risk' for s in statuses) or len(splits)>1: status='near_duplicate_risk'
    else: status='visual_similarity_only'
    rows.append({'candidate_group_id':gid,'role':c.get('role',''),'dhash':c.get('dhash',''),'member_ids':' '.join(ids),'member_count':len(ids),'splits':' '.join(splits),'edge_count':len(es),'edge_statuses':' '.join(sorted(set(statuses))),'sha256_equal':sha_equal,'status':status})
with open(REP/'dhash_candidate_groups.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with open(REP/'dhash_components.jsonl','w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary={'candidate_group_count':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in ['confirmed_duplicate','near_duplicate_risk','visual_similarity_only']},'edge_count':len(edges)}
json.dump(summary,open(REP/'dhash_audit_summary.json','w'),indent=2)
# high-risk IDs quarantine links and sensitivity split
risk=list(csv.DictReader(open(REP/'dhash_cross_split_risk.csv'))); ids=set()
for r in risk: ids|={r['left_id'],r['right_id']}
exact=set(x.strip() for x in open(OUT/'splits_person_disjoint/quarantine.txt') if x.strip())
ids-=exact
qroot=OUT/'quarantine/high_risk_dhash'; qroot.mkdir(parents=True,exist_ok=True)
# copy/hardlink all files associated with id from source tree preserving relative paths where discoverable
for id in sorted(ids):
 d=qroot/id; d.mkdir(exist_ok=True)
 for p in ROOT.rglob(f'*{id}*'):
  if p.is_file():
   rel=p.relative_to(ROOT); dest=d/rel; dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    try: os.link(p,dest)
    except OSError: shutil.copy2(p,dest)
# sensitivity split remove high risk ids
for split in ['train','val','test']:
 src=OUT/'splits_person_disjoint'/f'{split}.txt'; vals=[x.strip() for x in open(src) if x.strip() and x.strip() not in ids]
 open(OUT/'splits_person_disjoint'/f'{split}_sensitivity.txt','w').write('\n'.join(vals)+'\n')
open(OUT/'splits_person_disjoint/quarantine_high_risk.txt','w').write('\n'.join(sorted(ids))+'\n')
summary.update({'high_risk_cross_split_edge_count':len(risk),'high_risk_unique_ids':len(ids),'exact_quarantine_ids':len(exact)})
json.dump(summary,open(REP/'dhash_audit_summary.json','w'),indent=2)
