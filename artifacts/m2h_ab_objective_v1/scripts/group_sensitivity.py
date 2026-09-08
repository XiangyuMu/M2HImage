"""Candidate-component sensitivity; never writes a final split or true IDs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


class UnionFind:
    def __init__(self,n):
        self.parent=list(range(n))
        self.size=[1]*n

    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]]
            i=self.parent[i]
        return i

    def union(self,i,j):
        i,j=self.find(i),self.find(j)
        if i==j:
            return
        if self.size[i]<self.size[j]:
            i,j=j,i
        self.parent[j]=i
        self.size[i]+=self.size[j]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--neighbors',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.neighbors,allow_pickle=False) as z:
        ids=z['ids'].tolist(); v=z['cosine']; j=z['neighbor_index']
    assert np.isfinite(v).all() and j.min()>=0 and j.max()<len(ids)
    with args.manifest.open(newline='') as h:
        meta={r['id']:r for r in csv.DictReader(h)}
    parents=defaultdict(list)
    for i,sid in enumerate(ids):
        r=meta[sid]; key=r['source_key']; stem,sep,suffix=key.rpartition('_')
        group=r['source_dataset']+':'+(stem if sep and suffix.isdigit() else key)
        parents[group].append(i)
    report={'warning':'Components from truncated top-k identity candidates; counts upper-bound full threshold-graph component count, not biological identities.',
            'formal_split_ready':False,'thresholds':{}}
    for threshold in (.6,.7,.8,.9):
        uf=UnionFind(len(ids))
        a,b=np.where(v>=threshold)
        for left,col in zip(a.tolist(),b.tolist()):
            uf.union(left,int(j[left,col]))
        groups=defaultdict(list)
        for i in range(len(ids)):
            groups[uf.find(i)].append(i)
        result={'identity_candidate_components':len(groups),'largest_identity_candidate_component':max(map(len,groups.values()))}
        for indices in parents.values():
            for i in indices[1:]:
                uf.union(indices[0],i)
        groups=defaultdict(list)
        for i in range(len(ids)):
            groups[uf.find(i)].append(i)
        result.update(source_union_components=len(groups),largest_source_union_component=max(map(len,groups.values())),
                      cross_split_components=sum(len({meta[ids[i]]['split'] for i in members})>1 for members in groups.values()),
                      kth_neighbor_saturated_rows=int((v[:,-1]>=threshold).sum()))
        report['thresholds'][str(threshold)]=result
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
