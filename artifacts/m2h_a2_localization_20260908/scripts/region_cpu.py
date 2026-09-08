"""Same frozen regional metrics on CPU with bounded threads and durable output."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU runner requires CUDA_VISIBLE_DEVICES empty')
    started=time.monotonic()
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    import region_diagnostics as region
    original=region.RegionMetricExtractor
    class CPUExtractor(original):
        def __init__(self,repo,device):
            if device!='cpu': raise ValueError('CPU only')
            super().__init__(repo,device)
    region.RegionMetricExtractor=CPUExtractor
    region.main()
    out=Path(sys.argv[sys.argv.index('--out')+1])
    p=out/'resources.json'; result=json.loads(p.read_text())
    result.update(gpu_metrics_launched=False,device='cpu',torch_threads=4,
                  cpu_wall_seconds_including_imports=time.monotonic()-started,gpu_hours=0)
    p.write_text(json.dumps(result,indent=2)+'\n')
    for p in out.iterdir():
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    fd=os.open(out,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


if __name__=='__main__':main()
