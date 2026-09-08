from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from qual_eval_common import (
    MAIN_RUNS,
    build_panel_context,
    read_json,
    resolve_path,
    write_csv,
    write_json,
)


QUESTION_FIELDS = {
    'q1': ['garment_consistency'],
    'q2': ['identity_match'],
    'q3': ['garment_fidelity', 'realism'],
}
FIELD_LABELS = {
    'garment_consistency': '这四张图的服装一致程度（花色、版型、位置）',
    'identity_match': '生成人物与参考身份是同一个人的程度',
    'garment_fidelity': '生成图服装与源人台服装一致的程度',
    'realism': '生成人物的整体真实感',
}
PROMPTS = {
    'q1': '请只判断四张图之间的服装一致性，不评价人物身份。',
    'q2': '请判断生成图中的人物是否与参考身份为同一个人。',
    'q3': '请分别判断服装对源人台的保真度，以及生成人物的整体真实感。',
}


def question_id(index: int) -> str:
    return f'q{int(index):05d}'


def build_normal_questions(context: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(context['qcfg']['seed_for_panels'])
    identity_positions = [int(value) for value in context['qcfg']['human_identity_positions']]
    questions: list[dict[str, Any]] = []
    next_index = 1
    for row in manifest['rows']:
        mid = str(row['mid'])
        jids = [str(value) for value in row['identity_ids']]
        source = context['sources'][mid]
        for run_name in MAIN_RUNS:
            questions.append({
                'question_id': question_id(next_index),
                'question_type': 'q1',
                'run': run_name,
                'mid': mid,
                'jid': None,
                'seed': seed,
                'assets': [str(source['generated'][(run_name, jid)]) for jid in jids],
                'asset_labels': ['Image A', 'Image B', 'Image C', 'Image D'],
                'score_fields': QUESTION_FIELDS['q1'],
                'attention': False,
            })
            next_index += 1
        chosen = [jids[index] for index in identity_positions]
        for jid in chosen:
            for run_name in MAIN_RUNS:
                questions.append({
                    'question_id': question_id(next_index),
                    'question_type': 'q2',
                    'run': run_name,
                    'mid': mid,
                    'jid': jid,
                    'seed': seed,
                    'assets': [str(source['faces'][jid]), str(source['generated'][(run_name, jid)])],
                    'asset_labels': ['Reference identity', 'Generated image'],
                    'score_fields': QUESTION_FIELDS['q2'],
                    'attention': False,
                })
                next_index += 1
            for run_name in MAIN_RUNS:
                questions.append({
                    'question_id': question_id(next_index),
                    'question_type': 'q3',
                    'run': run_name,
                    'mid': mid,
                    'jid': jid,
                    'seed': seed,
                    'assets': [str(source['mannequin']), str(source['generated'][(run_name, jid)])],
                    'asset_labels': ['Source mannequin', 'Generated image'],
                    'score_fields': QUESTION_FIELDS['q3'],
                    'attention': False,
                })
                next_index += 1
    counts = Counter(question['question_type'] for question in questions)
    expected = {'q1': 48, 'q2': 96, 'q3': 96}
    if counts != expected:
        raise RuntimeError(f'normal question counts differ from preregistration: {counts} != {expected}')
    return questions


def attention_templates(context: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest['rows']
    first, second, last = rows[0], rows[1], rows[-1]
    first_source = context['sources'][first['mid']]
    second_source = context['sources'][second['mid']]
    last_source = context['sources'][last['mid']]
    consistent_one = str(first_source['generated'][('b2cont', first['identity_ids'][0])])
    consistent_two = str(second_source['generated'][('a4', second['identity_ids'][1])])
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row['garment_type']].append(row)
    garment_types = sorted(by_type)
    mismatch_first = [str(context['sources'][by_type[name][0]['mid']]['mannequin']) for name in garment_types]
    mismatch_last = [str(context['sources'][by_type[name][-1]['mid']]['mannequin']) for name in garment_types]
    face_one = str(first_source['faces'][first['identity_ids'][0]])
    face_two = str(last_source['faces'][last['identity_ids'][-1]])
    return [
        {
            'question_type': 'q1', 'assets': [consistent_one] * 4,
            'asset_labels': ['Image A', 'Image B', 'Image C', 'Image D'],
            'score_fields': QUESTION_FIELDS['q1'],
            'expected': {'field': 'garment_consistency', 'operator': '>=', 'threshold': 4},
        },
        {
            'question_type': 'q1', 'assets': mismatch_first,
            'asset_labels': ['Image A', 'Image B', 'Image C', 'Image D'],
            'score_fields': QUESTION_FIELDS['q1'],
            'expected': {'field': 'garment_consistency', 'operator': '<=', 'threshold': 2},
        },
        {
            'question_type': 'q2', 'assets': [face_one, face_one],
            'asset_labels': ['Reference identity', 'Comparison image'],
            'score_fields': QUESTION_FIELDS['q2'],
            'expected': {'field': 'identity_match', 'operator': '>=', 'threshold': 4},
        },
        {
            'question_type': 'q1', 'assets': [consistent_two] * 4,
            'asset_labels': ['Image A', 'Image B', 'Image C', 'Image D'],
            'score_fields': QUESTION_FIELDS['q1'],
            'expected': {'field': 'garment_consistency', 'operator': '>=', 'threshold': 4},
        },
        {
            'question_type': 'q1', 'assets': mismatch_last,
            'asset_labels': ['Image A', 'Image B', 'Image C', 'Image D'],
            'score_fields': QUESTION_FIELDS['q1'],
            'expected': {'field': 'garment_consistency', 'operator': '<=', 'threshold': 2},
        },
        {
            'question_type': 'q2', 'assets': [face_two, face_two],
            'asset_labels': ['Reference identity', 'Comparison image'],
            'score_fields': QUESTION_FIELDS['q2'],
            'expected': {'field': 'identity_match', 'operator': '>=', 'threshold': 4},
        },
    ]


def assign_questions(
    questions: list[dict[str, Any]],
    rater_count: int,
    ratings_per_question: int,
    seed: int,
) -> dict[str, list[str]]:
    if ratings_per_question >= rater_count:
        raise ValueError('ratings_per_question must be smaller than rater_count')
    if (len(questions) * ratings_per_question) % rater_count:
        raise RuntimeError('question/rater counts cannot be balanced exactly')
    rng = random.Random(seed)
    shuffled = [question['question_id'] for question in questions]
    rng.shuffle(shuffled)
    offsets = [0, 3, 5, 7, 2, 6, 1, 4][:ratings_per_question]
    if len(set(offsets)) != ratings_per_question:
        raise RuntimeError('assignment offsets are not unique')
    assignments = {f'rater_{index + 1:02d}': [] for index in range(rater_count)}
    for position, qid in enumerate(shuffled):
        for offset in offsets:
            rater = f'rater_{(position + offset) % rater_count + 1:02d}'
            assignments[rater].append(qid)
    expected = len(questions) * ratings_per_question // rater_count
    counts = {rater: len(values) for rater, values in assignments.items()}
    if set(counts.values()) != {expected}:
        raise RuntimeError(f'unbalanced rater assignment: {counts}')
    return assignments


def spread_questions(qids: list[str], questions: dict[str, dict[str, Any]], seed: int) -> list[str]:
    rng = random.Random(seed)
    by_mid: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        by_mid[str(questions[qid]['mid'])].append(qid)
    for values in by_mid.values():
        rng.shuffle(values)
    tie_break = {mid: rng.random() for mid in by_mid}
    output: list[str] = []
    while any(by_mid.values()):
        recent_two = {str(questions[qid]['mid']) for qid in output[-2:]}
        candidates = [mid for mid, values in by_mid.items() if values and mid not in recent_two]
        if not candidates:
            previous = str(questions[output[-1]]['mid']) if output else None
            candidates = [mid for mid, values in by_mid.items() if values and mid != previous]
        if not candidates:
            raise RuntimeError('could not order rater questions without adjacent same-mid items')
        chosen = max(candidates, key=lambda mid: (len(by_mid[mid]), tie_break[mid], mid))
        output.append(by_mid[chosen].pop())
    return output


def insert_attention(normal: list[str], attention: list[str]) -> list[str]:
    output = list(normal)
    for index, qid in enumerate(attention):
        position = round((index + 1) * (len(normal) + len(attention)) / (len(attention) + 1))
        output.insert(min(position, len(output)), qid)
    return output


class AssetEncoder:
    def __init__(self, max_size: tuple[int, int], quality: int):
        self.max_size = max_size
        self.quality = int(quality)
        self.cache: dict[str, str] = {}

    def encode(self, path: str | Path) -> str:
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.quality, optimize=True)
        value = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        self.cache[key] = value
        return value


def sanitized_payload(
    rater_id: str,
    ordered_qids: list[str],
    questions: dict[str, dict[str, Any]],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    encoder = AssetEncoder(
        (int(eval_cfg['html_image_max_width']), int(eval_cfg['html_image_max_height'])),
        int(eval_cfg['html_jpeg_quality']),
    )
    asset_ids: dict[str, str] = {}
    assets: dict[str, str] = {}

    def asset_id(path: str) -> str:
        if path not in asset_ids:
            opaque = f'asset_{len(asset_ids) + 1:04d}'
            asset_ids[path] = opaque
            assets[opaque] = encoder.encode(path)
        return asset_ids[path]

    tasks = []
    for qid in ordered_qids:
        question = questions[qid]
        tasks.append({
            'question_id': qid,
            'question_type': question['question_type'],
            'prompt': PROMPTS[question['question_type']],
            'assets': [asset_id(path) for path in question['assets']],
            'asset_labels': question['asset_labels'],
            'scales': [
                {'field': field, 'label': FIELD_LABELS[field]}
                for field in question['score_fields']
            ],
        })
    return {
        'rater_id': rater_id,
        'scale_min': int(eval_cfg['scale_min']),
        'scale_max': int(eval_cfg['scale_max']),
        'tasks': tasks,
        'assets': assets,
    }


HTML_TEMPLATE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M2HImage blind evaluation</title>
<style>
:root { color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f5f7; color: #17191d; }
header { position: sticky; top: 0; z-index: 2; display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: center; padding: 12px 20px; background: #fff; border-bottom: 1px solid #c7ccd3; }
h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
#progress { color: #4c5561; font-size: 14px; }
main { max-width: 1280px; margin: 0 auto; padding: 20px; }
.prompt { margin: 0 0 16px; font-size: 17px; font-weight: 700; }
.stimuli { display: grid; gap: 12px; align-items: start; margin-bottom: 20px; }
.stimuli.q1 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.stimuli.q2, .stimuli.q3 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
figure { margin: 0; min-width: 0; }
figure img { display: block; width: 100%; height: min(62vh, 683px); object-fit: contain; background: #fff; border: 1px solid #bac0c8; }
figcaption { padding: 7px 2px; text-align: center; color: #4c5561; font-size: 13px; }
.scale { margin: 0 0 14px; padding: 14px 0; border: 0; border-top: 1px solid #d2d6dc; }
.scale legend { padding: 0 0 10px; font-weight: 700; }
.options { display: grid; grid-template-columns: repeat(5, minmax(72px, 1fr)); gap: 8px; }
.option { display: grid; place-items: center; min-height: 52px; background: #fff; border: 1px solid #aeb5bf; cursor: pointer; }
.option:has(input:checked) { border: 2px solid #1463d6; background: #eaf2ff; }
.option input { margin-right: 6px; }
.anchors { display: flex; justify-content: space-between; margin-top: 6px; color: #59626e; font-size: 12px; }
nav { display: flex; justify-content: space-between; gap: 12px; padding-top: 12px; border-top: 1px solid #c7ccd3; }
button { min-height: 42px; padding: 0 16px; border: 1px solid #8f98a4; background: #fff; color: #17191d; font-weight: 700; cursor: pointer; }
button.primary { background: #1463d6; color: #fff; border-color: #1463d6; }
button:disabled { opacity: .45; cursor: default; }
@media (max-width: 760px) {
  main { padding: 12px; }
  .stimuli.q1 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  figure img { height: 42vh; }
  .options { grid-template-columns: repeat(5, minmax(48px, 1fr)); }
}
</style>
</head>
<body>
<header><h1>M2HImage 盲测评分</h1><div id="progress"></div></header>
<main>
  <p class="prompt" id="prompt"></p>
  <section class="stimuli" id="stimuli"></section>
  <section id="scales"></section>
  <nav>
    <button id="previous" type="button">上一题</button>
    <button id="next" class="primary" type="button">下一题</button>
    <button id="export" type="button">导出 CSV</button>
  </nav>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const storageKey = `m2h-human-eval-${payload.rater_id}`;
const answers = JSON.parse(localStorage.getItem(storageKey) || '{}');
let index = 0;
function save() { localStorage.setItem(storageKey, JSON.stringify(answers)); }
function answered(task) { return task.scales.every(s => answers[task.question_id]?.[s.field]); }
function render() {
  const task = payload.tasks[index];
  document.getElementById('progress').textContent = `${payload.rater_id} | ${index + 1} / ${payload.tasks.length} | 已完成 ${payload.tasks.filter(answered).length}`;
  document.getElementById('prompt').textContent = task.prompt;
  const stimuli = document.getElementById('stimuli');
  stimuli.className = `stimuli ${task.question_type}`;
  stimuli.replaceChildren();
  task.assets.forEach((asset, i) => {
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = payload.assets[asset]; image.alt = `Stimulus ${i + 1}`;
    const caption = document.createElement('figcaption'); caption.textContent = task.asset_labels[i];
    figure.append(image, caption); stimuli.append(figure);
  });
  const scales = document.getElementById('scales'); scales.replaceChildren();
  task.scales.forEach(scale => {
    const fieldset = document.createElement('fieldset'); fieldset.className = 'scale';
    const legend = document.createElement('legend'); legend.textContent = scale.label; fieldset.append(legend);
    const options = document.createElement('div'); options.className = 'options';
    for (let value = payload.scale_min; value <= payload.scale_max; value++) {
      const label = document.createElement('label'); label.className = 'option';
      const input = document.createElement('input'); input.type = 'radio'; input.name = scale.field; input.value = value;
      input.checked = Number(answers[task.question_id]?.[scale.field]) === value;
      input.addEventListener('change', () => { answers[task.question_id] ||= {}; answers[task.question_id][scale.field] = value; save(); render(); });
      label.append(input, document.createTextNode(String(value))); options.append(label);
    }
    fieldset.append(options);
    const anchors = document.createElement('div'); anchors.className = 'anchors'; anchors.innerHTML = '<span>1 = 很差 / 完全不一致</span><span>5 = 很好 / 完全一致</span>'; fieldset.append(anchors);
    scales.append(fieldset);
  });
  document.getElementById('previous').disabled = index === 0;
  document.getElementById('next').disabled = index === payload.tasks.length - 1;
}
document.getElementById('previous').onclick = () => { if (index > 0) { index--; render(); scrollTo(0, 0); } };
document.getElementById('next').onclick = () => { if (index + 1 < payload.tasks.length) { index++; render(); scrollTo(0, 0); } };
document.getElementById('export').onclick = () => {
  const incomplete = payload.tasks.filter(task => !answered(task));
  if (incomplete.length) { alert(`还有 ${incomplete.length} 题未完成，完成后才能导出。`); return; }
  const fields = ['rater_id','question_id','question_type','garment_consistency','identity_match','garment_fidelity','realism'];
  const quote = value => `"${String(value ?? '').replaceAll('"','""')}"`;
  const rows = [fields.join(',')];
  payload.tasks.forEach(task => rows.push(fields.map(field => quote(field === 'rater_id' ? payload.rater_id : field === 'question_id' ? task.question_id : field === 'question_type' ? task.question_type : answers[task.question_id]?.[field])).join(',')));
  const blob = new Blob([rows.join('\\n') + '\\n'], {type: 'text/csv;charset=utf-8'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${payload.rater_id}.csv`; link.click(); URL.revokeObjectURL(link.href);
};
render();
</script>
</body>
</html>
'''


def write_sheet(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML_TEMPLATE.replace('__PAYLOAD__', serialized), encoding='utf-8')


def write_distribution_readme(path: Path, key: dict[str, Any]) -> None:
    protocol = key['protocol']
    lines = [
        '# Human Evaluation Distribution',
        '',
        f"- Unique normal questions: `{protocol['normal_question_count']}`",
        f"- Raters: `{protocol['rater_count']}`",
        f"- Planned ratings per normal question: `{protocol['ratings_per_question']}`",
        f"- Per rater: `{protocol['normal_questions_per_rater']}` normal + `{protocol['attention_checks_per_rater']}` checks",
        f"- Seed: `{protocol['seed']}`",
        '- Do not distribute `key.json` to raters. Distribute only the matching `sheets/rater_XX.html` file.',
        '- Each rater opens the HTML locally, completes every item, exports CSV, and returns only that CSV.',
        '- Put returned CSV files under `human_eval/responses/`, then run `human_eval_score.py`.',
        '',
        'Run names, sample IDs, paths, and attention flags are absent from the HTML task payload. Images are resized JPEG data URLs embedded once per sheet; no network or server is required.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build balanced, self-contained blind human-evaluation sheets.')
    parser.add_argument('--config', default='configs/qualitative.yaml')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    context = build_panel_context(args.config)
    qcfg = context['qcfg']
    eval_cfg = qcfg['human_eval']
    root = context['root']
    selection_path = resolve_path(root, qcfg['selection_json'])
    if not selection_path.exists():
        raise FileNotFoundError(f'panel manifest missing: {selection_path}; run make_qual_panels.py first')
    manifest = read_json(selection_path)
    actual = [(row['mid'], row['identity_ids']) for row in manifest['rows']]
    expected = [(mid, context['identities'][mid]) for mid in context['selected']]
    if actual != expected:
        raise RuntimeError('panel manifest does not match deterministic current selection')

    normal = build_normal_questions(context, manifest)
    rater_count = int(eval_cfg['rater_count'])
    ratings_per_question = int(eval_cfg['ratings_per_question'])
    seed = int(qcfg['seed'])
    assignments = assign_questions(normal, rater_count, ratings_per_question, seed)
    questions = {question['question_id']: question for question in normal}
    normal_counts = Counter(qid for values in assignments.values() for qid in values)
    if set(normal_counts.values()) != {ratings_per_question}:
        raise RuntimeError('not every normal question has the requested rating count')

    templates = attention_templates(context, manifest)
    check_count = int(eval_cfg['attention_checks_per_rater'])
    if len(templates) != check_count:
        raise RuntimeError(f'attention template count {len(templates)} != configured {check_count}')
    ordered_assignments: dict[str, list[str]] = {}
    next_id = len(normal) + 1
    for rater_index, (rater_id, qids) in enumerate(sorted(assignments.items())):
        normal_order = spread_questions(qids, questions, seed + 1000 + rater_index)
        checks = []
        for template in templates:
            qid = question_id(next_id)
            next_id += 1
            questions[qid] = {
                **template,
                'question_id': qid,
                'run': None,
                'mid': None,
                'jid': None,
                'seed': None,
                'attention': True,
                'assigned_rater': rater_id,
            }
            checks.append(qid)
        ordered = insert_attention(normal_order, checks)
        for first, second in zip(ordered, ordered[1:]):
            left, right = questions[first].get('mid'), questions[second].get('mid')
            if left is not None and left == right:
                raise RuntimeError(f'adjacent same-mid tasks for {rater_id}: {left}')
        ordered_assignments[rater_id] = ordered

    output_dir = resolve_path(root, eval_cfg['output_dir'])
    sheet_dir = resolve_path(root, eval_cfg['sheet_dir'])
    response_dir = resolve_path(root, eval_cfg['response_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    for rater_id, qids in ordered_assignments.items():
        path = sheet_dir / f'{rater_id}.html'
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'sheet already exists; pass --overwrite: {path}')
        payload = sanitized_payload(rater_id, qids, questions, eval_cfg)
        sanitized_text = json.dumps(payload['tasks'], ensure_ascii=False)
        forbidden = [str(context['root']), '__id', 'b2cont', 'alpha075', 'a4_gen', 'a2_gen']
        leaked = [token for token in forbidden if token in sanitized_text]
        if leaked:
            raise RuntimeError(f'blind payload leaked run/path metadata for {rater_id}: {leaked}')
        write_sheet(path, payload)

    normal_per_rater = len(normal) * ratings_per_question // rater_count
    key = {
        'schema_version': 1,
        'protocol': {
            'seed': seed,
            'selection': str(selection_path),
            'normal_question_count': len(normal),
            'question_counts': dict(Counter(question['question_type'] for question in normal)),
            'rater_count': rater_count,
            'ratings_per_question': ratings_per_question,
            'normal_questions_per_rater': normal_per_rater,
            'attention_checks_per_rater': check_count,
            'max_attention_failures': int(eval_cfg['max_attention_failures']),
            'min_retained_ratings': int(eval_cfg['min_retained_ratings']),
            'identity_positions': [int(value) for value in qcfg['human_identity_positions']],
            'same_mid_ordering': 'no adjacent same-mid normal tasks; greedy two-item recent exclusion',
            'attention_policy': 'exclude a rater when any of six checks fails',
            'scale': [int(eval_cfg['scale_min']), int(eval_cfg['scale_max'])],
            'visible_mean_threshold': float(eval_cfg['visible_mean_threshold']),
            'noninferiority_margin': float(eval_cfg['noninferiority_margin']),
            'blindness': 'HTML contains opaque question/asset IDs and base64 pixels only; key retains run/sample mapping',
        },
        'questions': questions,
        'assignments': ordered_assignments,
    }
    key_path = resolve_path(root, eval_cfg['key'])
    write_json(key_path, key)
    write_distribution_readme(output_dir / 'README.md', key)
    question_rows = [
        {
            'question_id': qid,
            'question_type': question['question_type'],
            'run': question.get('run'),
            'mid': question.get('mid'),
            'jid': question.get('jid'),
            'attention': question['attention'],
            'assigned_count': sum(qid in values for values in ordered_assignments.values()),
        }
        for qid, question in questions.items()
    ]
    write_csv(
        output_dir / 'question_bank.csv',
        question_rows,
        ['question_id', 'question_type', 'run', 'mid', 'jid', 'attention', 'assigned_count'],
    )
    digest = hashlib.sha256(key_path.read_bytes()).hexdigest()[:16]
    print(
        f"built {len(normal)} normal questions, {len(questions) - len(normal)} rater-specific checks, "
        f"and {rater_count} offline sheets under {sheet_dir}; key hash={digest}"
    )


if __name__ == '__main__':
    main()
