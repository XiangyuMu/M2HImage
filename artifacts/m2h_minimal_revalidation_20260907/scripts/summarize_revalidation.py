"""Summarize M2H minimal revalidation metrics.

This script is descriptive only: two-factor pigeonhole bootstrap intervals are
exploratory, use source-group/reference-file proxies, and do not estimate
training-seed uncertainty.
"""
import argparse
from collections import Counter, defaultdict
import math
import hashlib
import json
from pathlib import Path
import random
import statistics


ARMS = (
    "b2_legacyM_freshI",
    "a4_legacyM_freshI",
    "b2_inputOnly_freshI",
    "a4_inputOnly_freshI",
)
METRIC_FIELDS = (
    "garment_dino",
    "garment_hf_lpips",
    "id_penalized",
    "body_penalized",
    "head_penalized",
    "body_coverage",
    "head_coverage",
)
KEY_FIELDS = ("mid", "jid", "seed")
FULL_KEY_FIELDS = ("mid", "jid", "seed", "branch", "tau", "k")
INFRA_FAILURE_STATUSES = {"failed", "reference_failed", "ref_failed"}
ALLOWED_STATUSES = {"ok", "output_face_failed", "pose_failed"}


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl") from exc
    return rows


def write_jsonl(path, rows):
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_key(row):
    require_fields(row, FULL_KEY_FIELDS, "row key")
    return tuple(row[field] for field in FULL_KEY_FIELDS)


def pair_key(row):
    require_fields(row, KEY_FIELDS, "pair key")
    return tuple(row[field] for field in KEY_FIELDS)


def require_fields(row, fields, label):
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"missing {label} field(s): {missing}")


def is_number_or_none(value):
    return value is None or isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_metric_row(row):
    require_fields(row, FULL_KEY_FIELDS, "metric row key")
    require_fields(row, ("metric_status",), "metric status")
    status = row.get("metric_status")
    if status in INFRA_FAILURE_STATUSES:
        raise ValueError(f"metric infrastructure status blocks READY: {status} at {full_key(row)}")
    if status is not None and status not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported metric_status {status!r} at {full_key(row)}")
    return row


def load_manifest_pairs(path):
    if not path:
        return None, {}, None
    data = json.loads(path.read_text())
    pair_rows = data.get("pairs", data if isinstance(data, list) else None)
    if not isinstance(pair_rows, list):
        raise ValueError("--manifest-json must contain a list or an object with pairs")
    mapping = {}
    for row in pair_rows:
        require_fields(row, ("mid", "jid"), "manifest pair")
        key = (row["mid"], row["jid"])
        if key in mapping:
            raise ValueError(f"duplicate manifest pair: {key}")
        mapping[key] = dict(row)
    return set(mapping), mapping, data


def load_sensitivity_pairs(path, manifest_by_pair, manifest_data=None):
    if not path:
        return None
    data = json.loads(path.read_text())
    return sensitivity_pairs_from_data(data, manifest_by_pair, manifest_data=manifest_data)


def sensitivity_pairs_from_data(data, manifest_by_pair, manifest_data=None):
    if "sensitivity_pairs" in data:
        pairs = set()
        for item in data["sensitivity_pairs"]:
            if isinstance(item, dict):
                require_fields(item, ("mid", "jid"), "sensitivity pair")
                pairs.add((item["mid"], item["jid"]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.add((item[0], item[1]))
            else:
                raise ValueError("sensitivity_pairs entries must be objects or [mid,jid] lists")
        return {
            "mode": "sensitivity_pairs",
            "pairs": pairs,
            "excluded_mids": set(),
            "source": "explicit pair list",
        }
    if "sensitivity_pair_indices" in data:
        source_data = data if "pairs" in data else manifest_data
        if not source_data or "pairs" not in source_data:
            raise ValueError("sensitivity_pair_indices requires pairs in --sensitivity-json or --manifest-json")
        by_index = {}
        for position, pair in enumerate(source_data["pairs"]):
            index = pair.get("pair_index", position)
            if index in by_index:
                raise ValueError(f"duplicate pair_index in manifest pairs: {index}")
            require_fields(pair, ("mid", "jid"), "manifest pair")
            by_index[index] = (pair["mid"], pair["jid"])
        pairs = set()
        for index in data["sensitivity_pair_indices"]:
            if index not in by_index:
                raise ValueError(f"sensitivity_pair_indices entry absent from manifest pairs: {index}")
            pairs.add(by_index[index])
        return {
            "mode": "sensitivity_pair_indices",
            "pairs": pairs,
            "excluded_mids": set(),
            "source": "protocol manifest indices",
        }
    excluded = data.get("excluded_mids")
    if not isinstance(excluded, list):
        raise ValueError("--sensitivity-json must contain sensitivity_pair_indices, sensitivity_pairs, or excluded_mids")
    excluded_mids = set(excluded)
    pairs = {key for key in manifest_by_pair if key[0] not in excluded_mids} if manifest_by_pair else None
    return {
        "mode": "excluded_mids",
        "pairs": pairs,
        "excluded_mids": excluded_mids,
        "source": "legacy excluded_mids compatibility",
    }


def merge_metrics(run_dir):
    inputs = []
    metric_paths = [
        run_dir / "metrics_image" / "per_image.jsonl",
        run_dir / "metrics_id" / "per_image.jsonl",
        run_dir / "metrics_pose" / "per_image.jsonl",
    ]
    merged = {}
    status_counts = {}
    for path in metric_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs.append(path)
        rows = [canonical_metric_row(row) for row in read_jsonl(path)]
        stage = path.parent.name.replace("metrics_", "")
        keys = [full_key(row) for row in rows]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate metric row(s) in {path}: {duplicates[:5]}")
        status_counts[stage] = dict(Counter(row.get("metric_status", "missing") for row in rows))
        if not merged:
            merged = {full_key(row): base_output_row(row) for row in rows}
        elif set(keys) != set(merged):
            raise ValueError(f"metric row coverage mismatch in {path}")
        for row in rows:
            key = full_key(row)
            for field in METRIC_FIELDS:
                if field in row:
                    value = row[field]
                    if value is not None and not is_finite_number(value):
                        raise ValueError(f"non-numeric metric {field} at {key}: {value!r}")
                    merged[key][field] = value
            merged[key][f"status_{stage}"] = row.get("metric_status")
            if row.get("metric_status") == "output_face_failed" and merged[key].get("id_penalized") is None:
                merged[key]["id_penalized"] = -1
    return list(merged.values()), status_counts, inputs


def base_output_row(row):
    out = {field: row[field] for field in FULL_KEY_FIELDS}
    for field in ("path", "m_source_group"):
        if field in row:
            out[field] = row[field]
    for field in METRIC_FIELDS:
        if field in row:
            out[field] = row[field]
    return out


def attach_manifest(rows, manifest_by_pair):
    if not manifest_by_pair:
        return
    for row in rows:
        meta = manifest_by_pair.get((row["mid"], row["jid"]))
        if meta is None:
            raise ValueError(f"row absent from manifest pairs: {(row['mid'], row['jid'])}")
        if "m_source_group" in meta:
            row["m_source_group"] = meta["m_source_group"]


def validate_rows(rows, manifest_pairs=None):
    if not rows:
        raise ValueError("no metric rows")
    bad_branches = sorted({row["branch"] for row in rows} - set(ARMS))
    if bad_branches:
        raise ValueError(f"unexpected branch(es): {bad_branches}")
    if any(row["tau"] != 0 or row["k"] != 1 for row in rows):
        raise ValueError("all rows must have tau=0 and k=1")
    keys = [full_key(row) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate full key(s): {duplicates[:5]}")
    by_arm = {arm: {pair_key(row) for row in rows if row["branch"] == arm} for arm in ARMS}
    counts = {arm: len(keys) for arm, keys in by_arm.items()}
    if any(count != 128 for count in counts.values()):
        raise ValueError(f"each arm must contain exactly 128 paired rows: {counts}")
    first = by_arm[ARMS[0]]
    for arm, arm_keys in by_arm.items():
        if arm_keys != first:
            raise ValueError(f"{arm} does not share the common 128-pair key set")
    if manifest_pairs is not None:
        row_pairs = {(mid, jid) for mid, jid, _seed in first}
        if row_pairs != manifest_pairs:
            raise ValueError("run pair set does not match manifest pairs")
    missing = {}
    for field in METRIC_FIELDS:
        bad = [full_key(row) for row in rows if row.get(field) is None]
        if bad:
            missing[field] = bad[:5]
    if missing:
        raise ValueError(f"critical metric None blocks READY: {missing}")


def source_group(row):
    return row.get("m_source_group", row["mid"])


def means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["branch"]].append(row)
    result = {}
    for arm in ARMS:
        arm_rows = grouped.get(arm, [])
        result[arm] = {
            "n": len(arm_rows),
            "source_groups": len({source_group(row) for row in arm_rows}),
            "reference_files": len({row["jid"] for row in arm_rows}),
            "metrics": {
                field: {
                    "n": len(arm_rows),
                    "mean": statistics.fmean(row[field] for row in arm_rows) if arm_rows else None,
                }
                for field in METRIC_FIELDS
            },
        }
    return result


def bootstrap_weights(pairs, draws=10000, seed=20260907):
    sources = sorted({source_group(row) for row in pairs})
    refs = sorted({row["jid"] for row in pairs})
    rng = random.Random(seed)
    weights = []
    for _draw in range(draws):
        source_counts = Counter(rng.choice(sources) for _ in sources)
        ref_counts = Counter(rng.choice(refs) for _ in refs)
        weights.append([source_counts[source_group(row)] * ref_counts[row["jid"]] for row in pairs])
    return weights, {
        "draws": draws,
        "seed": seed,
        "source_groups": len(sources),
        "reference_files": len(refs),
    }


def quantile(sorted_values, q):
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_delta(left_rows, right_rows, field, draws=10000, seed=20260907):
    left = {pair_key(row): row for row in left_rows}
    right = {pair_key(row): row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("unmatched experimental pairs")
    keys = sorted(left)
    deltas = [left[key][field] - right[key][field] for key in keys]
    base_rows = [left[key] for key in keys]
    weights, info = bootstrap_weights(base_rows, draws=draws, seed=seed)
    samples = []
    for draw_weights in weights:
        denom = sum(draw_weights)
        if denom > 0:
            samples.append(sum(weight * delta for weight, delta in zip(draw_weights, deltas)) / denom)
    samples.sort()
    return {
        "n": len(keys),
        "delta_mean": float(statistics.fmean(deltas)),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "bootstrap": {**info, "nonempty_replicates": len(samples)},
    }


def did_delta(a4_legacy, b2_legacy, a4_input, b2_input, field, draws=10000, seed=20260907):
    arms = [{pair_key(row): row for row in rows} for rows in (a4_legacy, b2_legacy, a4_input, b2_input)]
    common = set.intersection(*(set(arm) for arm in arms))
    if any(set(arm) != common for arm in arms):
        raise ValueError("unmatched DID pairs")
    keys = sorted(common)
    values = [
        (arms[2][key][field] - arms[3][key][field])
        - (arms[0][key][field] - arms[1][key][field])
        for key in keys
    ]
    base_rows = [arms[0][key] for key in keys]
    weights, info = bootstrap_weights(base_rows, draws=draws, seed=seed)
    samples = []
    for draw_weights in weights:
        denom = sum(draw_weights)
        if denom > 0:
            samples.append(sum(weight * value for weight, value in zip(draw_weights, values)) / denom)
    samples.sort()
    return {
        "n": len(keys),
        "delta_mean": float(statistics.fmean(values)),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "definition": "(A4-B2 under inputOnly) - (A4-B2 under legacyM)",
        "bootstrap": {**info, "nonempty_replicates": len(samples)},
    }


def compute_effects(rows, draws=10000, seed=20260907):
    by_arm = {arm: [row for row in rows if row["branch"] == arm] for arm in ARMS}
    result = {
        "a4_minus_b2": {
            "legacyM_freshI": {
                field: paired_delta(by_arm["a4_legacyM_freshI"], by_arm["b2_legacyM_freshI"], field, draws, seed)
                for field in METRIC_FIELDS
            },
            "inputOnly_freshI": {
                field: paired_delta(by_arm["a4_inputOnly_freshI"], by_arm["b2_inputOnly_freshI"], field, draws, seed)
                for field in METRIC_FIELDS
            },
        },
        "difference_in_differences": {
            field: did_delta(
                by_arm["a4_legacyM_freshI"],
                by_arm["b2_legacyM_freshI"],
                by_arm["a4_inputOnly_freshI"],
                by_arm["b2_inputOnly_freshI"],
                field,
                draws,
                seed,
            )
            for field in METRIC_FIELDS
        },
    }
    return result


def pose_qualification(rows):
    by_pair_branch = {(pair_key(row), row["branch"]): row for row in rows}
    pair_keys = sorted({pair_key(row) for row in rows})
    result = {}
    for part, field in (("body", "body_coverage"), ("head", "head_coverage")):
        qualified = []
        unqualified = []
        for key in pair_keys:
            coverages = [by_pair_branch[(key, arm)][field] for arm in ARMS]
            if all(value > 0 for value in coverages):
                qualified.append(key)
            else:
                unqualified.append(key)
        result[part] = {
            "label": "joint_detection_support",
            "symmetric_qualified_pairs": len(qualified),
            "symmetric_unqualified_pairs": len(unqualified),
            "denominator_note": "joint detection support only; true input qualification comes from M scores>=.3 and this statistic does not filter samples",
        }
    return result


def subset_rows(rows, pair_keys=None, excluded_mids=None):
    excluded_mids = excluded_mids or set()
    if pair_keys is None and not excluded_mids:
        return list(rows)
    if pair_keys is not None:
        return [row for row in rows if (row["mid"], row["jid"]) in pair_keys]
    return [row for row in rows if row["mid"] not in excluded_mids]


def subset_summary(rows, selector, draws, seed):
    subset = subset_rows(rows, pair_keys=selector.get("pairs"), excluded_mids=selector.get("excluded_mids"))
    validate_rows_relaxed_subset(subset)
    removed_pairs = len({(row["mid"], row["jid"]) for row in rows}) - len({(row["mid"], row["jid"]) for row in subset})
    return {
        "mode": selector["mode"],
        "source": selector["source"],
        "excluded_mids": sorted(selector.get("excluded_mids", set())),
        "excluded_mid_count": len(selector.get("excluded_mids", set())),
        "removed_pairs": removed_pairs,
        "n_pairs": len({pair_key(row) for row in subset}),
        "source_groups": len({source_group(row) for row in subset}),
        "reference_files": len({row["jid"] for row in subset}),
        "interpretation": "strict reference-sensitivity; candidate-stripped subset, not certified clean",
        "means": means(subset),
        "effects": compute_effects(subset, draws=draws, seed=seed),
    }


def validate_rows_relaxed_subset(rows):
    by_arm = {arm: {pair_key(row) for row in rows if row["branch"] == arm} for arm in ARMS}
    first = by_arm[ARMS[0]]
    for arm, keys in by_arm.items():
        if keys != first:
            raise ValueError(f"sensitivity subset is not symmetric for {arm}")


def build_report(summary):
    def fmt(effect, field):
        item = effect[field]
        ci = item["ci95"]
        return f"{item['delta_mean']:.6g} [{ci[0]:.6g}, {ci[1]:.6g}]"

    lines = [
        "# M2H 最小复验客观指标摘要",
        "",
        "状态：READY 表示三类 per-image 指标已完整合并、4 个 arm 在同一 128 个 `(mid,jid,seed)` 键上配对、关键指标无 None、未发现基础设施 failed/reference_failed。",
        "",
        "解释边界：本摘要是描述性复验，不自动晋级训练，不给出方法成功结论；区间为 source-group/ref-file 双向 pigeonhole bootstrap，10000 draws，seed=20260907，不估计训练 seed 不确定性。",
        "",
        "## 样本与资格",
        "",
        f"- 主结果 pair 数：{summary['n_pairs']}；source group：{summary['source_groups']}；reference file：{summary['reference_files']}",
        f"- 身体关键点共同检出支持（不用于筛样本）：{summary['pose_qualification']['body']['symmetric_qualified_pairs']}/{summary['n_pairs']}",
        f"- 头部关键点共同检出支持（不用于筛样本）：{summary['pose_qualification']['head']['symmetric_qualified_pairs']}/{summary['n_pairs']}",
        "",
        "## A4-B2 差值与 DID",
        "",
        "| 指标 | legacyM A4-B2 | inputOnly A4-B2 | DID |",
        "|---|---:|---:|---:|",
    ]
    for field in METRIC_FIELDS:
        lines.append(
            f"| {field} | "
            f"{fmt(summary['effects']['a4_minus_b2']['legacyM_freshI'], field)} | "
            f"{fmt(summary['effects']['a4_minus_b2']['inputOnly_freshI'], field)} | "
            f"{fmt(summary['effects']['difference_in_differences'], field)} |"
        )
    if summary.get("sensitivity"):
        sens = summary["sensitivity"]
        lines.extend(
            [
                "",
                "## 敏感性子集",
                "",
                f"- 模式：{sens['mode']}；移除 pair：{sens['removed_pairs']}；保留 pair：{sens['n_pairs']}；source group：{sens['source_groups']}；reference file：{sens['reference_files']}",
                "- 这是 strict reference-sensitivity：GlobalSSIM candidate 过滤后的 candidate-stripped 子集，不是 certified clean。",
                "- I 侧重复风险按 reference-sensitivity 报告，不自动等同于答案泄露。",
                "- 同一敏感性 pair 规则已同时作用于 4 个 arm。",
            ]
        )
    lines.extend(
        [
            "",
            "## 字段策略",
            "",
            "- 只汇总 garment_dino、garment_hf_lpips、id_penalized、body_penalized、head_penalized、body_coverage、head_coverage。",
            "- 所有 carryin 字段按协议视为 adapter self image 衍生，不进入均值、差值或区间。",
            "- 协议 GlobalSSIM 是全局统计 proxy，不等同于 skimage 局部 SSIM。",
            "- joint_detection_support 只是共同检出支持，不是输入资格；真实资格来自 M scores>=.3，本统计不筛样本。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--sensitivity-json", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    args = parser.parse_args(argv)

    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest_pairs, manifest_by_pair, manifest_data = load_manifest_pairs(args.manifest_json)
    rows, status_counts, inputs = merge_metrics(args.run)
    attach_manifest(rows, manifest_by_pair)
    validate_rows(rows, manifest_pairs=manifest_pairs)
    sensitivity_selector = load_sensitivity_pairs(
        args.sensitivity_json,
        manifest_by_pair,
        manifest_data=manifest_data,
    )

    summary = {
        "formal_result": False,
        "training_steps": 0,
        "protocol": "M2H minimal revalidation, descriptive only",
        "n_rows": len(rows),
        "n_pairs": len({pair_key(row) for row in rows}),
        "source_groups": len({source_group(row) for row in rows}),
        "reference_files": len({row["jid"] for row in rows}),
        "branches": list(ARMS),
        "metric_fields": list(METRIC_FIELDS),
        "ignored_field_policy": "carryin fields are adapter self-image derived and ignored",
        "status_counts": status_counts,
        "pose_qualification": pose_qualification(rows),
        "joint_detection_support": pose_qualification(rows),
        "means": means(rows),
        "effects": compute_effects(rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed),
        "statistics": f"Exploratory two-factor pigeonhole bootstrap on source-group/ref-file proxies, {args.bootstrap_draws} draws seed{args.bootstrap_seed}; paired rows share one weight scheme; no training-seed inference.",
        "warnings": [
            "READY is blocked by infrastructure failed/reference_failed or critical metric None.",
            "output_face_failed may be retained with id_penalized=-1.",
            "pose qualification denominators are reported, not used to auto-pass rows.",
            "Protocol GlobalSSIM is a global-statistic proxy, not skimage local SSIM.",
            "Sensitivity subset is candidate-stripped strict reference-sensitivity, not certified clean.",
            "I-side duplication risk is reported separately and is not automatically answer leakage.",
            "No training promotion or method-success claim is made by this summary.",
        ],
        "sha256": {str(path): digest(path) for path in inputs},
    }
    if sensitivity_selector:
        summary["sensitivity"] = subset_summary(rows, sensitivity_selector, args.bootstrap_draws, args.bootstrap_seed)
    summary["script_sha256"] = digest(Path(__file__))

    args.out.mkdir(parents=True)
    write_jsonl(args.out / "rows.jsonl", sorted(rows, key=full_key))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True))
    (args.out / "REPORT.md").write_text(build_report(summary))
    (args.out / "READY").write_text("objective metrics joined; descriptive paired summaries computed\n")
    print(json.dumps({"rows": len(rows), "pairs": summary["n_pairs"], "ready": True}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
