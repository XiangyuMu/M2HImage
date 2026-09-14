from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ZH_HEADERS = {
    "annotator_id": "标注者编号",
    "image_id": "图像编号",
    "unit_id": "标注单元编号",
    "feature_family": "特征族代码",
    "attribute": "属性代码",
    "support_label": "支持标签代码",
    "occlusion_order": "遮挡顺序代码",
    "owner_observable": "所有者是否可观察",
    "unit_weight": "单元权重",
    "canvas_width": "画布宽度",
    "canvas_height": "画布高度",
    "boundary_polygon_json": "边界多边形JSON",
    "artifacts": "伪影",
    "notes": "备注",
    "adjudicator_id": "裁决者编号",
    "adjudication_reason": "裁决理由",
}

FAMILY_ZH = {
    "identity_morphology": "身份形态",
    "hair_appearance_geometry": "头发外观与几何",
    "global_geometry_pose": "整体几何与姿态",
    "garment_appearance": "服装外观",
    "garment_drape": "服装垂坠",
    "scene": "场景",
    "occlusion": "遮挡与交互",
}
ATTRIBUTE_ZH = {
    "face_morphology": "面部形态",
    "reliable_local_hair": "可靠的局部头发",
    "unreliable_local_hair": "不可靠的局部头发",
    "global_head_pose": "头部位置与姿态",
    "body_morphology_pose": "身体形态与姿态",
    "garment_design": "服装设计",
    "garment_gross_drape": "服装整体垂坠",
    "background_scene": "背景与场景",
    "hair_face_contact": "头发与脸部接触",
    "hair_garment_contact": "头发与服装接触",
    "unresolved": "未解决区域",
}
SUPPORT_ZH = {
    "S_I": "身份图像专属",
    "S_M": "人台/场景图像专属",
    "S_X": "交互或遮挡",
    "S_U": "不确定/不归属",
}
OCCLUSION_ZH = {
    "none": "无遮挡",
    "hair_over_face": "头发遮挡脸部",
    "face_over_hair": "脸部覆盖头发",
    "hair_over_garment": "头发遮挡服装",
    "garment_over_hair": "服装遮挡头发",
    "ambiguous": "遮挡顺序不明确",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    try:
        return ImageFont.truetype(path, size=size, index=0)
    except OSError:
        return ImageFont.load_default()


def stitch_pair(human_path: Path, mannequin_path: Path, output_path: Path, image_id: str) -> None:
    human = Image.open(human_path).convert("RGB")
    mannequin = Image.open(mannequin_path).convert("RGB")
    if human.size != mannequin.size:
        raise ValueError(f"{image_id}: image sizes differ: {human.size} vs {mannequin.size}")
    width, height = human.size
    header_height = 72
    canvas = Image.new("RGB", (width * 2, height + header_height), "white")
    canvas.paste(human, (0, header_height))
    canvas.paste(mannequin, (width, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = font(30, bold=True)
    small_font = font(22)
    draw.rectangle((0, 0, width, header_height), fill=(34, 89, 160))
    draw.rectangle((width, 0, width * 2, header_height), fill=(78, 118, 68))
    draw.text((24, 18), "真人图像（Identity）", fill="white", font=title_font)
    draw.text((width + 24, 18), "人台图像（Mannequin）", fill="white", font=title_font)
    draw.rectangle((0, height + header_height - 3, width * 2, height + header_height), fill=(0, 0, 0))
    draw.text((24, height + header_height - 32), f"image_id: {image_id}", fill=(40, 40, 40), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, subsampling=0)


def make_zh_row(row: dict[str, str], pair_path: str, human_rel: str, mannequin_rel: str, adjudication: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, header in ZH_HEADERS.items():
        output[header] = row.get(key, "")
    output["特征族中文说明"] = FAMILY_ZH.get(row.get("feature_family", ""), "")
    output["属性中文说明"] = ATTRIBUTE_ZH.get(row.get("attribute", ""), "")
    output["支持标签中文说明"] = SUPPORT_ZH.get(row.get("support_label", ""), "")
    output["遮挡顺序中文说明"] = OCCLUSION_ZH.get(row.get("occlusion_order", ""), "")
    output["拼接图相对路径"] = pair_path
    output["真人原图相对路径"] = human_rel
    output["人台原图相对路径"] = mannequin_rel
    return output


def prepare(pack: Path, output: Path, font_check: bool = True) -> dict[str, Any]:
    selection = pack / "selection_manifest.csv"
    selected_rows = read_csv(selection)
    if not selected_rows:
        raise ValueError("selection_manifest.csv is empty")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source_dir = output / "source_images"
    stitched_dir = output / "stitched_pairs"
    rater_dir = output / "paired_images"
    source_dir.mkdir(parents=True)
    stitched_dir.mkdir(parents=True)
    rater_dir.mkdir(parents=True)

    zh_rows_by_rater = {index: [] for index in (1, 2, 3)}
    copied = []
    for row in selected_rows:
        image_id = row["image_id"]
        human_src = Path(row["human_path"])
        mannequin_src = Path(row["mannequin_path"])
        if not human_src.is_file() or not mannequin_src.is_file():
            raise FileNotFoundError(f"missing source image for {image_id}")
        human_dst = source_dir / "human" / f"{image_id}{human_src.suffix.lower()}"
        mannequin_dst = source_dir / "mannequin" / f"{image_id}{mannequin_src.suffix.lower()}"
        human_dst.parent.mkdir(parents=True, exist_ok=True)
        mannequin_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(human_src, human_dst)
        shutil.copy2(mannequin_src, mannequin_dst)
        pair_name = f"{image_id}_pair.jpg"
        pair_dst = stitched_dir / pair_name
        stitch_pair(human_dst, mannequin_dst, pair_dst, image_id)
        for index in (1, 2, 3):
            rater_pair_dir = rater_dir / f"rater_{index:02d}"
            rater_pair_dir.mkdir(parents=True, exist_ok=True)
            os.link(pair_dst, rater_pair_dir / pair_name)
        copied.append({
            "image_id": image_id,
            "source_dataset": row["source_dataset"],
            "human_relative_path": str(human_dst.relative_to(output)),
            "mannequin_relative_path": str(mannequin_dst.relative_to(output)),
            "stitched_relative_path": str(pair_dst.relative_to(output)),
            "human_sha256": sha256_file(human_dst),
            "mannequin_sha256": sha256_file(mannequin_dst),
            "stitched_sha256": sha256_file(pair_dst),
        })

    zh_fields = list(ZH_HEADERS.values()) + ["特征族中文说明", "属性中文说明", "支持标签中文说明", "遮挡顺序中文说明", "拼接图相对路径", "真人原图相对路径", "人台原图相对路径"]
    for index in (1, 2, 3):
        rows = read_csv(pack / f"annotations_rater_{index:02d}.csv")
        by_image = {row["image_id"]: row for row in copied}
        zh_rows_by_rater[index] = [
            make_zh_row(row, by_image[row["image_id"]]["stitched_relative_path"], by_image[row["image_id"]]["human_relative_path"], by_image[row["image_id"]]["mannequin_relative_path"])
            for row in rows
        ]
        write_csv(output / f"annotations_rater_{index:02d}_中文.csv", zh_fields, zh_rows_by_rater[index])

    adjudication_rows = read_csv(pack / "adjudication.csv")
    by_image = {row["image_id"]: row for row in copied}
    write_csv(output / "adjudication_中文.csv", zh_fields, [
        make_zh_row(row, by_image[row["image_id"]]["stitched_relative_path"], by_image[row["image_id"]]["human_relative_path"], by_image[row["image_id"]]["mannequin_relative_path"], adjudication=True)
        for row in adjudication_rows
    ])

    metadata = {
        "schema_version": 1,
        "purpose": "G2 Chinese annotation workspace with copied source pairs and stitched previews",
        "source_pack": str(pack),
        "source_selection_manifest_sha256": sha256_file(selection),
        "sample_count": len(copied),
        "copied_source_image_count": len(copied) * 2,
        "stitched_pair_count": len(copied),
        "rater_hardlink_count": len(copied) * 3,
        "image_layout": "human left, mannequin right; 768x1024 panels; 1536x1096 composite",
        "machine_csvs_preserved": True,
        "chinese_csvs": [f"annotations_rater_{index:02d}_中文.csv" for index in (1, 2, 3)] + ["adjudication_中文.csv"],
        "files": copied,
    }
    (output / "asset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pack, args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
