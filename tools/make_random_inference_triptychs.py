from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


NAME_RE = re.compile(
    r"^(?P<mid>\d+)__id(?P<jid>\d+)__seed(?P<seed>\d+)\.png$"
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def find_source(folder: Path, sample_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    matches = sorted(path for path in folder.glob(f"{sample_id}.*") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one source image for id={sample_id} under {folder}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def contained(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    result = Image.new("RGB", size, "white")
    scaled = image.copy()
    scaled.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return result


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    text_font: ImageFont.ImageFont,
    *,
    fill: str = "#17191c",
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=text_font,
        fill=fill,
    )


def make_triptych(row: dict[str, str | int], output: Path) -> None:
    cell_size = (768, 1024)
    title_h, label_h, footer_h = 64, 44, 38
    width = cell_size[0] * 3
    height = title_h + label_h + cell_size[1] + footer_h
    canvas = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill="#dde3e9")
    centered_text(
        draw,
        (0, 0, width, title_h),
        f"mid={row['mid']}  jid={row['jid']}  seed={row['seed']}",
        font(26, bold=True),
    )

    columns = (
        (Path(str(row["mannequin"])), f"Mannequin input  m_i={row['mid']}"),
        (Path(str(row["identity"])), f"Target identity  c_j={row['jid']}"),
        (Path(str(row["generated"])), "Generated output"),
    )
    image_y = title_h + label_h
    for index, (path, label) in enumerate(columns):
        x = index * cell_size[0]
        draw.rectangle((x, title_h, x + cell_size[0], image_y), fill="white")
        centered_text(
            draw,
            (x, title_h, x + cell_size[0], image_y),
            label,
            font(20, bold=True),
        )
        canvas.paste(contained(open_rgb(path), cell_size), (x, image_y))
        draw.rectangle(
            (x, title_h, x + cell_size[0] - 1, image_y + cell_size[1] - 1),
            outline="#9da5ae",
            width=2,
        )

    footer_y = image_y + cell_size[1]
    centered_text(
        draw,
        (0, footer_y, width, height),
        "Inputs and output are shown at the native 3:4 frame without cropping or stretching.",
        font(16),
        fill="#4e5660",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def make_overview(rows: list[dict[str, str | int]], output: Path) -> None:
    tile_size = (180, 240)
    title_h, row_title_h, gap, outer = 54, 30, 16, 16
    strip_w = tile_size[0] * 3
    strip_h = row_title_h + tile_size[1]
    columns = 2
    row_count = (len(rows) + columns - 1) // columns
    width = outer * 2 + strip_w * columns + gap * (columns - 1)
    height = title_h + outer + strip_h * row_count + gap * (row_count - 1) + outer
    canvas = Image.new("RGB", (width, height), "#eef0f3")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill="#d9e0e7")
    centered_text(
        draw,
        (0, 0, width, title_h),
        "Random 10: mannequin input | target identity | generated output",
        font(19, bold=True),
    )
    for index, row in enumerate(rows):
        grid_x = index % columns
        grid_y = index // columns
        left = outer + grid_x * (strip_w + gap)
        top = title_h + outer + grid_y * (strip_h + gap)
        draw.rectangle((left, top, left + strip_w, top + row_title_h), fill="white")
        centered_text(
            draw,
            (left, top, left + strip_w, top + row_title_h),
            f"#{index + 1:02d}  mid={row['mid']}  jid={row['jid']}  seed={row['seed']}",
            font(13, bold=True),
        )
        paths = (row["mannequin"], row["identity"], row["generated"])
        for column, value in enumerate(paths):
            x = left + column * tile_size[0]
            image = contained(open_rgb(Path(str(value))), tile_size)
            canvas.paste(image, (x, top + row_title_h))
            draw.rectangle(
                (x, top + row_title_h, x + tile_size[0] - 1, top + strip_h - 1),
                outline="#a3aab3",
                width=1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible mannequin/identity/generated triptychs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/muxiangyu/datasets/M2HImage/M2H_Final_v2"),
    )
    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path(
            "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/"
            "spatial_quality_repair_gen"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial_quality_repair_random10"),
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(args.gen_dir.glob("*.png")):
        match = NAME_RE.match(path.name)
        if match is not None:
            candidates.append((path, match))
    if len(candidates) < args.count:
        raise RuntimeError(
            f"requested {args.count} samples but found {len(candidates)} valid outputs"
        )

    selected = random.Random(args.seed).sample(candidates, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for index, (generated, match) in enumerate(selected, start=1):
        mid = match.group("mid")
        jid = match.group("jid")
        sample_seed = int(match.group("seed"))
        row: dict[str, str | int] = {
            "order": index,
            "mid": mid,
            "jid": jid,
            "seed": sample_seed,
            "mannequin": str(find_source(args.root / "images/mannequin", mid)),
            "identity": str(find_source(args.root / "images/human", jid)),
            "generated": str(generated),
        }
        panel = args.output_dir / (
            f"sample_{index:02d}__mid{mid}__jid{jid}__seed{sample_seed}.png"
        )
        row["panel"] = str(panel.resolve())
        make_triptych(row, panel)
        rows.append(row)

    overview = args.output_dir / "overview_random10.png"
    make_overview(rows, overview)
    manifest = {
        "selection": "uniform random sample without replacement from generated PNG files",
        "selection_seed": args.seed,
        "available_outputs": len(candidates),
        "selected_count": len(rows),
        "columns": ["mannequin input", "target identity input", "generated output"],
        "overview": str(overview.resolve()),
        "samples": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Spatial Quality Repair Random 10",
        "",
        f"- Selection seed: `{args.seed}`",
        f"- Available outputs: `{len(candidates)}`",
        "- Columns: mannequin input | target identity input | generated output",
        "- Images are contained without cropping or stretching.",
        "",
        f"Overview: [overview_random10.png]({overview.name})",
        "",
        "| # | mid | jid | seed | panel |",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        panel_name = Path(str(row["panel"])).name
        lines.append(
            f"| {row['order']} | `{row['mid']}` | `{row['jid']}` | "
            f"{row['seed']} | [{panel_name}]({panel_name}) |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
