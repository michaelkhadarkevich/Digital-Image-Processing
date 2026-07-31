import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INPUT_DIR = Path("data/Data 1/test")
OUTPUT_DIR = Path("result/results task 1/Extra - dividing image by more then 2")
IMAGE_SIZE = (224, 224)
DOWNSCALE_FACTORS = [2, 3, 4, 5, 6, 8, 10]


def find_default_image() -> Path:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = sorted(
        p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
    if not candidates:
        raise FileNotFoundError("No clean image found in data/Data 1/test")
    return candidates[0]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


def downsample(image: Image.Image, factor: int) -> Image.Image:
    low_size = (max(1, image.width // factor), max(1, image.height // factor))
    return image.resize(low_size, Image.Resampling.BICUBIC)


def lanczos_reconstruct(low_resolution: Image.Image) -> Image.Image:
    return low_resolution.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def metrics(original: Image.Image, reconstructed: Image.Image) -> dict[str, float]:
    original_pixels = list(original.getdata())
    reconstructed_pixels = list(reconstructed.getdata())
    count = len(original_pixels) * 3

    squared_error = 0.0
    absolute_error = 0.0
    for original_rgb, reconstructed_rgb in zip(original_pixels, reconstructed_pixels):
        for original_value, reconstructed_value in zip(original_rgb, reconstructed_rgb):
            diff = float(original_value - reconstructed_value)
            squared_error += diff * diff
            absolute_error += abs(diff)

    mse = squared_error / count
    rmse = math.sqrt(mse)
    mae = absolute_error / count
    psnr = math.inf if mse == 0 else 20 * math.log10(255.0 / rmse)
    reconstruction_score = max(0.0, min(100.0, 100.0 * (1.0 - mae / 255.0)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "reconstruction_score": reconstruction_score,
    }


def save_psnr_svg(rows: list[dict[str, float]], output_path: Path) -> None:
    width, height = 760, 420
    margin_left, margin_top, margin_right, margin_bottom = 70, 35, 30, 60
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    psnr_values = [row["psnr"] for row in rows]
    min_y = math.floor(min(psnr_values)) - 1
    max_y = math.ceil(max(psnr_values)) + 1
    if min_y == max_y:
        max_y += 1

    def x_pos(index: int) -> float:
        if len(rows) == 1:
            return margin_left + chart_w / 2
        return margin_left + index * chart_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        return margin_top + chart_h * (1 - (value - min_y) / (max_y - min_y))

    points = [(x_pos(i), y_pos(row["psnr"])) for i, row in enumerate(rows)]
    point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9f7"/>',
        '<text x="30" y="24" font-family="Arial" font-size="18" font-weight="700" fill="#182228">Lanczos quality after stronger downsampling</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#667" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#667" stroke-width="1"/>',
        f'<polyline points="{point_string}" fill="none" stroke="#2666ac" stroke-width="3"/>',
    ]

    for tick in range(min_y, max_y + 1, max(1, (max_y - min_y) // 5)):
        y = y_pos(tick)
        lines.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}" stroke="#d8dddd" stroke-width="1"/>')
        lines.append(f'<text x="18" y="{y + 4:.1f}" font-family="Arial" font-size="11" fill="#56636b">{tick}</text>')

    for i, row in enumerate(rows):
        x, y = points[i]
        factor = int(row["factor"])
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#008074"/>')
        lines.append(f'<text x="{x - 10:.1f}" y="{margin_top + chart_h + 25}" font-family="Arial" font-size="12" fill="#182228">x{factor}</text>')
        lines.append(f'<text x="{x - 18:.1f}" y="{y - 10:.1f}" font-family="Arial" font-size="10" fill="#182228">{row["psnr"]:.2f}</text>')

    lines.append(f'<text x="{width / 2 - 55}" y="{height - 15}" font-family="Arial" font-size="12" fill="#56636b">Downsample factor</text>')
    lines.append('<text x="16" y="220" transform="rotate(-90 16,220)" font-family="Arial" font-size="12" fill="#56636b">PSNR</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_comparison_grid(original: Image.Image, reconstructions: list[tuple[str, Image.Image]], output_path: Path) -> None:
    thumb_size = (112, 112)
    padding = 16
    label_h = 24
    columns = 4
    rows = math.ceil((len(reconstructions) + 1) / columns)
    width = columns * thumb_size[0] + (columns + 1) * padding
    height = rows * (thumb_size[1] + label_h) + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    cells = [("Original", original)] + reconstructions
    for index, (label, image) in enumerate(cells):
        row = index // columns
        col = index % columns
        x = padding + col * (thumb_size[0] + padding)
        y = padding + row * (thumb_size[1] + label_h + padding)
        canvas.paste(image.resize(thumb_size, Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_size[1] + 5), label, fill=(24, 34, 40))

    canvas.save(output_path, optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = find_default_image()
    original = load_image(image_path)

    rows = []
    reconstructions = []
    for factor in DOWNSCALE_FACTORS:
        low_resolution = downsample(original, factor)
        reconstructed = lanczos_reconstruct(low_resolution)
        row = {
            "factor": factor,
            "low_width": low_resolution.width,
            "low_height": low_resolution.height,
            **metrics(original, reconstructed),
        }
        rows.append(row)
        reconstructions.append((f"x{factor}", reconstructed))

    with (OUTPUT_DIR / "lanczos_extra_downscale_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "factor",
                "low_width",
                "low_height",
                "mse",
                "rmse",
                "mae",
                "psnr",
                "reconstruction_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    original.save(OUTPUT_DIR / "original_clean.png", optimize=True)
    save_psnr_svg(rows, OUTPUT_DIR / "lanczos_psnr_by_downscale_factor.svg")
    save_comparison_grid(original, reconstructions, OUTPUT_DIR / "lanczos_extra_downscale_grid.png")
    (OUTPUT_DIR / "README.txt").write_text(
        "\n".join(
            [
                "Extra experiment: dividing image by more then 2",
                "",
                f"Input image: {image_path}",
                "Method: downsample by factor, then reconstruct to 224x224 with Lanczos.",
                "Higher PSNR and reconstruction score are better.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved results to {OUTPUT_DIR}")
    for row in rows:
        print(
            f"x{int(row['factor'])}: low={int(row['low_width'])}x{int(row['low_height'])}, "
            f"PSNR={row['psnr']:.2f}, score={row['reconstruction_score']:.2f}"
        )


if __name__ == "__main__":
    main()
