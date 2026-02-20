import os
from pathlib import Path
import random
import pandas as pd

# ====== НАСТРОЙКИ ======
SEED = 42

# Укажи пути до папок (можно абсолютные, можно относительные)
EXDARK_ROOT = Path("ExDark")   # папка с категориями типа Bicycle/Boat/...
IMAGES_ROOT = Path("Images")   # папка с категориями типа airport_inside/bedroom/...

OUT_WITH_NIGHT = "night_labels.csv"
OUT_COMPAT = "night_labels_compat.csv"  # без колонки night (как ты просил)

# какие расширения считать картинками
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =======================


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_images_recursive(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if is_image(p)]


def norm_path(p: Path) -> str:
    # нормализуем под ОС; можно заменить на str(p.resolve()) если хочешь абсолютные пути
    return os.path.normpath(str(p))


def main():
    random.seed(SEED)

    rows = []

    # -------- ExDark: night=1, under=random(0/1), blur=0, over=0 --------
    if not EXDARK_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка ExDark: {EXDARK_ROOT}")

    exdark_files = list_images_recursive(EXDARK_ROOT)

    for p in exdark_files:
        under = random.randint(0, 1)
        rows.append({
            "path": norm_path(p),
            "blur": 0,
            "under": under,
            "over": 0,
            "night": 1,

            # поля "как в твоей схеме"; для этих данных можно оставить 0
            "dist_type": 0,
            "dist_level": 0,

            # ref можно оставить как путь до исходного файла (как ты просил)
            "ref": norm_path(p),
        })

    # -------- Images: night=0; внутри каждой папки половина under=1, половина under=0 --------
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка Images: {IMAGES_ROOT}")

    subdirs = [d for d in IMAGES_ROOT.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: x.name.lower())

    for d in subdirs:
        files = list_images_recursive(d)
        if not files:
            continue

        random.shuffle(files)
        half = len(files) // 2  # первая половина -> under=1

        for i, p in enumerate(files):
            under = 1 if i < half else 0
            rows.append({
                "path": norm_path(p),
                "blur": 0,
                "under": under,
                "over": 0,
                "night": 0,

                "dist_type": 0,
                "dist_level": 0,
                "ref": norm_path(p),
            })

    # -------- Сохраняем --------
    df = pd.DataFrame(rows)

    # базовая проверка
    if df.empty:
        raise RuntimeError("Не нашёл ни одного изображения. Проверь пути и расширения файлов.")

    df.to_csv(OUT_WITH_NIGHT, index=False)

    # совместимая версия БЕЗ night (строго по твоему списку полей)
    compat_cols = ["path", "blur", "under", "over", "dist_type", "dist_level", "ref"]
    df[compat_cols].to_csv(OUT_COMPAT, index=False)

    # -------- Статистика (для контроля) --------
    total = len(df)
    ex_cnt = (df["night"] == 1).sum()
    im_cnt = (df["night"] == 0).sum()
    under_pos = (df["under"] == 1).sum()

    print("[OK] Готово.")
    print(f"  Всего изображений: {total}")
    print(f"  ExDark (night=1): {ex_cnt}")
    print(f"  Images (night=0): {im_cnt}")
    print(f"  under=1 всего: {under_pos}")
    print(f"  Сохранено:\n    - {OUT_WITH_NIGHT}\n    - {OUT_COMPAT}")


if __name__ == "__main__":
    main()
