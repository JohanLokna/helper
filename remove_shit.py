from pathlib import Path
import os

yes_f = Path("data/unique_images/yes")
yes = [p.stem for p in yes_f.glob("./*")]
mask_f = Path("data/masks")

for p in yes:
    if len(list(mask_f.glob(f"./{p}.npy"))) == 0:
        for shit in yes_f.glob(f"./{p}*"):
            os.remove(shit)
