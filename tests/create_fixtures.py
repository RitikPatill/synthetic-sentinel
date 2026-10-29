"""
One-off script to generate test fixture images.

Run once: python tests/create_fixtures.py
Commits the resulting JPEGs so tests don't regenerate on every run.
"""

from pathlib import Path

import numpy as np
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)


def create_real_jpg():
    """
    Simulate a real camera photo: random Gaussian noise over a natural-looking
    gradient base. No EXIF is embedded (both fixtures lack EXIF so the noise-floor
    signal carries the test), but the noise is high enough to score low on
    score_noise_floor.
    """
    rng = np.random.default_rng(42)
    size = (128, 128)

    # Gradient base (blue-sky-ish gradient)
    base = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        base[y, :, 0] = int(100 + y * 0.5)   # R
        base[y, :, 1] = int(140 + y * 0.4)   # G
        base[y, :, 2] = int(200 - y * 0.5)   # B

    # Add strong sensor-like Gaussian noise (std ≈ 18 → noise_floor score low)
    noise = rng.normal(0, 18, (*size, 3)).astype(np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(noisy, mode="RGB")
    # Save with a lower quality to encourage natural DCT roll-off
    img.save(FIXTURES / "real.jpg", format="JPEG", quality=75)
    print(f"Created {FIXTURES / 'real.jpg'}")


def create_synthetic_jpg():
    """
    Simulate an AI-generated image: smooth gradient with virtually no noise.
    No EXIF, flat appearance → scores high on noise_floor.
    Saved at quality=95 → flatter quant table → scores higher on dct_artifacts.
    """
    size = (128, 128)

    # Smooth gradient (no noise at all)
    base = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        base[y, :, 0] = int(180 + y * 0.3)   # R
        base[y, :, 1] = int(120 + y * 0.2)   # G
        base[y, :, 2] = int(200 - y * 0.3)   # B

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    # High quality → flatter quant table (smaller max/min ratio → higher AI score)
    img.save(FIXTURES / "synthetic.jpg", format="JPEG", quality=95)
    print(f"Created {FIXTURES / 'synthetic.jpg'}")


if __name__ == "__main__":
    create_real_jpg()
    create_synthetic_jpg()
    print("Fixtures created successfully.")
