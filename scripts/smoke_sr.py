"""Smoke test for utils.sr super-resolution module (zero-dep path)."""
import numpy as np

from utils.sr import create_sr_engine


def main() -> None:
    # none/""/None -> None (SR disabled)
    assert create_sr_engine("none") is None
    assert create_sr_engine("") is None
    assert create_sr_engine(None) is None

    # bicubic: zero-dep baseline
    e = create_sr_engine("bicubic", scale=2)
    img = (np.random.rand(20, 60, 3) * 255).astype("uint8")
    out = e.enhance(img)
    assert out.shape[0] == 40 and out.shape[1] == 120, out.shape
    assert out.dtype == np.uint8
    assert img.shape == (20, 60, 3), "input must not be mutated"
    print(f"bicubic: {img.shape} -> {out.shape} dtype={out.dtype}")

    # guard empty/None input
    assert e.enhance(None) is None
    empty = np.empty((0, 0, 3), dtype="uint8")
    assert e.enhance(empty).size == 0

    # unknown engine raises ValueError
    try:
        create_sr_engine("bogus")
        raise AssertionError("expected ValueError for unknown engine")
    except ValueError:
        print("unknown engine -> ValueError: ok")

    # lcofl stub raises NotImplementedError on construction
    try:
        create_sr_engine("lcofl")
        raise AssertionError("expected NotImplementedError for lcofl stub")
    except NotImplementedError:
        print("lcofl stub -> NotImplementedError: ok")

    print("SR module OK")


if __name__ == "__main__":
    main()
