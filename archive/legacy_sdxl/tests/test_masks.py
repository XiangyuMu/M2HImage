import numpy as np
from PIL import Image

from mcic.preprocess.parsing import compose_training_masks


def test_cloth_safe_excludes_identity_region():
    arrays = {}
    for key in ("face", "hair", "cloth", "person"):
        arrays[key] = np.zeros((64, 64), dtype=np.uint8)
    arrays["face"][8:18, 25:40] = 255
    arrays["hair"][5:12, 23:42] = 255
    arrays["cloth"][15:55, 10:54] = 255
    arrays["person"][5:60, 8:56] = 255
    masks = {key: Image.fromarray(value) for key, value in arrays.items()}
    output = compose_training_masks(
        masks, {"preprocess": {"boundary_dilate_radius": 3, "cloth_erode_radius": 1}}
    )
    identity = np.asarray(output["cf_mask"]) > 0
    cloth = np.asarray(output["cloth_safe_mask"]) > 0
    assert not np.logical_and(identity, cloth).any()
