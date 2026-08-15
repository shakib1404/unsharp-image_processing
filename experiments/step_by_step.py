"""
One real image, every single stage of Paper A's default pipeline, saved as its
own PNG -- so each step's visual effect can be inspected individually rather
than only seeing the final before/after.

For every intensity-channel stage, two files are written:
  NN_name_gray.png   -- the raw processed intensity channel (grayscale)
  NN_name_color.png  -- that channel merged back with the ORIGINAL hue/
                         saturation, so it previews as a full endoscopy image
                         at that point in the pipeline
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/shakib_islam/Desktop/image_processing ")
import enhance as E

IMG_PATH = "/home/shakib_islam/Desktop/image_processing /dataset/downloaded/polyp_wikimedia.jpeg"
OUT_DIR = "/home/shakib_islam/Desktop/image_processing /experiments/steps"
WORK_SIZE = 512

GAMMA, CLIP_LIMIT, TILE, CN, SIGMA = 0.8, 2.0, (8, 8), 0.85, 1.0


def save_gray(step, name, chan_uint8):
    cv2.imwrite(os.path.join(OUT_DIR, f"{step:02d}_{name}_gray.png"), chan_uint8)


def save_color(step, name, chan_uint8, ctx):
    bgr = E.merge_intensity(chan_uint8, ctx)
    cv2.imwrite(os.path.join(OUT_DIR, f"{step:02d}_{name}_color.png"), bgr)
    return bgr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (WORK_SIZE, WORK_SIZE))
    cv2.imwrite(os.path.join(OUT_DIR, "00_original.png"), img)

    # Step 1: HSV split
    v_u8, ctx = E.split_intensity(img, "hsv")
    save_gray(1, "hsv_split_V", v_u8)
    save_color(1, "hsv_split", v_u8, ctx)

    # Step 2: normalize (visually identical to step 1 once re-quantized to uint8;
    # saved anyway for completeness of the step list)
    v_norm_u8 = v_u8.copy()
    save_gray(2, "normalize", v_norm_u8)
    save_color(2, "normalize", v_norm_u8, ctx)

    # Step 3: invert
    v_inv = 255 - v_u8
    save_gray(3, "invert", v_inv)
    save_color(3, "invert", v_inv, ctx)

    # Step 4: gamma correction (on inverted, normalized channel)
    v_inv_f = v_inv.astype(np.float32) / 255.0
    v_gamma_f = np.power(np.clip(v_inv_f, 0, 1), GAMMA)
    v_gamma = np.clip(v_gamma_f * 255, 0, 255).astype(np.uint8)
    save_gray(4, "gamma", v_gamma)
    save_color(4, "gamma", v_gamma, ctx)

    # Step 5: CLAHE (paper order: contrast BEFORE Hu-WBI)
    v_clahe = E.apply_contrast(v_gamma, "clahe", clip_limit=CLIP_LIMIT, tile_size=TILE)
    save_gray(5, "clahe", v_clahe)
    save_color(5, "clahe", v_clahe, ctx)

    # Step 6: Hu-WBI 2x upsample (H/S upsampled too, just for this preview merge)
    v_up = E.upsample_2x(v_clahe, "hu_wbi")
    save_gray(6, "huwbi_upsample", v_up)
    ctx_2x = dict(ctx)
    ctx_2x["h"] = cv2.resize(ctx["h"], (v_up.shape[1], v_up.shape[0]), interpolation=cv2.INTER_LINEAR)
    ctx_2x["s"] = cv2.resize(ctx["s"], (v_up.shape[1], v_up.shape[0]), interpolation=cv2.INTER_LINEAR)
    save_color(6, "huwbi_upsample", v_up, ctx_2x)

    # Step 7: downsample back
    v_down = cv2.resize(v_up, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_LINEAR)
    save_gray(7, "downsample", v_down)
    save_color(7, "downsample", v_down, ctx)

    # Step 8: unsharp mask
    lpf = cv2.GaussianBlur(v_down.astype(np.float32), (0, 0), SIGMA)
    ps = (v_down.astype(np.float32) - lpf) * CN
    v_sharp = np.clip(v_down.astype(np.float32) + ps, 0, 255).astype(np.uint8)
    save_gray(8, "unsharp_mask", v_sharp)
    save_color(8, "unsharp_mask", v_sharp, ctx)

    # Step 9: SRGAN
    sr_method = E.resolve_sr_method("srgan_v")
    v_sr = E.apply_sr_intensity(v_sharp, sr_method)
    save_gray(9, "srgan", v_sr)
    save_color(9, "srgan", v_sr, ctx)

    # Step 10: invert back
    v_final = 255 - v_sr
    save_gray(10, "invert_back", v_final)
    enhanced_bgr = save_color(10, "invert_back_merged", v_final, ctx)

    # Step 11: colour transfer (Eq. 6) -- final output
    final_bgr = E.color_transfer(img, enhanced_bgr, strength=0.6)
    cv2.imwrite(os.path.join(OUT_DIR, "11_final_colour_transfer.png"), final_bgr)

    print(f"Wrote {len(os.listdir(OUT_DIR))} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
