import argparse, gc
# only import torch if you actually use it; otherwise omit
try:
    import torch
except ImportError:
    torch = None
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--medium-ram', action='store_true',
                   help='reduce memory usage by downsizing and single-threading')
    return p.parse_args()

def maybe_configure_medium_ram():
    # limit threads at runtime (some libraries honor these)
    cv2.setNumThreads(1)
    if torch:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

def load_and_downscale(image_path, max_dim=800):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"couldn’t load {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_size = (int(w*scale), int(h*scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def main():
    args = parse_args()
    if args.medium_ram:
        maybe_configure_medium_ram()

    # … your setup …

    for fname in image_files:
        # load + optional downscale
        image = load_and_downscale(os.path.join(screenshots_dir, fname)) \
                if args.medium_ram else cv2.imread(os.path.join(screenshots_dir, fname))

        # convert/crop/detect as before…
        face = detect_and_crop_face_retinaface(image, margin=0.5)  # pass already-loaded image if you refactor func

        # after each major step, free memory
        del image
        gc.collect()
        if torch:
            torch.cuda.empty_cache()

        # … subprocess calls, moving files, etc. …

    # end loop

if __name__ == '__main__':
    main()
