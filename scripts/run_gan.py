import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pyprojroot
import matplotlib
import shutil
import cv2
import os
from CloudGAN.CloudGAN import remove_clouds
from src.preprocessing.subtile_esd_hw02 import grid_slice
from src.preprocessing.file_utils import load_satellite
import argparse
root = pyprojroot.here()

def create_parser(args=None):
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--weights_GAN", type=str,
                        help="Weights of SN-PatchGAN model")
    parser.add_argument("--config_GAN", type=str,
                        help="Config file (.yml) of SN-PatchGAN")

    parser.add_argument("--weights_AE", type=str,
                        help="Weights of AE model")
    parser.add_argument("--config_AE", type=str,
                        help="Config file (.yml) of AE")

    parser.add_argument("--img", type=str,
                        help="Location of image to be processed")
    parser.add_argument("--no_AE", action="store_true",
                        help="Do not use AE but use given mask (via argument '--mask')")
    parser.add_argument("--target", type=str, default=None,
                        help="Ground truth for input image")
    parser.add_argument("--output", type=str,
                        help="Filename of output")
    parser.add_argument("--mask", type=str,
                        help="Filename of mask (default does not save mask)")
   
    return parser.parse_args(args)

def remove_all_clouds(root_mask_dir, image_dir, save_dir):
    for mask_dir in Path(root_mask_dir).iterdir():
        for mask_file in mask_dir.iterdir():
            if mask_file.suffix == '.jpg':
                mask_name = mask_file.name
                tile_dir = Path(mask_dir.name)
                img_name = Path(mask_name)
                img = image_dir / tile_dir / img_name
                if not os.path.exists(save_dir / tile_dir) and not os.path.isdir(save_dir / tile_dir): 
                    (save_dir / tile_dir).mkdir(parents=True, exist_ok=True)

                simulated_args = [
                    "--img", str(image_dir / tile_dir / img_name),
                    "--mask", str(root_mask_dir / tile_dir / img_name),
                    "--no_AE",
                    "--output", str(save_dir / tile_dir / img_name),
                    "--weights_GAN", "./models/SN_PatchGAN",
                    "--config_GAN", "./config/cloud_removal_config.yml",
                    "--weights_AE", "./models/AE-CloudGAN/aecheckpoint.h5",
                    "--config_AE", "./config/cloud_detection_config.yml"
                ]

                curr_img = cv2.imread(str(img))
                args = create_parser(simulated_args)
                remove_clouds(curr_img, args)
                new_img = cv2.imread(str(save_dir / tile_dir / img_name))
                resized_new_img = cv2.resize(new_img, (800, 800), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(save_dir / tile_dir / img_name), resized_new_img)

def copy_missing_files_and_directories(src_dir, dest_dir):
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    for src_path in src_dir.rglob('*'):
        relative_path = src_path.relative_to(src_dir)
        dest_path = dest_dir / relative_path

        if src_path.is_dir() and not dest_path.exists():
            shutil.copytree(src_path, dest_path)
        elif src_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)

def npy_to_img(src_dir, out_dir, size):
    for time_ax in range(0, 4):
        for np_file in Path(src_dir).iterdir():
            image = np.load(np_file)
            image = image[time_ax, [3,2,1], :, :]
            image = np.transpose(image, (1, 2, 0))
            if size != 800: 
                image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

            curr_dir = Path(out_dir / f"{np_file.stem}")
            if not os.path.exists(curr_dir) and not os.path.isdir(curr_dir): 
                curr_dir.mkdir(parents=True, exist_ok=True)
            plt.imsave(curr_dir / f"{time_ax}.jpg", image, vmin=-0.5, vmax=3.5)

def combine_gan_imgs(original_np_dir, gan_imgs_dir, out_dir):
    for tile_dir in Path(gan_imgs_dir).iterdir():
        images = []
        for img_path in tile_dir.iterdir():
            img = cv2.imread(str(img_path))
            images.append(img)
        stitched_images = np.stack(images, axis=0)   
        stitched_images = stitched_images[:, :, :, ::-1]    
        stitched_images = stitched_images.transpose(0, 3, 1, 2)
        for tile in Path(original_np_dir).iterdir():
            tile_name = tile.name.split('.')[0]
            if tile_dir.name == tile_name:
                img_stack = np.load(tile)
                img_stack[:, 1:4, :, :] = stitched_images
                break

        np.save(out_dir / Path(tile_dir.name), img_stack)


def clear_directory(source_dir):
    for dir in Path(source_dir).iterdir():
        shutil.rmtree(dir)

if __name__ == '__main__':
    root_mask_dir = root / "data" / "masks"
    restitched_image_dir = root / "data" / "restitched" /"restitched_imgs"
    gan_imgs_dir = root / "data" / "restitched" /"gan_imgs"
    restitched_np_dir = root / "data" / "restitched" /"restitched_np"
    restitched_gan_np_dir = root / "data" / "restitched" /"restitched_gan_np"
    train_metadata_dir = root / "data" / "processed"/ "4x4" /"Train" / "metadata"
    val_metadata_dir = root / "data" / "processed"/ "4x4" /"Val" / "metadata"
    preprocessed_dir = root / "data" / "preprocessed"
    npy_to_img(restitched_np_dir, restitched_image_dir, 256)
    remove_all_clouds(root_mask_dir, restitched_image_dir, gan_imgs_dir)
    clear_directory(restitched_image_dir)
    npy_to_img(restitched_np_dir, restitched_image_dir, 800)
    copy_missing_files_and_directories(restitched_image_dir, gan_imgs_dir)
    combine_gan_imgs(restitched_np_dir, gan_imgs_dir, restitched_gan_np_dir)


