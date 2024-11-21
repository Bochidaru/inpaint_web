
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import os
import sys
import yaml
from omegaconf import OmegaConf
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from .crop import recover_size, resize_and_pad, crop_for_filling_post, crop_for_filling_pre

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from .lama.saicinpainting.evaluation.utils import move_to_device
from .lama.saicinpainting.training.trainers import load_checkpoint
from .lama.saicinpainting.evaluation.data import pad_tensor_to_modulo


class ImageProcessor:
    def __init__(self, sam_model_type="vit_h", sam_ckpt="inpaint/pretrained_models/sam_vit_h_4b8939.pth", 
                 lama_config="inpaint/lama/configs/prediction/default.yaml", lama_ckpt="inpaint/pretrained_models/big-lama"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Khởi tạo mô hình SAM
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(self.device)
        self.predictor = SamPredictor(self.sam)
        print()
        print('----- Load SAM done -----')
        print()

        # Khởi tạo mô hình Stable Diffusion
        self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
        ).to(self.device)
        print()
        print('----- Load Stable Diffusion done -----')
        print()

        # Khởi tạo mô hình LaMa
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        predict_config = OmegaConf.load(self.lama_config)
        predict_config.model.path = self.lama_ckpt
        # Load checkpoint cho LaMa
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
        self.lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        if not predict_config.get('refine', False):
            self.lama_model.to(self.device)
        print()
        print('----- Load LaMa done -----')
        print()


    def predict_masks_with_sam(self, img: np.ndarray, point_coords: list[list[float]], point_labels: list[int]):
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def fill_img_with_sd(self, img: np.ndarray, mask: np.ndarray, text_prompt: str):
        img_crop, mask_crop = crop_for_filling_pre(img, mask)
        img_crop_filled = self.sd_pipe(
            prompt=text_prompt,
            image=Image.fromarray(img_crop),
            mask_image=Image.fromarray(mask_crop)
        ).images[0]
        img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
        return img_filled

    def replace_img_with_sd(self, img: np.ndarray, mask: np.ndarray, text_prompt: str, step: int = 50):
        img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
        img_padded = self.sd_pipe(
            prompt=text_prompt,
            image=Image.fromarray(img_padded),
            mask_image=Image.fromarray(255 - mask_padded),
            num_inference_steps=step,
        ).images[0]
        height, width, _ = img.shape
        img_resized, mask_resized = recover_size(
            np.array(img_padded), mask_padded, (height, width), padding_factors)
        mask_resized = np.expand_dims(mask_resized, -1) / 255
        img_resized = img_resized * (1 - mask_resized) + img * mask_resized
        return img_resized

    @torch.no_grad()
    def inpaint_with_lama(self, img: np.ndarray, mask: np.ndarray):
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = {
            'image': img.permute(2, 0, 1).unsqueeze(0),
            'mask': mask[None, None]
        }
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.lama_model(batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        if unpad_to_size is not None:
            cur_res = cur_res[:unpad_to_size[0], :unpad_to_size[1]]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def load_image_and_resized_if_needed(self, image, point_coords_list, max_resolution=1280):
        img = Image.open(image)
        original_width, original_height = img.width, img.height

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if original_width > max_resolution or original_height > max_resolution:
            img.thumbnail((max_resolution, max_resolution), Image.LANCZOS)

        new_width, new_height = img.width, img.height
        width_ratio = new_width / original_width
        height_ratio = new_height / original_height
        resized_coords = [[x * width_ratio, y * height_ratio] for x, y in point_coords_list]

        img = np.array(img)

        return img, resized_coords

    def remove_anything(self, filename, input_img, point_coords, dilate_kernel_size):
        point_labels = [1]
        name, ext = os.path.splitext(filename)

        img, point_coords = self.load_image_and_resized_if_needed(input_img, point_coords_list=point_coords)

        masks, _, _ = self.predict_masks_with_sam(img, point_coords, point_labels)
        masks = masks.astype(np.uint8) * 255

        for i in range(len(masks)):
            masks[i] = cv2.dilate(masks[i], np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8), iterations=1)

        img_inpainted_array_list = []
        name_list = []
        for i in range(len(masks)):
            img_inpainted_array_list.append(self.inpaint_with_lama(img, masks[i]))
            name_list.append(f'{name}_inpainted_{i + 1}{ext}')

        return img_inpainted_array_list, name_list

    def replace_anything(self, filename, input_img, point_coords, text_prompt, mask_index):
        point_labels = [1]
        name, ext = os.path.splitext(filename)

        img, point_coords = self.load_image_and_resized_if_needed(input_img, point_coords_list=point_coords)

        masks, _, _ = self.predict_masks_with_sam(img, point_coords, point_labels)
        masks = masks.astype(np.uint8) * 255
        mask = masks[mask_index]  # select only 1 mask

        img_replaced = self.replace_img_with_sd(img, mask, text_prompt)
        filename_after = f'{name}_inpainted{ext}'
        img_replaced = img_replaced.astype(np.uint8)

        return img_replaced, filename_after

    def fill_anything(self, filename, input_img, point_coords, text_prompt, dilate_kernel_size, mask_index):
        point_labels = [1]
        name, ext = os.path.splitext(filename)

        img, point_coords = self.load_image_and_resized_if_needed(input_img, point_coords_list=point_coords)

        masks, _, _ = self.predict_masks_with_sam(img, point_coords, point_labels)
        masks = masks.astype(np.uint8) * 255
        mask = masks[mask_index]  # select only 1 mask

        mask = cv2.dilate(mask, np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8), iterations=1)

        img_replaced = self.fill_img_with_sd(img, mask, text_prompt)
        filename_after = f'{name}_inpainted{ext}'
        img_replaced = img_replaced.astype(np.uint8)

        return img_replaced, filename_after
