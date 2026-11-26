import logging
from typing import Optional, Sequence

from torchvision import transforms

from dinov2.data.transforms import GaussianBlur, make_normalize_transform

logger = logging.getLogger("explora")


class DataAugmentationDEXPloRA(object):
    """Build DINO-style pathology augmentations with two global crops of 224 pixels and four local crops of 98 pixels. Use ImageNet normalization."""

    def __init__(
        self,
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        global_crops_size=224,
        local_crops_size=98,
        local_crops_number=4,
        dataset_mean: Optional[Sequence[float]] = (0.485, 0.456, 0.406),
        dataset_std: Optional[Sequence[float]] = (0.229, 0.224, 0.225),
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using pathology D-ExPLoRA augmentation:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose([GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2)])
        local_transfo_extra = GaussianBlur(p=0.5)

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(dataset_mean, dataset_std),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    @staticmethod
    def _check_mode(img, step_name):
        """Check at runtime whether a PIL image has unexpectedly left RGB mode."""
        from PIL import Image as _PILImage
        if isinstance(img, _PILImage.Image) and img.mode != "RGB":
            logger.error(
                f"[MODE CHANGED] after '{step_name}': mode={img.mode}, "
                f"size={img.size}, bands={img.getbands()}"
            )
            raise RuntimeError(
                f"Image mode became '{img.mode}' after transform '{step_name}'. "
                f"Expected 'RGB'. size={img.size}"
            )

    def __call__(self, image):
        output = {}

        self._check_mode(image, "input")

        # global crops
        im1_base = self.geometric_augmentation_global(image)
        self._check_mode(im1_base, "geometric_augmentation_global[1]")

        global_crop_1 = self._apply_transfo_with_check(self.global_transfo1, im1_base, "global_transfo1")

        im2_base = self.geometric_augmentation_global(image)
        self._check_mode(im2_base, "geometric_augmentation_global[2]")

        global_crop_2 = self._apply_transfo_with_check(self.global_transfo2, im2_base, "global_transfo2")

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops
        local_crops = []
        for i in range(self.local_crops_number):
            lc_base = self.geometric_augmentation_local(image)
            self._check_mode(lc_base, f"geometric_augmentation_local[{i}]")
            lc = self._apply_transfo_with_check(self.local_transfo, lc_base, f"local_transfo[{i}]")
            local_crops.append(lc)
        output["local_crops"] = local_crops
        output["offsets"] = ()
        return output

    def _apply_transfo_with_check(self, transfo, img, transfo_name):
        """Check the image mode after each transform to locate the operation that changed it."""
        from PIL import Image as _PILImage
        x = img
        for i, t in enumerate(transfo.transforms):
            x = t(x)
            if isinstance(x, _PILImage.Image):
                if x.mode != "RGB":
                    logger.error(
                        f"[MODE CHANGED] '{transfo_name}' step {i} "
                        f"({t.__class__.__name__}): mode={x.mode}"
                    )
                    raise RuntimeError(
                        f"Image mode became '{x.mode}' after "
                        f"'{transfo_name}' step {i} ({t.__class__.__name__}). "
                        f"Expected 'RGB'."
                    )
        return x
