import random
import torch
import math
import warnings
import numbers
from torchvision.transforms import functional as F

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            return F.erase(img, x, y, h, w, v, self.inplace), target
        return img, target


# class ColorJitter(object):
#     """Randomly change the brightness, contrast and saturation of an image.

#     Args:
#         brightness (float or tuple of float (min, max)): How much to jitter brightness.
#             brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
#             or the given [min, max]. Should be non negative numbers.
#         contrast (float or tuple of float (min, max)): How much to jitter contrast.
#             contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
#             or the given [min, max]. Should be non negative numbers.
#         saturation (float or tuple of float (min, max)): How much to jitter saturation.
#             saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
#             or the given [min, max]. Should be non negative numbers.
#         hue (float or tuple of float (min, max)): How much to jitter hue.
#             hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
#             Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
#     """
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = self._check_input(brightness, 'brightness')
#         self.contrast = self._check_input(contrast, 'contrast')
#         self.saturation = self._check_input(saturation, 'saturation')
#         self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
#                                      clip_first_on_zero=False)

#     def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
#         if isinstance(value, numbers.Number):
#             if value < 0:
#                 raise ValueError("If {} is a single number, it must be non negative.".format(name))
#             value = [center - value, center + value]
#             if clip_first_on_zero:
#                 value[0] = max(value[0], 0)
#         elif isinstance(value, (tuple, list)) and len(value) == 2:
#             if not bound[0] <= value[0] <= value[1] <= bound[1]:
#                 raise ValueError("{} values should be between {}".format(name, bound))
#         else:
#             raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

#         # if value is 0 or (1., 1.) for brightness/contrast/saturation
#         # or (0., 0.) for hue, do nothing
#         if value[0] == value[1] == center:
#             value = None
#         return value

#     @staticmethod
#     def get_params(brightness, contrast, saturation, hue):
#         """Get a randomized transform to be applied on image.

#         Arguments are same as that of __init__.

#         Returns:
#             Transform which randomly adjusts brightness, contrast and
#             saturation in a random order.
#         """
#         transforms = []

#         if brightness is not None:
#             brightness_factor = random.uniform(brightness[0], brightness[1])
#             transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

#         if contrast is not None:
#             contrast_factor = random.uniform(contrast[0], contrast[1])
#             transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

#         if saturation is not None:
#             saturation_factor = random.uniform(saturation[0], saturation[1])
#             transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

#         if hue is not None:
#             hue_factor = random.uniform(hue[0], hue[1])
#             transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

#         random.shuffle(transforms)
#         transform = Compose(transforms)

#         return transform

#     def __call__(self, img, target):
#         """
#         Args:
#             img (PIL Image): Input image.

#         Returns:
#             PIL Image: Color jittered image.
#         """
#         transform = self.get_params(self.brightness, self.contrast,
#                                     self.saturation, self.hue)
#         img = transform(img)
#         return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
