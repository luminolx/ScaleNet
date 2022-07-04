
import numbers
import random, math
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
import copy

def adjust_brightness(img, brightness_factor):
    table = np.array([i*brightness_factor for i in range(0, 256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_contrast(img, contrast_factor):
    table = np.array([(i - 74) * contrast_factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')

    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_saturation(img, saturation_factor):

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def adjust_hue(img, hue_factor):
    # After testing, found that OpenCV calculates the Hue in a call to
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _get_affine_matrix(center, angle=(0, 0), translations=(0, 0), zoom=(1, 1), shear=None, mirror=None, corr_list=[]):
    matrix = np.eye(3).astype(np.float64)

    # rotation
    M = cv2.getRotationMatrix2D(center, angle, 1)
    matrix[:2, :] = M.astype(np.float64)

    # translate
    matrix[0, 2] += translations[0] * 1.0
    matrix[1, 2] += translations[1] * 1.0
    # zoom
    matrix[:, :2] *= zoom
    matrix[0, 2] += (1.0 - zoom) * center[0]
    matrix[1, 2] += (1.0 - zoom) * center[1]

    mirror_flag = False
    if mirror is not None:
        mirror_rng = random.uniform(0., 1.)
        if mirror_rng < mirror['mirror_prob']:
            mirror_flag = True
            matrix[0, 0] = -matrix[0, 0]
            matrix[0, 1] = -matrix[0, 1]
            matrix[0, 2] = (center[0] + 0.5) * 2.0 - matrix[0, 2]

    return matrix[:2, :], mirror_flag


def _get_corr_list(num_pts=106):
    if num_pts == 16: # for smoke mirror
        corr_list = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
    elif num_pts == 68:
        corr_list = [0, 16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 17, 26, 18, 25, 19, 24, 20, 23, 21, 22,
                     36,
                     45, 37, 44, 38, 43, 39, 42, 41, 46, 40, 47, 31, 35, 32, 34, 48, 54, 49, 53, 50, 52, 60, 64, 61,
                     63,
                     67, 65, 59, 55, 58, 56]
    elif num_pts == 106: # for face landmark mirror
        corr_list = [0, 32, 1, 31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, 7, 25, 8, 24, 9, 23, 10, 22, 11, 21, 12, 20,
                     13,
                     19,
                     14, 18, 15, 17, 33, 42, 34, 41, 35, 40, 36, 39, 37, 38, 64, 71, 65, 70, 66, 69, 67, 68, 52, 61,
                     53,
                     60,
                     72, 75, 54, 59, 55, 58, 56, 63, 73, 76, 57, 62, 74, 77, 104, 105, 78, 79, 80, 81, 82, 83, 47,
                     51,
                     48, 50,
                     84, 90, 96, 100, 85, 89, 86, 88, 95, 91, 94, 92, 97, 99, 103, 101]
    else:
        raise ValueError('landmarks only support 16/68/106')
    corr_list = np.array(corr_list, dtype=np.uint8).reshape(-1, 2)
    return corr_list

class NewSize:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        assert 'output_size' in self.kwargs
        self.output_size = self.kwargs['output_size']
        self.change_ldm = self.kwargs.get('change_ldm', True)
        assert isinstance(self.output_size, (int, list, tuple))
        if isinstance(self.output_size, (list, tuple)):
            assert len(self.output_size) == 2
        # assert len(list(self.kwargs.keys())) == 1

    def __call__(self, **inputs):
        image = inputs['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        self.size = (new_w, new_h)
        if not (w <= h and w == self.output_size) and not (h <= w and h == self.output_size):
            image = cv2.resize(image, (new_w, new_h))
        if self.change_ldm:
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
            if 'landmarks' in inputs:
                landmarks = inputs['landmarks']
                if landmarks is not None:
                    for key, landmark in landmarks.items():
                        points = np.resize(np.array(landmark), (int(len(landmark) / 2), 2))
                        points = points * [new_w / w, new_h / h]
                        for i in range(int(len(landmark) / 2)):
                            x, y = points[i, :]
                            landmark[2 * i] = x
                            landmark[2 * i + 1] = y
                        landmarks.update({key: landmark})
                    inputs.update({'landmarks': landmarks})
            if 'rect' in inputs:
                rect = inputs['rect']
                for i in range(int(len(rect) // 2)):
                    rect[2 * i] = rect[2 * i] / (w / new_w)
                    rect[2 * i + 1] = rect[2 * i + 1] / (h / new_h)
                inputs.update({'rect': rect})
        inputs.update({'image': image})
        return inputs

class Resize:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        assert 'output_size' in self.kwargs
        self.output_size = self.kwargs['output_size']
        assert isinstance(self.output_size, (int, list, tuple))
        if isinstance(self.output_size, (list, tuple)):
            assert len(self.output_size) == 2
        assert len(list(self.kwargs.keys())) == 1

    def __call__(self, **inputs):
        image = inputs['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        self.size = (new_w, new_h)
        if not (w <= h and w == self.output_size) and not (h <= w and h == self.output_size):
            image = cv2.resize(image, (new_w, new_h))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                for key, landmark in landmarks.items():
                    points = np.resize(np.array(landmark), (int(len(landmark) / 2), 2))
                    points = points * [new_w / w, new_h / h]
                    for i in range(int(len(landmark) / 2)):
                        x, y = points[i, :]
                        landmark[2 * i] = x
                        landmark[2 * i + 1] = y
                    landmarks.update({key: landmark})
                inputs.update({'landmarks': landmarks})
        if 'rect' in inputs:
            rect = inputs['rect']
            for i in range(int(len(rect) // 2)):
                rect[2 * i] = rect[2 * i] / (w / new_w)
                rect[2 * i + 1] = rect[2 * i + 1] / (h / new_h)
            inputs.update({'rect': rect})
        inputs.update({'image': image})
        return inputs


class RandomResizedCrop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        assert 'output_size' in self.kwargs
        self.output_size = self.kwargs['output_size']
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        elif isinstance(self.output_size, (list, tuple)):
            assert len(self.output_size) == 2
        else:
            raise ValueError('output size must be int or tuple or list')

        assert 'scale' in self.kwargs
        self.scale = self.kwargs['scale']
        assert isinstance(self.scale, (list, tuple)) and len(self.scale) == 2

        assert 'ratio' in self.kwargs
        self.ratio = self.kwargs['ratio']
        assert isinstance(self.ratio, (list, tuple)) and len(self.ratio) == 2

        n_p = len(list(self.kwargs.keys()))
        if 'interpolation' not in self.kwargs:
            assert n_p == 3
            self.interpolation = cv2.INTER_LINEAR
        else:
            assert n_p == 4
            self.interpolation = self.kwargs['interpolation']

    @staticmethod
    def get_params(img, scale, ratio):

        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if h <= img.shape[0] and w <= img.shape[1]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, **inputs):
        image = inputs['image']
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = image[i:i + h, j:j + w, ...]
        image = cv2.resize(image, self.output_size, self.interpolation)
        inputs.update({'image': image})
        return inputs

class CenterCrop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        assert 'output_size' in self.kwargs
        self.output_size = self.kwargs['output_size']
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        elif isinstance(self.output_size, (list, tuple)):
            assert len(self.output_size) == 2
        else:
            raise ValueError('output size must be int or tuple or list')
        n_p = len(list(self.kwargs.keys()))
        if 'offset' not in self.kwargs:
            assert n_p == 1
            self.offset = 0
        else:
            assert n_p == 2
            self.offset = self.kwargs['offset']
            assert isinstance(self.offset, int)

    def __call__(self, **inputs):
        image = inputs['image']
        image = np.array(image)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        top += self.offset
        assert new_h + top <= h and new_w <= w
        if len(image.shape) == 3:
            image = image[top: top + new_h, left: left + new_w, :]
        else:
            image = image[top: top + new_h, left: left + new_w]
        # image = image.crop(bbox)
        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                for key, landmark in landmarks.items():
                    points = np.resize(np.array(landmark), (int(len(landmark) / 2), 2))
                    points = points - [left, top]
                    for i in range(int(len(landmark) / 2)):
                        x, y = points[i, :]
                        landmark[2 * i] = x
                        landmark[2 * i + 1] = y
                    landmarks.update({key: landmark})
                inputs.update({'landmarks': landmarks})
        inputs.update({'image': image})
        return inputs

class RandomCrop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        n_p = len(list(self.kwargs.keys()))
        assert n_p == 1
        assert 'output_size' in self.kwargs
        self.output_size = self.kwargs['output_size']
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        elif isinstance(self.output_size, (list, tuple)):
            assert len(self.output_size) == 2
        else:
            raise ValueError('output size must be int or tuple or list')

    def __call__(self, **inputs):
        image = inputs['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        if len(image.shape) == 3:
            image = image[top: top + new_h,
                    left: left + new_w, :]
        else:
            image = image[top: top + new_h,
                    left: left + new_w]
        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                for key, landmark in landmarks.items():
                    points = np.resize(np.array(landmark), (int(len(landmark) / 2), 2))
                    points = points - [left, top]
                    for i in range(int(len(landmark) / 2)):
                        x, y = points[i, :]
                        landmark[2 * i] = x
                        landmark[2 * i + 1] = y
                    landmarks.update({key: landmark})
                inputs.update({'landmarks': landmarks})
        inputs.update({'image': image})
        return inputs


class Pad:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        assert 'padding' in self.kwargs
        self.padding = self.kwargs['padding']
        assert isinstance(self.padding, (int, tuple, list))
        # order: up down left right
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding, self.padding, self.padding)
        else:
            assert len(self.padding) == 2 or len(self.padding) == 4
            if len(self.padding) == 2:
                self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        n_p = len(list(self.kwargs.keys()))
        if 'fill' in self.kwargs:
            assert n_p == 2
            self.fill = self.kwargs['fill']
        else:
            assert n_p == 1
            self.fill = 0

    def __call__(self, **inputs):
        image = inputs['image']
        image_size = image.shape[:2]
        new_shape = (
            image_size[0] + self.padding[0] + self.padding[1], image_size[1] + self.padding[2] + self.padding[3])
        new_img = np.full((new_shape[0], new_shape[1], image.shape[2]), self.fill, dtype=np.uint8)
        new_img[self.padding[0]:self.padding[0] + image_size[0], self.padding[2]:self.padding[2] + image_size[1], ...] = image
        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                landmarks = np.array(landmarks)
                for i in range(int(len(landmarks) / 2)):
                    landmarks[i * 2] = landmarks[i * 2] + self.padding[2]
                    landmarks[i * 2 + 1] = landmarks[i * 2 + 1] + self.padding[0]
                inputs.update({'landmarks': landmarks})
        inputs.update({'image': new_img})
        return inputs

class Occlusion(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        self.cente_min = self.kwargs['center_min']
        self.center_max = self.kwargs['center_max']
        self.width_min = self.kwargs['width_min']
        self.width_max = self.kwargs['width_max']
        self.height_min = self.kwargs['height_min']
        self.height_max = self.kwargs['height_max']
        self.color_min = self.kwargs['color_min']
        self.color_max = self.kwargs['color_max']

    @staticmethod
    def get_params(center_min, center_max, width_min, width_max,height_min,height_max,color_min,color_max,final_channel = 3):
        """Get parameters for occluison"""
        center_x = random.uniform(center_min,center_max)
        center_y = random.uniform(center_min,center_max)
        occ_center = (np.int(center_x), np.int(center_y))

        width = random.uniform(width_min,width_max)
        height = random.uniform(height_min,height_max)
        outsize = (np.int(width),np.int(height))

        if final_channel == 3:
            channel1 = np.uint8(random.uniform(color_min, color_max))
            channel2 = np.uint8(random.uniform(color_min, color_max))
            channel3 = np.uint8(random.uniform(color_min, color_max))
            occ_color = (channel1, channel2, channel3)
        elif final_channel == 1:
            occ_color = np.uint8(random.uniform(color_min, color_max))
        else:
            raise NotImplementedError

        return occ_center,outsize,occ_color

    def __call__(self, **inputs):
        img = inputs['image']
        image_size = img.shape
        channel = img.shape[2]
        occ_center, outsize, occ_color = self.get_params(self.cente_min,self.center_max,self.width_min,self.width_max,self.height_min,self.height_max,
                                                         self.color_min,self.color_max,channel)
        #center x,y outsize:width,height
        occ_left = max(occ_center[0] - outsize[0] // 2, 0)
        occ_right = min(occ_center[0] + outsize[0] // 2, image_size[1])
        occ_top = max(occ_center[1] - outsize[1] // 2, 0)
        occ_bottom = min(occ_center[1] + outsize[1] // 2, image_size[0])

        img[occ_top:occ_bottom, occ_left:occ_right, :] = occ_color
        landmark_dict = inputs['landmarks']
        assert 'occlusion' in landmark_dict.keys() and 'occlusion' in inputs.keys()
        occlusion_label = np.array(inputs['occlusion'])
        assert len(occlusion_label) == 106

        for i in range(len(occlusion_label)):
            if (landmark_dict['occlusion'][2 * i] >= occ_left and landmark_dict['occlusion'][2 * i] < occ_right) and (
                    landmark_dict['occlusion'][2 * i + 1] >= occ_top and landmark_dict['occlusion'][2 * i + 1] < occ_bottom):
                occlusion_label[i] = 0

            if landmark_dict['occlusion'][2 * i] < 0 or landmark_dict['occlusion'][2 * i] > image_size[1] or landmark_dict['occlusion'][2 * i + 1] < 0 or landmark_dict['occlusion'][2 * i + 1] > image_size[0]:
                occlusion_label[i] = 0

        inputs.update({'image':img,'occlusion':occlusion_label,'landmarks':landmark_dict})

        return inputs

class Affine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self._check_params()

    def _check_params(self):
        self.num_pts = self.kwargs.get('num_pts', 106)
        self.corr_list = _get_corr_list(self.num_pts)
        degrees = (0, 0)
        if 'rotate' in self.kwargs:
            rotate_parma = self.kwargs['rotate']
            degrees = (-rotate_parma['rotate_angle_max'], rotate_parma['rotate_angle_max'])
        self.degrees = degrees
        trans = (0, 0)
        if 'translate' in self.kwargs:
            trans = (-self.kwargs['translate']['trans_value_max'], self.kwargs['translate']['trans_value_max'])
        self.trans = trans
        zoom = (1, 1)
        if 'zoom' in self.kwargs:
            zoom = (self.kwargs['zoom']['zoom_min'], self.kwargs['zoom']['zoom_max'])
        self.zoom = zoom
        self.mirror = self.kwargs['mirror']

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        if degrees is not None:
            angle = random.uniform(degrees[0], degrees[1])
        else:
            angle = (0, 0)
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear


    def __call__(self, **inputs):
        image = inputs['image']
        h, w = image.shape[:2]
        center = (w / 2. - 0.5, h / 2 - 0.5)
        angle, translations, zoom_v, shear = self.get_params(self.degrees, self.trans, self.zoom, None, [h, w])
        matrix, mirrored = _get_affine_matrix(center, angle, translations, zoom_v, None, self.mirror)
        src = np.array(image).astype(np.uint8)
        same_size = (src.shape[1], src.shape[0])
        image = cv2.warpAffine(src, matrix, same_size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                               borderValue=(127, 127, 127))
        # landmark transformation
        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                corr_list = {}
                for key, value in landmarks.items():
                    num_pt = len(value) // 2
                    corr_list[key] = _get_corr_list(num_pt)
                # landmarks tranformation
                for key, landmark in landmarks.items():
                    matrix = np.resize(matrix, (6,))
                    points = np.resize(np.array(landmark).copy(), (int(len(landmark) / 2), 2))
                    # print('we have %d landmarks.' % points.shape[0])
                    for i in range(int(len(landmark) / 2)):
                        x, y = points[i, :]
                        x_new = matrix[0] * x + matrix[1] * y + matrix[2]
                        y_new = matrix[3] * x + matrix[4] * y + matrix[5]

                        landmark[2 * i] = x_new
                        landmark[2 * i + 1] = y_new

                    if mirrored:
                        # TODO(bbox mirror)
                        if len(landmark) == 4:
                            temp = landmark[0]
                            landmark[0] = landmark[2]
                            landmark[2] = temp
                        else:
                            cur_corr_list = corr_list[key]
                            for k in range(cur_corr_list.shape[0]):
                                temp_x = landmark[2 * cur_corr_list[k, 0]]
                                temp_y = landmark[2 * cur_corr_list[k, 0] + 1]
                                landmark[2 * cur_corr_list[k, 0]], landmark[2 * cur_corr_list[k, 0] + 1] = \
                                    landmark[2 * cur_corr_list[k, 1]], landmark[2 * cur_corr_list[k, 1] + 1]
                                landmark[2 * cur_corr_list[k, 1]], landmark[2 * cur_corr_list[k, 1] + 1] = temp_x, temp_y

                            if key == 'occlusion':
                                for k in range(cur_corr_list.shape[0]):
                                    occlusion_label = input['occlusion']
                                    temp_x = occlusion_label[cur_corr_list[k, 0]]
                                    temp_y = occlusion_label[cur_corr_list[k, 0] + 1]
                                    occlusion_label[cur_corr_list[k, 0]], occlusion_label[cur_corr_list[k, 0] + 1] = occlusion_label[cur_corr_list[k, 1]], occlusion_label[cur_corr_list[k, 1] + 1]
                                    occlusion_label[cur_corr_list[k, 1]], occlusion_label[cur_corr_list[k, 1] + 1] = temp_x, temp_y
                                inputs.update({'occlusion':occlusion_label})
                    landmarks[key] = landmark
                inputs.update({'landmarks': landmarks})
        # translate rects wtih the same setting as original image

        if 'rect' in inputs:
            rect = inputs['rect']
            if mirrored:
                rect_new = copy.deepcopy(rect)
                for i in range(int(len(rect) // 2)):
                    rect_new[2 * i] = w - rect[2 * i]
                # left right eye change, corner changed
                rect[0] = rect_new[6]
                rect[1] = rect_new[5]
                rect[2] = rect_new[4]
                rect[3] = rect_new[7]
                rect[4] = rect_new[2]
                rect[5] = rect_new[1]
                rect[6] = rect_new[0]
                rect[7] = rect_new[3]
                inputs.update({'rect': rect})
                if 'cls' in inputs:
                    cls = inputs['cls']
                    if cls == 1:
                        cls = 2
                    elif cls == 2:
                        cls = 1
                    inputs.update({'cls': cls})
                if 'gaze' in inputs:
                    inputs.update({'gaze': -inputs['gaze']})
            if translations != (0, 0):
                trans_x, trans_y = translations
                for i in range(int(len(rect) // 2)):
                    rect[2 * i] += trans_x
                    if rect[2 * i] < 0.:
                        rect[2 * i] = 0.0
                    elif rect[2 * i] > h:
                        rect[2 * i] = h
                    rect[2 * i + 1] += trans_y
                    if rect[2 * i + 1] < 0.:
                        rect[2 * i + 1] = 0.0
                    elif rect[2 * i + 1] > w:
                        rect[2 * i + 1] = w
                inputs.update({'rect': rect})
        inputs.update({'image': image})
        return inputs

class Gray:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        n_p = len(list(self.kwargs.keys()))
        assert n_p == 2
        self.num_output_channels = self.kwargs['num_output_channels']
        self.gray_prob = self.kwargs['gray_prob']
    def __call__(self, **inputs):
        image = inputs['image']
        if random.random() < self.gray_prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            if self.num_output_channels == 3:
                image = np.dstack([image, image, image])
        inputs.update({'image': image})
        return inputs


class ColorJitter(object):

    def __init__(self,  **kwargs):
        n_p = len(list(kwargs.keys()))
        assert n_p == 4
        brightness, contrast, saturation, hue = kwargs['brightness'], kwargs['contrast'], kwargs['saturation'], kwargs['hue']
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        _transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            _transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            _transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            _transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            _transforms.append(lambda img: adjust_hue(img, hue_factor))

        random.shuffle(_transforms)
        transform = transforms.Compose(_transforms)

        return transform

    def __call__(self, **inputs):
        image = inputs['image']
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        image = transform(image)
        inputs.update({'image': image})
        return inputs


class MotionBlur:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        n_p = len(list(self.kwargs.keys()))
        assert n_p <= 5
        self.l_min = self.kwargs.get('l_min', 0)
        self.l_max = self.kwargs.get('l_max', 0)
        self.theta_min = self.kwargs.get('theta_min', 0)
        self.theta_max = self.kwargs.get('theta_max', 0)
        self.p = self.kwargs['p']

    @staticmethod
    def get_params(l_min, l_max, theta_min, theta_max):
        l = int(np.round(random.uniform(l_min, l_max)))
        theta = random.uniform(theta_min, theta_max)
        return l, theta

    def __call__(self, **inputs):
        if random.random() < self.p:
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            L, theta = self.get_params(self.l_min, self.l_max, self.theta_min, self.theta_max)
            theta_radian = theta * np.pi / 180
            sintheta = np.sin(theta_radian)
            costheta = np.cos(theta_radian)
            kernel = np.zeros((int(abs(sintheta * L / 2)) + 1, int(abs(costheta * L / 2)) + 1))
            cv2.line(kernel, (0, 0), (int(abs(costheta * L / 2)), int(abs(sintheta * L / 2))), 1)
            if sintheta < 0:
                kernel = cv2.flip(kernel, 0)
            if costheta < 0:
                kernel = cv2.flip(kernel, 1)
            M = cv2.getRotationMatrix2D((int(abs(costheta * L / 2)), int(abs(sintheta * L / 2))), theta, 1)
            kernel = cv2.warpAffine(kernel, M, (L, L))
            kernel = kernel / L
            image = inputs['image']
            blurred_image = cv2.filter2D(image, -1, kernel)
            cv2.normalize(blurred_image, blurred_image, 0, 255, cv2.NORM_MINMAX)
            image = blurred_image
            inputs.update({'image': image})
        return inputs


class AddNoise:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()
    def _check_params(self):
        n_p = len(list(self.kwargs.keys()))
        assert n_p == 6
        self.x_off = self.kwargs['x_off']
        self.y_off = self.kwargs['y_off']
        self.mean = self.kwargs['mean']
        self.std = self.kwargs['std']
        self.noise_tasks = self.kwargs['tasks']
        self.weight = self.kwargs['weight']
        assert isinstance(self.noise_tasks,list) or isinstance(self.noise_tasks,list)
        assert len(self.x_off) == len(self.y_off) == len(self.mean) == len(self.std) == len(self.noise_tasks) == len(self.weight)
    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        if degrees is not None:
            angle = random.uniform(degrees[0], degrees[1])
        else:
            angle = (0, 0)
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0
        return angle, translations, scale, shear

    def __call__(self,  **inputs):  # numpy image(w,h,c)
        assert 'task' in inputs
        self.tasks = inputs['task']
        if not set(list(self.noise_tasks)).isdisjoint(set(list(self.tasks))):
            for subtask in self.noise_tasks:
                if subtask in self.tasks:
                    idx = self.noise_tasks.index(subtask)
                    image = inputs['image']
                    final_h, final_w = image.shape[:2]
                    final_center = (final_h / 2. - 0.5, final_w / 2 - 0.5)
                    angle, translations, zoom, shear = self.get_params((0, 0),
                                                                       (self.x_off[idx], self.y_off[idx]),
                                                                       (1, 1), (0, 0), [1, 1])
                    matrix, mirrored = _get_affine_matrix(final_center, angle, translations, zoom, shear, None, [])
                    src_noise = np.array(image).astype(np.uint8)
                    same_size = (src_noise.shape[1], src_noise.shape[0])
                    image_noise = cv2.warpAffine(src_noise, matrix, same_size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                                 borderValue=(127, 127, 127))
                    noise = np.random.normal(size=image_noise.shape, loc=self.mean[idx], scale=self.std[idx])
                    image_noise = image_noise + noise
                    inputs.update({'image_noise': image_noise})
                    break
        else:
            return inputs

        if 'landmarks' in inputs:
            landmarks = inputs['landmarks']
            if landmarks is not None:
                for key, value in landmarks.items():
                    if not key in self.noise_tasks:
                        continue
                    # landmarks tranformation
                    landmarks_noise = value.copy()
                    matrix = np.resize(matrix, (6,))
                    points = np.resize(np.array(value).copy(), (int(len(value) / 2), 2))
                    for i in range(int(len(value) / 2)):
                        x, y = points[i, :]
                        x_new = matrix[0] * x + matrix[1] * y + matrix[2]
                        y_new = matrix[3] * x + matrix[4] * y + matrix[5]

                        landmarks_noise[2 * i] = x_new
                        landmarks_noise[2 * i + 1] = y_new
                    value = np.concatenate((value,landmarks_noise), axis=0)
                    landmarks[key] = value
                inputs.update({'landmarks': landmarks})

        return inputs


class Normalize:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._check_params()

    def _check_params(self):
        self.normalize_type = self.kwargs.get('normalize_type', 'mean_std')
        if self.normalize_type == 'mean_std':
            assert 'mean' in self.kwargs and 'std' in self.kwargs
            self.mean = self.kwargs['mean']
            assert isinstance(self.mean, (list, tuple))
            self.std = self.kwargs['std']
            assert isinstance(self.std, (list, tuple))

    def _normalize(self, image):
        if self.normalize_type == 'mean_std':
            if len(self.mean) == 1:
                for i in range(image.shape[2]):
                    image[..., i] = (image[..., i] - self.mean[0]) / (self.std[0] + 1e-6)
            else:
                for i in range(image.shape[2]):
                    image[..., i] = (image[..., i] - (self.mean[i])) / (self.std[i] + 1e-6)

        elif self.normalize_type == 'z_score':
            mean, std = cv2.meanStdDev(image)
            split_imgs = cv2.split(image)
            for i in range(len(split_imgs)):
                if std[i] < 1e-6:
                    std[i] = 1.
                split_imgs[i] = (split_imgs[i] - mean[i]) / std[i]
            image = cv2.merge(split_imgs)
        return image

    def __call__(self, **inputs):  # numpy image(w,h,c)
        image = inputs['image']
        image = image.astype(np.float32)
        image = self._normalize(image)
        if len(image.shape) == 2:
            image = image[...,np.newaxis]
        image = torch.FloatTensor(image).permute(2, 0, 1)
        if 'image_noise' in inputs :
            image_noise = inputs['image_noise']
            image_noise = self._normalize(image_noise)
            image_noise = torch.FloatTensor(image_noise).permute(2, 0, 1)
            image = torch.cat((image, image_noise), dim=0)
        inputs.update({'image': image})
        return inputs


class Augementations(object):

    def __init__(self):
        self.transforms = []

    def __call__(self, **inputs):
        for aug in self.transforms:
            inputs = aug(**inputs)
        return inputs


def augmentation_cv(param):
    ''' Build augmentation for training and testing '''
    ##################config reading##################
    func_mapping = {
        'newsize': NewSize,
        'resize': Resize,
        'rand_resize': RandomResizedCrop,
        'center_crop': CenterCrop,
        'random_crop': RandomCrop,
        'affine': Affine,
        'gray': Gray,
        'colorjitter': ColorJitter,
        'motion_blur': MotionBlur,
        'add_noise': AddNoise,
        'normalize': Normalize,
        'occlusion': Occlusion
    }
    AUGS = Augementations()
    for k in param:
        if not isinstance(param[k], dict) or not k in func_mapping.keys():
            continue
        _t = func_mapping[k](**param[k])
        AUGS.transforms.append(_t)
    return AUGS


if __name__ == '__main__':
    import yaml
    import numpy as np
    import cv2
    config = yaml.load(open('/home/SENSETIME/fangkairen/dev_ymm/config/fusion/example_fusion_lost_debug.yaml', 'r'))
    cfg_data = config.pop('data')
    training_param = cfg_data.pop('training_param')
    augs = augmentation_cv(training_param)
    #data = cv2.imread('1.jpeg')
    # data= (np.random.rand(500,500,3)*255).astype(np.uint8)
    data = cv2.imread('/home/SENSETIME/fangkairen/zhuchunsen_3_218.jpg')
    # print(data)
    # data = cv2.resize(data, (112,112))
    # / home / SENSETIME / fangkairen / test.jpg
    # landmarks = (np.random.random(212)*255).astype(np.uint8)
    landmarks=[99.9063,153.666,103.328685714,157.980285714,106.751071429,162.294571429,110.173457143,166.608857143,113.595842857,170.923142857,117.018228571,175.237428571,120.440614286,179.551714286,123.863,183.866,103.33,151.564,106.678857143,155.823428571,110.027714286,160.082857143,113.376571429,164.342285714,116.725428571,168.601714286,120.074285714,172.861142857,123.423142857,177.120571429,126.772,181.38]
    landmarks = np.array(landmarks).astype(np.int)
    for i in range(len(landmarks)//2):
        cv2.circle(data,(landmarks[2*i],landmarks[2*i+1]),1,(255,0,255),1)
    cv2.imshow('1', data)
    inputs = {'image': data, 'landmarks': {'landmark': landmarks,'judge_lost':landmarks.copy()}, 'task': ['landmark']}
    output = augs(**inputs)
    result = output['image']
    landmarks = output['landmarks']['landmark'].astype(np.int)
    result = ((result.permute(1,2,0).numpy()*79.6875)+127.5).astype(np.uint8)
    # print(output['image'].shape)
    # print(output['landmarks']['judge_lost'].shape)
    for i in range(len(landmarks)//2):
        cv2.circle(result,(landmarks[2*i],landmarks[2*i+1]),1,(255,0,0),1)
    cv2.imshow('2', result)
    cv2.waitKey()
