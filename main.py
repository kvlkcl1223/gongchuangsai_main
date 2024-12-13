# -*- coding: utf-8 -*-

import time
import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import threading
import multiprocessing
import serial
import re
import onnxruntime as ort
import numpy as np
import math
import Jetson.GPIO as GPIO

def is_gpio_low(pin):
    """
    检查指定 GPIO 引脚是否为低电平。

    参数:
    - pin: GPIO 引脚号 (根据 BCM 编号)

    返回:
    - True: 如果引脚是低电平
    - False: 如果引脚是高电平
    """
    GPIO.setmode(GPIO.BOARD)  # 使用 BCM 引脚编号
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(1)
    state = GPIO.input(pin)  # 读取引脚状态
    GPIO.cleanup(pin)  # 清理引脚以释放资源

    return state == GPIO.LOW  # 返回是否为低电平


def process_image(image):

    # 步骤 1: 非局部均值滤波去噪
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 步骤 2: 提升饱和度
    hsv_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 50)  # 增加饱和度
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # # 步骤 3: 使用 CLAHE 增强对比度
    # lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab_image)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced_l = clahe.apply(l)
    # enhanced_lab = cv2.merge((enhanced_l, a, b))
    # final_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    final_image = enhanced_image
    return final_image


def extract_region(image, points=  [(340,28),(950,20),(970,506),(340,520)], output_size=(640, 640)):
    """
    从给定的图像中提取四边形区域，并将其调整为指定的输出大小。

    参数:
    - image: 要提取的图像。
    - points: 四个坐标点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]，表示四边形的四个角。
    - output_size: 提取的区域图像的输出大小 (width, height)，默认为 (640, 640)。

    返回:
    - 提取的区域图像，尺寸为指定的 output_size。
    """
    # 定义目标图像的四个角点，大小为指定的输出大小
    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(np.array(points, dtype="float32"), dst_points)

    # 应用透视变换，并调整为指定的输出大小
    extracted_region = cv2.warpPerspective(image, M, output_size)

    return extracted_region


def read_kernel(image):
    # 定义卷积核（用于锐化）
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 锐化滤波核

    # 对每个颜色通道进行锐化
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """
        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single
        # Create color palette
        self.classes = ["background", "dianchi", "jiaonang", "luobo", "shitou", "shuiping", "taoci", "xiaotudou",
                        "yaobaozhuang", "yilaguan"]
        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]
        # 为每个类别定义面积范围
        self.category_area_rules = {
            0: (0, 0),  # 类别 0 的面积范围在 500-2000 之间
            1: (0.01, 0.4),  # 类别 1 的面积范围在 100-1500 之间
            2: (0.001, 0.1),  # 类别 2 的面积范围在 300-2500 之间
            3: (0.01, 0.3),
            4: (0.001, 0.3),
            5: (0.01, 0.6),
            6: (0.01, 0.3),
            7: (0.001, 0.2),
            8: (0.01, 0.5),
            9: (0.01, 0.5),

        }
        self.total_image_area = 640*640
    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
             # 类别 置信度 掩码 角度 中心 图像
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        if im0.size == 0:
            print("Error: Input image has no content!")
            return [], [], [], [], [], im0, [], []

        # Pre-process
        im0 = extract_region(im0)
        im0 = read_kernel(im0)

        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        cls_, confs, masks, angles, centers, image, areas, width = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        # cv2.imshow('im0', image)
        # cv2.waitKey(1)
        return cls_, confs, masks, angles, centers, image, areas, width  # 类别 置信度 掩码 角度 中心 图像 面积比例 宽度

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32,area_threshold=200000):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos
        # Transpose dim 1: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            # Compute mask areas and filter
            areas = np.sum(masks, axis=(1, 2))  # Calculate areas (number of True values per mask)
            # 标准化面积为占总图片面积的比例
            area_image = im0.shape[0] * im0.shape[1]
            areas = areas / area_image
            areas = np.round(areas, 5)
            print("面积比例", areas)
            # 筛选合理的切片
            valid_indices = []
            for i, (area, cls) in enumerate(zip(areas, x[:, 5].astype(int))):  # x[:, 5] 假设是 cls_
                if cls in self.category_area_rules:
                    min_area, max_area = self.category_area_rules[cls]
                    if min_area <= area <= max_area:
                        valid_indices.append(i)
            if (len(valid_indices) == 0):
                print("面积筛选过后，已无合理值")
                return [], [], [], [], [], im0, [], []
            else:
                x_centers = ((x[valid_indices, 0] + x[valid_indices, 2]) / 2).astype(int)
                y_centers = ((x[valid_indices, 1] + x[valid_indices, 3]) / 2).astype(int)
                centers = np.stack((x_centers, y_centers), axis=-1)

                # Masks -> Segments(contours)
                segments, angles, width = self.masks2segments(masks[valid_indices])
                bboxes = x[valid_indices, :6]
                im_canvas = im0.copy()
                fixed_color = (0, 0, 255)  # Red color in BGR format
                for (*box, conf, cls_), segment in zip(bboxes, segments):
                    # 使用固定颜色替换 self.color_palette(int(cls_), bgr=True)
                    cv2.polylines(im0, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
                    cv2.fillPoly(im_canvas, np.int32([segment]), fixed_color)

                    cv2.rectangle(
                        im0,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        fixed_color,
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        im0,
                        f"{self.classes[int(cls_)]}: {conf:.3f}",
                        (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        fixed_color,
                        2,
                        cv2.LINE_AA,
                    )

                # Mix image
                im0 = cv2.addWeighted(im_canvas, 0.3, im0, 0.7, 0)
                cv2.imwrite(f"{time.time()}.jpg",im0)
                cls_ = np.array(x[valid_indices, 5], dtype=int).tolist()

                return cls_, x[valid_indices, 4], masks[valid_indices], angles, centers, im0, areas[valid_indices], width

        else:
            return [], [], [], [], [], im0, [], []
    @staticmethod
    def masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        angles = []
        short_edges = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            box = cv2.boxPoints(cv2.minAreaRect(c.astype("float32")))
            box = np.intp(box)

            # 找到长边的两个点
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[1] - box[2])

            if width > height:
                point1, point2 = box[0], box[1]
                short_edge = height
            else:
                point1, point2 = box[1], box[2]
                short_edge = width

                # 保存短边长度
            short_edges.append(short_edge)

            slope = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')
            # 计算角度（弧度转角度）
            angle = np.degrees(np.arctan(slope)) if slope != float('inf') else 90
            angle = angle if angle >= 0 else 90-angle
            # 理论计算应该是 -1*angle+240
            angle = -1*angle + 240-90
            if angle > 180:
                angle = angle-180
            elif angle < 0:
                angle = angle+180
            angles.append(round(angle, 1))


            segments.append(c.astype("float32"))
        return segments, angles, short_edges

    @staticmethod
    def crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape

        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks




def process_string(input_string):
    # 检查帧头是否符合 "帧头=xx！" 的格式
    header_match = re.match(r'^(.*?)=(.*?)!$', input_string)
    if not header_match:
        print("无效的格式")
        return

    frame_header = header_match.group(1).strip()
    data = header_match.group(2).strip()

    print(f"帧头: {frame_header}")

    # 处理不同帧头的操作
    if frame_header == "garbage":
        quantities = re.findall(r'i(\d+)\+q(\d+)', data)
        for index, quantity in quantities:
            print(f"Index: {index}, Quantity: {quantity}")
    else:
        print(f"未知的帧头: {frame_header}")

class SimpleApp:

    def __init__(self, root):
        self.root = root
        # self.root.title("美化的应用")
        self.root.attributes('-fullscreen', True)
        self.index = 0
        self.index_double = ""
        self.name = ""
        self.name_double = ""
        self.quantity = 1
        self.quantity_double = ""
        self.state = ""
        self.state_double = ""
        self.flag_start = 0
        self.quantity_harmful = 0
        self.quantity_recyclable = 0
        self.quantity_kitchen = 0
        self.quantity_other = 0
        self.names = ["harmful", "recyclable", "kitchen", "other"]
        self.last_frame_header = ''
        self.last_frame_header_double = ''
        self.full_image_path = "full.png"
        self.video_path = "video.mp4"
        # 加载背景图片
        self.background_image = Image.open('background.jpg')  # 替换为你的背景图片路径
        self.background_image = self.background_image.resize(
            (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.bg_img = ImageTk.PhotoImage(self.background_image)

        # 创建背景标签
        self.background_label = Label(root, image=self.bg_img)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # 设置背景填满整个窗口

        #垃圾总数标签
        self.label_harmful = tk.Label(root, text="有害垃圾 :0   ", font=("fangsong ti", 16), bg='lightblue')
        self.label_harmful.place(relx=0.2, rely=0.2, anchor='center')

        self.label_recyclable = tk.Label(root, text="可回收垃圾 :0   ", font=("fangsong ti", 16), bg='lightblue')
        self.label_recyclable.place(relx=0.4, rely=0.2, anchor='center')

        self.label_kitchen = tk.Label(root, text="厨余垃圾 :0   ", font=("fangsong ti", 16), bg='lightblue')
        self.label_kitchen.place(relx=0.6, rely=0.2, anchor='center')

        self.label_other = tk.Label(root, text="其他垃圾 :0   ", font=("fangsong ti", 16), bg='lightblue')
        self.label_other.place(relx=0.8, rely=0.2, anchor='center')


        # 某次垃圾分类的标签 标号 名字 数量 是否成功
        self.label_index = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_index.place(relx=0.2, rely=0.5, anchor='center')

        self.label_name = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_name.place(relx=0.4, rely=0.5, anchor='center')

        self.label_quantity = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_quantity.place(relx=0.6, rely=0.5, anchor='center')

        self.label_success = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_success.place(relx=0.8, rely=0.5, anchor='center')

        # 双次垃圾分类的标签 标号 名字 数量 是否成功
        self.label_index_double = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_index_double.place(relx=0.2, rely=0.7, anchor='center')

        self.label_name_double = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_name_double.place(relx=0.4, rely=0.7, anchor='center')

        self.label_quantity_double = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_quantity_double.place(relx=0.6, rely=0.7, anchor='center')

        self.label_success_double = tk.Label(root, text="   ", font=("fangsong ti", 24), bg='lightblue')
        self.label_success_double.place(relx=0.8, rely=0.7, anchor='center')

        # self.label_index = tk.Label(root, text="index:   ", font=("fangsong ti", 32), bg='lightblue')
        # self.label_index.place(relx=0.2, rely=0.5, anchor='center')
        #
        # self.label_name = tk.Label(root, text="name:   ", font=("fangsong ti", 32), bg='lightblue')
        # self.label_name.place(relx=0.4, rely=0.5, anchor='center')
        #
        # self.label_quantity = tk.Label(root, text="quantity:   ", font=("fangsong ti", 32), bg='lightblue')
        # self.label_quantity.place(relx=0.6, rely=0.5, anchor='center')
        #
        # self.label_success = tk.Label(root, text="state:   ", font=("fangsong ti", 32), bg='lightblue')
        # self.label_success.place(relx=0.8, rely=0.5, anchor='center')

        # self.entry = tk.Entry(root, font=("fangsong ti", 24))
        # self.entry.place(relx=0.5, rely=0.2, anchor='center')

        # self.button = tk.Button(root, text="更新参数", command=self.update_parameter, font=("fangsong ti", 24))
        # self.button.place(relx=0.5, rely=0.3, anchor='center')

        self.exit_button = tk.Button(root, text="退出", command=self.exit_app, font=("fangsong ti", 24))
        self.exit_button.place(relx=0.5, rely=0.4, anchor='center')

        self.video_label = Label(root)
        self.video_label.place(x=0, y=0, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())  # 设置为全屏
        self.play_video(self.video_path)  # 替换为你的视频文件路径

        # 显示图片
        self.image_label = Label(root)
        self.background_label.lower()

    # 初始化参数
    def init_parameter(self):
        self.index = ""
        self.name = ""
        self.quantity = ""
        self.state = ""

    def update_display(self):
        self.label_harmful.config(text=f"有害垃圾 : {self.quantity_harmful}")
        self.label_recyclable.config(text=f"可回收垃圾 : {self.quantity_recyclable}")
        self.label_kitchen.config(text=f"厨余垃圾 : {self.quantity_kitchen}")
        self.label_other.config(text=f"其他垃圾 : {self.quantity_other}")
        # self.label_index.config(text=f"index: {self.index}")
        # self.label_name.config(text=f"name: {self.name}")
        # self.label_quantity.config(text=f"quantity: {self.quantity}")
        # self.label_success.config(text=f"state: {self.state}")
        self.label_index.config(text=f"{self.index} ")
        self.label_name.config(text=f"{self.name} ")
        self.label_quantity.config(text=f"{self.quantity} ")
        self.label_success.config(text=f"{self.state} ")
        self.label_index_double.config(text=f"{self.index_double} ")
        self.label_name_double.config(text=f"{self.name_double} ")
        self.label_quantity_double.config(text=f"{self.quantity_double} ")
        self.label_success_double.config(text=f"{self.state_double} ")
    def exit_app(self):
        self.root.quit()  # 退出应用

    def play_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.update_video()

    def update_video(self):
        if self.flag_start:
            self.video_label.destroy()
        else:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))  # 调整为全屏
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(img)

                self.video_label.img = img
                self.video_label.config(image=img)
            else:
                # 视频播放结束，重新播放
                self.cap.release()  # 释放视频资源
                self.play_video(self.video_path)  # 重新打开视频文件
            self.video_label.after(10, self.update_video)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((400, 300))  # 调整图片大小
        img = ImageTk.PhotoImage(img)

        image_label = Label(self.root, image=img)
        image_label.image = img
        image_label.pack(pady=20)

    # 满载显示
    def full_display(self):
        img = cv2.imread(self.full_image_path)
        img = cv2.resize(img, (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))  # 调整为全屏
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道
        img = Image.fromarray(img)  # 转换为PIL格式
        img = ImageTk.PhotoImage(img)

        image_label = Label(self.root, image=img)
        image_label.image = img
        image_label.place(x=0, y=0, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())  # 设置为全屏
        image_label.tkraise()
        self.root.after(3000, image_label.destroy)  # 3秒后销毁标签


def display_process(queue_display,queue_display_ser):
    flag_start = 1
    while queue_display.empty():
        time.sleep(0.1)
    data_display_discard = queue_display.get()
    print("data_display_discard",data_display_discard)
    root = tk.Tk()
    app = SimpleApp(root)
    # 定期检查队列消息
    # 逻辑是 接收到垃圾种类存到last_frame_header再次收到success则把这个种类的垃圾加1
    def check_queue():

        # try:
        #     # 来自main的命令
        #     message = queue_display.get_nowait()  # 尝试获取消息
        #     print("来自main",message)
        #     # app.update_label(message)  # 更新标签
        #     app.flag_start = 0
        #     app.video_label.destroy()
        #     # 使用正则表达式提取多个 key=value! 对
        #     pattern = r'(\w+)=([\w\W]*?)!'
        #     matches = re.findall(pattern, message)
        #     if matches:
        #         if len(matches) ==1:
        #             for key, value in matches:
        #                 key = key.strip()
        #                 value = value.strip()
        #
        #                 # 根据解析到的 key 和 value 处理逻辑
        #                 if key == "fail":
        #                     app.state = "fail"
        #                 elif key == "ok":
        #                     print()
        #                 else:
        #                     app.index += 1
        #                     app.name = key
        #                     app.state = "classifying"
        #                     app.last_frame_header = key
        #                     app.name_double = ""
        #                     app.quantity_double = ""
        #                     app.last_frame_header = ""
        #                     app.index_double = ""
        #                     app.state_double = ""
        #         elif len(matches) == 2:
        #             app.index += 1
        #             app.state = "classifying"
        #             app.state_double = "classifying"
        #             key1, value1 = matches[0]
        #             key2, value2 = matches[1]
        #             if key1 == key2:
        #                 app.name = key1
        #                 app.quantity = 2
        #                 app.last_frame_header = key1
        #                 app.name_double = ""
        #                 app.quantity_double = ""
        #                 app.last_frame_header = ""
        #                 app.index_double = ""
        #             else:
        #                 app.name = key1
        #                 app.quantity = 1
        #                 app.last_frame_header = key1
        #                 app.name_double = key2
        #                 app.quantity_double = 1
        #                 app.last_frame_header_double = key2
        #                 app.index_double = app.index
        #             # 调用更新显示的函数
        #         app.update_display()
        #
        #
        #
        #             # # 处理不同帧头的操作
        #             # if frame_header == "garbage":
        #             #     app.init_parameter()
        #             #     quantities = re.findall(r'i(\d+)\+q(\d+)', data)
        #             #     for index, quantity in quantities:
        #             #         print(f"Index: {index}, Quantity: {quantity}")
        #             #         app.index += str(index) + ","  # 将 index 转换为字符串并加上逗号
        #             #         app.name += app.names[int(index)] + ","
        #             #         app.quantity += str(quantity) + ","
        #             #         app.success = "OK"
        #             #     app.index = app.index[:-1]  # 移除最后的逗号
        #             #     print(app.index)
        #             #     app.name = app.name[:-1]
        #             #     app.quantity = app.quantity[:-1]
        #             #     app.update_display()
        #             # elif frame_header == "harmful":
        #             #     app.index += 1
        #             # elif frame_header == "full":
        #             #     app.full_display()
        #             # else:
        #             #     print(f"未知的帧头: {frame_header}")
        #
        #
        # except Exception as e:
        #     pass

        try:
            # 来自串口的命令
            # 满载以及动作完成直接由串口发送
            # 来自main的命令
            message_ser = queue_display_ser.get_nowait()  # 尝试获取消息
            print("来自串口", message_ser)
            # app.update_label(message)  # 更新标签
            app.flag_start = 0
            app.video_label.destroy()
            header_match = re.match(r'^(.*?)=(.*?)!$', message_ser)
            if header_match:
                frame_header = header_match.group(1).strip()
                data = header_match.group(2).strip()
                if frame_header == "full":
                    app.full_display()
                elif frame_header == "success":
                    app.state = "OK!"
                    if app.last_frame_header == "有害垃圾":
                        app.quantity_harmful += 1
                    elif app.last_frame_header == "可回收垃圾":
                        app.quantity_recyclable += 1
                    elif app.last_frame_header == "厨余垃圾":
                        app.quantity_kitchen += 1
                    elif app.last_frame_header == "其他垃圾":
                        app.quantity_other += 1

                    if app.last_frame_header_double == "有害垃圾":
                        app.quantity_harmful += 1
                    elif app.last_frame_header_double == "可回收垃圾":
                        app.quantity_recyclable += 1
                    elif app.last_frame_header_double == "厨余垃圾":
                        app.quantity_kitchen += 1
                    elif app.last_frame_header_double == "其他垃圾":
                        app.quantity_other += 1
                else:
                    app.index += 1
                    app.name = frame_header
                    app.state = "classifying"
                    app.last_frame_header = frame_header
                    app.name_double = ""
                    app.quantity_double = ""
                    app.last_frame_header = ""
                    app.index_double = ""
                    app.state_double = ""
                app.update_display()
        except Exception:
            pass  # 如果队列为空，继续

        root.after(100, check_queue)  # 每100毫秒再次检查队列

    check_queue()  # 启动检查队列的函数
    root.mainloop()




def transform_point_to_rotated_coords_clockwise(point, angle=45.0):
    """
    计算坐标系顺时针旋转指定角度后，点在新坐标系中的位置。

    参数:
    - point (tuple): 原始坐标点 (x, y)。
    - angle (float): 坐标系顺时针旋转的角度，单位为度。

    返回:
    - tuple: 点在新坐标系中的坐标 (x', y')。
    """
    # 转换角度为弧度
    radians = math.radians(angle)

    # 提取点的坐标
    x, y = point

    # 使用旋转公式，方向改变
    x_new = x * math.cos(radians) + y * math.sin(radians)
    y_new = -x * math.sin(radians) + y * math.cos(radians)

    # 限制小数点位数
    x_new = round(x_new, 2)
    y_new = round(y_new, 2)

    return (x_new, y_new)




def group_coordinates_by_threshold(coords, threshold=30):
    """
    按距离阈值对坐标进行分组，返回每组的索引切片。

    Args:
        coords (list): 坐标列表，形如 [(x1, y1), (x2, y2), ...]。
        threshold (float): 判断重复的距离阈值。

    Returns:
        tuple:
            int: 非重复组的数量。
            list: 分组的索引切片，每组是一个列表。
    """
    coords = np.array(coords)  # 转为 NumPy 数组便于计算
    groups = []  # 存储分组的索引

    visited = set()  # 记录已处理的点索引
    for i, coord in enumerate(coords):
        if i in visited:
            continue
        group = [i]  # 初始化当前组，包含当前点
        visited.add(i)  # 标记当前点已访问

        # 找到与当前点接近的其他点
        for j in range(len(coords)):
            if j not in visited and np.linalg.norm(coord - coords[j]) <= threshold:
                group.append(j)
                visited.add(j)

        groups.append(group)  # 添加到分组列表

    return len(groups), groups



def open_camera(try_from: int = 0, try_to: int = 10):
    # 打开摄像头

    cam = cv2.VideoCapture()
    for i in range(try_from, try_to):
        cam.open(i, cv2.CAP_V4L2)
        if cam.isOpened():
            return cam, i
    raise Exception("Camera not found")

def yolo_process(queue_display,queue_receive, queue_transmit,queue_main_ser):

    # time.sleep(10)
    # while True:
    #     # 该部分处理为进行视觉识别算法，得到目标，将信息显示在屏幕上
    #     print("运行 YOLO 算法...")
    #     queue_main_ser.put('harmful=!')
    #     #queue.put("full=!")
    #     time.sleep(3)  # 模拟 YOLO 处理
    #     # queue_display.put('success=!')
    #     # time.sleep(3)
    #     # queue.put('garbage=i2+q2+i3+q19+i1+q7!')

    model_path = "train_17.onnx"
    model = YOLOv8Seg(model_path)
    model_large_path = "train_17.onnx"
    model_large = YOLOv8Seg(model_large_path)
    cap, i = open_camera()
    # cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # 等待画面刷新时间
    time_update = 0.2
    # 角度偏差量
    angle_error = 0
    # 垃圾轮数计数
    index_garbage = 0
    # 预热模型
    start_time = time.time()
    while time.time() - start_time < 0.2:
        ret, frame = cap.read()
        cls_, confs, _, angles, centers, image, areas, width = model(frame, conf_threshold=0.7, iou_threshold=0.5)
        cls_, confs, _, angles, centers, image, areas, width = model_large(frame, conf_threshold=0.7, iou_threshold=0.5)
    queue_display.put("OK=!")
    print("ok")
    #应该在所有的东西启动完成时，在屏幕上显示东西

    #     0: background
    #     1: dianchi
    #     2: jiaonang
    #     3: luobo
    #     4: shitou
    #     5: shuiping
    #     6: taoci
    #     7: xiaotudou
    #     8: yaobaozhuang
    #     9: yilaguan
    while True:
        time.sleep(0.1)
        # 非空表示有来自stm32的命令
        if (not queue_receive.empty()):
            command_receive = queue_receive.get()
            if command_receive == 'detect':
                index_garbage += 1
                # 更改屏幕 显示在分类
                command_display = f'classifying'
                queue_display.put(command_display)
                start_time = time.time()
                # 刷新画面
                while time.time()-start_time < 0.1:
                    ret, frame = cap.read()
                    time.sleep(0.1)

                final_cls_ = []
                final_confs = []
                final_angles = []
                final_centers = []
                final_images = []
                final_areas = []
                final_widths = []
                command = f'Tar='
                command_display = f''

                # 单垃圾分类
                if is_gpio_low(13):
                    print("当前单垃圾分类")
                    count = 0
                    while count < 5:
                        count += 1
                        # 置信度逐级递减
                        conf_threshold = 0.80-count*0.05
                        start_time = time.time()
                        while time.time() - start_time < time_update:
                            ret, frame = cap.read()
                        ret, frame = cap.read()
                        start_time = time.time()
                        cls_, confs, _, angles, centers, image, areas, width = model(frame, conf_threshold=conf_threshold, iou_threshold=0.5)
                        print("本次耗时：", time.time()-start_time)
                        print("第一次", cls_, confs, angles, centers, areas, width)

                        if len(cls_) == 1:
                            # 继续识别一次，与上次作比较
                            start_time = time.time()
                            while time.time()-start_time < time_update:
                                ret, frame = cap.read()
                            ret, frame = cap.read()
                            new_cls_, new_confs, _, new_angles, new_centers, new_image, new_areas, new_width = model(frame, conf_threshold=conf_threshold,
                                                                           iou_threshold=0.5)
                            print("第二次", new_cls_, new_confs, new_angles, new_centers, new_areas)
                            # 认为识别正确
                            if new_cls_==  cls_:
                                final_cls_ = cls_
                                final_confs = confs
                                final_angles = angles
                                final_centers = centers
                                final_image = image
                                break
                            # 可采取更大模型去识别本次或其他措施
                            else:
                                print("二次识别与一次识别矛盾")
                                time.sleep(2)
                                start_time = time.time()
                                while time.time() - start_time < time_update:
                                    ret, frame = cap.read()
                                ret, frame = cap.read()
                                large_cls_, large_confs, _, large_angles, large_centers, large_image, large_areas, large_width = model_large(frame,
                                                                                                          conf_threshold=conf_threshold,
                                                                                                          iou_threshold=0.5)
                                print("large", large_cls_,  large_confs, large_angles, large_centers, large_areas)
                                if len(large_cls_) > 0:
                                    # 仲裁哪次是正确的
                                    if large_cls_ == cls_ or large_cls_ == new_cls_:
                                        print("仲裁匹配")
                                        final_cls_ = large_cls_
                                        final_confs = large_confs
                                        final_angles = large_angles
                                        final_centers = large_centers
                                        final_image = large_image
                                        break
                                    # 仲裁之后依然无法判断，投到可回收垃圾
                                    # 仲裁之后依然无法判断, 直接为仲裁结果
                                    else:
                                        print("仲裁后依然无法处理")


                                # 仲裁结果为空值，将第一次作为最终结果
                                else:
                                    final_cls_ = cls_
                                    final_confs = confs
                                    final_angles = angles
                                    final_centers = centers
                                    final_image = image

                                    break
                    # 连续5次未识别出任何，大模型识别一次
                    else:
                        start_time = time.time()
                        while time.time() - start_time < time_update:
                            ret, frame = cap.read()
                        ret, frame = cap.read()
                        large_cls_, large_confs, _, large_angles, large_centers, large_image, large_areas, large_width = model_large(
                            frame,
                            conf_threshold=0.7,
                            iou_threshold=0.5)
                        print("large", large_cls_, large_confs, large_angles, large_centers, large_areas)
                        if len(large_cls_) > 0:
                            # 仲裁哪次是正确的

                            final_cls_ = large_cls_
                            final_confs = large_confs
                            final_angles = large_angles
                            final_centers = large_centers
                            final_image = large_image

                        else:
                            print("单垃圾,未成功识别出来")



                # 双垃圾分类
                else:
                    print("当前双垃圾分类")
                    sum_cls_ = []
                    sum_angles = []
                    sum_centers = []
                    sum_confs = []
                    sum_areas = []
                    sum_widths = []
                    # 统计结果值
                    count = 0
                    while count < 5:
                        count += 1
                        start_time = time.time()
                        while time.time() - start_time < time_update:
                            ret, frame = cap.read()
                        ret, frame = cap.read()
                        # 置信度逐级递减
                        conf_threshold = 0.75 - count * 0.05
                        start_time = time.time()
                        cls_, confs, _, angles, centers, image, areas, widths = model(frame, conf_threshold=conf_threshold,
                                                                              iou_threshold=0.5)
                        print("本次耗时：", time.time() - start_time)
                        print(f"第{count}次", cls_, confs, angles, centers, areas)
                        # 加入序列之中
                        sum_cls_.extend(cls_)
                        sum_angles.extend(angles)
                        sum_confs.extend(confs)
                        sum_areas.extend(areas)
                        sum_centers.extend(centers)
                        sum_widths.extend(widths)
                    print("sum:", sum_cls_, sum_confs, sum_angles, sum_centers, sum_areas)
                    # 对统计结果进行处理
                    group_count, grouped_indices = group_coordinates_by_threshold(sum_centers)
                    print("数量", group_count, "索引",grouped_indices)
                    if group_count == 2:

                        final_cls_.append(sum_cls_[grouped_indices[0][0]])
                        final_cls_.append(sum_cls_[grouped_indices[1][0]])
                        final_centers.append(sum_centers[grouped_indices[0][0]])
                        final_centers.append(sum_centers[grouped_indices[1][0]])
                        final_angles.append(sum_angles[grouped_indices[0][0]])
                        final_angles.append(sum_angles[grouped_indices[1][0]])
                        final_areas.append(sum_areas[grouped_indices[0][0]])
                        final_areas.append(sum_areas[grouped_indices[1][0]])
                    # 此情况出现概率较小，选择平均置信度最高的或者次数与概率积的和
                    elif group_count > 2:
                        print("多于两个")
                        # 计算每组的权重 (概率 × 次数)
                        weights = []
                        for group in grouped_indices:
                            group_weight = sum(sum_cls_[index] for index in group)  # 假设 sum_cls_ 中存储的是概率或次数
                            weights.append(group_weight)

                        # 按权重从大到小排序，取前两组
                        sorted_indices = np.argsort(weights)[::-1]  # 从大到小排序索引
                        top_two_groups = sorted_indices[:2]  # 取权重最大的两组索引

                        # 合并前两组的数据
                        for group_index in top_two_groups:
                            final_cls_.append(sum_cls_[group_index])
                            final_centers.append(sum_centers[group_index])
                            final_angles.append(sum_angles[group_index])
                            final_areas.append(sum_areas[group_index])
                            final_widths.append(sum_widths[group_index])
                            # for index in grouped_indices[group_index]:
                            #     final_cls_.append(sum_cls_[index])
                            #     final_centers.append(sum_centers[index])
                            #     final_angles.append(sum_angles[index])
                            #     final_areas.append(sum_areas[index])
                            #     final_widths.append(sum_widths[index])
                    # 少于两个，尝试用大模型补充几次结果
                    elif group_count < 2:
                        print("少于两个")
                        count = 0
                        while count < 2:
                            count += 1
                            start_time = time.time()
                            while time.time() - start_time < time_update:
                                ret, frame = cap.read()
                            ret, frame = cap.read()
                            # 置信度逐级递减
                            conf_threshold = 0.75 - count * 0.05
                            start_time = time.time()
                            cls_, confs, _, angles, centers, image, areas, widths = model_large(frame, conf_threshold=conf_threshold,
                                                                                  iou_threshold=0.5)
                            print("本次耗时：", time.time() - start_time)
                            print(f"large,第{count}次", cls_, confs, angles, centers, areas)
                            # 加入序列之中
                            sum_cls_.extend(cls_)
                            sum_angles.extend(angles)
                            sum_confs.extend(confs)
                            sum_areas.extend(areas)
                            sum_centers.extend(centers)
                            sum_widths.extend(widths)
                            group_count, grouped_indices = group_coordinates_by_threshold(sum_centers)
                            print("增添后","数量", group_count, "索引", grouped_indices)
                        # 根据添加后的结果进行上述操作
                        if group_count == 2:
                            print("增添后正好两个")
                            final_cls_.append(sum_cls_[grouped_indices[0][0]])
                            final_cls_.append(sum_cls_[grouped_indices[1][0]])
                            final_confs.append(sum_cls_[grouped_indices[0][0]])
                            final_confs.append(sum_cls_[grouped_indices[1][0]])
                            final_centers.append(sum_centers[grouped_indices[0][0]])
                            final_centers.append(sum_centers[grouped_indices[1][0]])
                            final_angles.append(sum_angles[grouped_indices[0][0]])
                            final_angles.append(sum_angles[grouped_indices[1][0]])
                            final_areas.append(sum_areas[grouped_indices[0][0]])
                            final_areas.append(sum_areas[grouped_indices[1][0]])
                            final_widths.append(sum_widths[grouped_indices[0][0]])
                            final_widths.append(sum_widths[grouped_indices[1][0]])
                        # 此情况出现概率较小，选择平均置信度最高的或者次数与概率积的和
                        elif group_count > 2:
                            print("增添后超过两个")
                            # 计算每组的权重 (概率 × 次数)
                            weights = []
                            for group in grouped_indices:
                                group_weight = sum(sum_cls_[index] for index in group)  # 假设 sum_cls_ 中存储的是概率或次数
                                weights.append(group_weight)

                            # 按权重从大到小排序，取前两组
                            sorted_indices = np.argsort(weights)[::-1]  # 从大到小排序索引
                            top_two_groups = sorted_indices[:2]  # 取权重最大的两组索引

                            # 合并前两组的数据
                            for group_index in top_two_groups:
                                final_cls_.append(sum_cls_[group_index[0]])
                                final_confs.append(sum_confs[group_index[0]])
                                final_centers.append(sum_centers[group_index[0]])
                                final_angles.append(sum_angles[group_index[0]])
                                final_areas.append(sum_areas[group_index[0]])
                                final_widths.append(sum_widths[group_index[0]])



                        elif group_count < 2 and group_count >0:
                            final_cls_.append(sum_cls_[0])
                            final_confs.append(sum_confs[0])
                            final_centers.append(sum_centers[0])
                            final_angles.append(sum_angles[0])
                            final_areas.append(sum_areas[0])
                            final_widths.append(sum_widths[0])
                    # if len(cls_) == 2:
                    #     # 继续识别一次，与上次作比较
                    #     ret, frame = cap.read()
                    #     time.sleep(0.1)
                    #     ret, frame = cap.read()
                    #     new_cls_, new_confs, _, new_angles, new_centers, image, new_areas = model(frame, conf_threshold=conf_threshold,
                    #                                                    iou_threshold=0.5)
                    #     print("第二次", new_cls_, new_confs, new_angles, new_centers, new_areas)
                    #     if(len(new_areas)==2):
                    #         # 直接认为识别正确
                    #         if set(new_cls_) == set(cls_) :
                    #             final_cls_ = cls_
                    #             final_confs = confs
                    #             final_angles = confs
                    #             final_centers = centers
                    #             final_image = image
                    #             break
                    #         # 两次不匹配，申请仲裁
                    #         else:
                    #             print("二次识别与一次识别矛盾")
                    #             ret, frame = cap.read()
                    #             time.sleep(0.1)
                    #             ret, frame = cap.read()
                    #             large_cls_, large_confs, _, large_angles, large_centers, image, large_areas = model_large(
                    #                 frame,
                    #                 conf_threshold=conf_threshold,
                    #                 iou_threshold=0.5)
                    #             print("large",large_cls_, large_confs, large_angles,large_centers,large_areas)
                    #
                    #             if (len(large_cls_) ==2):
                    #                 # 仲裁结果与某次结果匹配
                    #                 if set(large_cls_) == set(cls_) or set(large_cls_) == set(large_cls_):
                    #                     print()
                    #                 # 不匹配的结果
                    #                 else:
                    #                     print()


                # 最终结果处理
                print("final", final_cls_,final_confs,final_angles,final_centers,final_areas)
                # 单垃圾
                if is_gpio_low(13):
                    if (len(final_cls_)==1):
                        # 有害垃圾
                        if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                            command += f'q2!'
                            command_display += '有害垃圾=!'
                        # 可回收垃圾
                        elif final_cls_[0] == 5 or final_cls_[0]== 9:
                            command += f'q1!'
                            command_display += '可回收垃圾=!'
                        # 厨余垃圾
                        elif final_cls_[0] == 3 or final_cls_[0] == 7:
                            command += f'q3!'
                            command_display += '厨余垃圾=!'
                        # 其他垃圾
                        elif final_cls_[0] or final_cls_[0] == 6:
                            command += f'q4!'
                            command_display += '其他垃圾=!'
                    # 未成功识别，可采取震动措施
                    else:
                        time.sleep(4)
                        command += f'q4!'
                        command_display += '其他垃圾=!'
                #双垃圾
                else:

                    # 最好结果
                    if (len(final_cls_) == 2):
                        # 根据面积从大到小排序
                        # 使用 zip 打包，然后根据规则排序
                        area_threshold = 0.4
                        sorted_data = sorted(
                            zip(final_areas, final_cls_, final_centers, final_angles, final_widths),
                            key=lambda x: (x[0] > area_threshold, x[0]),  # 大于阈值的优先级低，按照面积排序
                            reverse=True  # 面积从大到小排序（前提是未超过阈值）
                        )

                        # 解包回到各自的列表
                        final_areas, final_cls_, final_centers, final_angles, final_widths = map(list,
                                                                                                 zip(*sorted_data))
                         # 两个种类相同，倾倒即可
                        if final_cls_[0] ==final_cls_[1]:
                            # 有害垃圾
                            if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                                command += f'q2!'
                                command_display += '有害垃圾=!有害垃圾=!'
                            # 可回收垃圾
                            elif final_cls_[0] == 5 or final_cls_[0] == 9:
                                command += f'q1!'
                                command_display += '可回收垃圾=!可回收垃圾=!'
                            # 厨余垃圾
                            elif final_cls_[0] == 3 or final_cls_[0] == 7:
                                command += f'q3!'
                                command_display += '厨余垃圾=!厨余垃圾=!'
                            # 其他垃圾
                            elif final_cls_[0] or final_cls_[0] == 6:
                                command += f'q4!'
                                command_display += '其他垃圾=!其他垃圾=!'
                        # 种类不同，先夹后倾倒
                        else:
                            if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                                command += f'j2x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}!'
                                command_display += '有害垃圾=!'
                            # 可回收垃圾
                            elif final_cls_[0] == 5 or final_cls_[0] == 9:
                                command += f'j1x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}!'
                                command_display += '可回收垃圾=!'
                            # 厨余垃圾
                            elif final_cls_[0] == 3 or final_cls_[0] == 7:
                                command += f'j3x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}!'
                                command_display += '厨余垃圾=!'
                            # 其他垃圾
                            elif final_cls_[0] == 4 or final_cls_[0] == 6:
                                command += f'j4x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}!'
                                command_display += '其他垃圾=!'
                    # 只发现一个 夹取一个，剩下的 选择一个不同的垃圾倾倒
                    elif (len(final_cls_) == 1):
                        # 有害垃圾
                        if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                            command += f'j2x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}q4!'
                            command_display += '有害垃圾=!其他垃圾=!'
                        # 可回收垃圾
                        elif final_cls_[0] == 5 or final_cls_[0] == 9:
                            command += f'j1x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}q4!'
                            command_display += '可回收垃圾=!其他垃圾=!'
                        # 厨余垃圾
                        elif final_cls_[0] == 3 or final_cls_[0] == 7:
                            command += f'j3x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}q4!'
                            command_display += '厨余垃圾=!其他垃圾=!'
                        # 其他垃圾
                        elif final_cls_[0] == 4 or final_cls_[0] == 6:
                            command += f'j4x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}l{final_widths[0]}q3!'
                            command_display += '其他垃圾=!厨余垃圾=!'
                    # 完蛋，一个没识别出来，随机倾倒吧
                    elif (len(final_cls_) == 0):
                        command += f'q4!'
                        command_display += '其他垃圾=!其他垃圾=!'
                        print()
                # 处理未成功识的情识别
                if (command == ""):
                    print()

                # 信息发送到其他进程
                if (command != ""):
                    queue_transmit.put(command)
                    print("放入发送队列信息",command)
                if (command_display != ""):
                    queue_main_ser.put(command_display)
                    print("放入显示队列信息",command_display)
                    # print("种类",model.classes[cls_],"置信度",confs, "角度",angles, "中心点",centers)
                    # # 先用夹子丢需要压缩的垃圾，再用夹子丢其他，最后直接倾倒
                    # if len(cls_) ==1:
                    #     # 先清队列
                    #     data_to_receive = queue_receive.get()
                    #     for i in range(len(cls_)):
                    #         command = f'Tar='
                    #         command_display = f''
                    #         if len(cls_)-i == 1 and cls_[i] != 5 and cls_[i] != 9:
                    #             # 有害垃圾
                    #             if cls_[i] == 1 or cls_[i] == 2 or cls_[i] == 8:
                    #                 command += f'q1!'
                    #                 command_display += 'harmful=!'
                    #             # 厨余垃圾
                    #             elif cls_[i] == 3 or cls_[i] == 7:
                    #                 command += f'q3!'
                    #                 command_display += 'kitchen=!'
                    #             # 其他垃圾
                    #             elif cls_[i] == 4:
                    #                 command += f'q4!'
                    #                 command_display += 'other=!'
                    #         else:
                    #             # 丢需压缩垃圾即可回收垃圾
                    #             if cls_[i] == 5 or cls_[i] == 9:
                    #                 command += f'j2x{centers[i][0]}y{centers[i][1]}a{angles[i]-0.0}!'
                    #                 command_display += 'recyclable=!'
                    #             # 有害垃圾
                    #             elif cls_[i] == 1 or cls_[i] == 2 or cls_[i] == 8:
                    #                 command += f'j1x{centers[i][0]}y{centers[i][1]}a{angles[i] - 0.0}!'
                    #                 command_display += 'harmful=!'
                    #             # 厨余垃圾
                    #             elif cls_[i] == 3 or cls_[i] == 7:
                    #                 command += f'j3x{centers[i][0]}y{centers[i][1]}a{angles[i] - 0.0}!'
                    #                 command_display += 'kitchen=!'
                    #             # 其他垃圾
                    #             elif cls_[i] == 4:
                    #                 command += f'j4x{centers[i][0]}y{centers[i][1]}a{angles[i] - 0.0}!'
                    #                 command_display += 'other=!'


                            # 不需要 直接串口发送到屏幕即可
                            # # 需要stm32发送完成信息
                            # start_time = time.time()
                            # while(not queue_receive.empty() and time.time() - start_time < 8):
                            #     time.sleep(0.1)
                            # # 时间过长也认为是fail或是error
                            # if (time.time() - start_time >=7):
                            #     command_display = "fail=!"
                            #     queue_display.put(command_display)
                            # else:
                            #     data_to_receive = queue_receive.get()
                            #     if data_to_receive=="success":
                            #         command_display = "success=!"
                            #         queue_display.put(command_display)

def uart_transition(com, ser_ttyAMA4):
    serial_cnt = 1  # 调用一次该程序

    while ser_ttyAMA4.in_waiting > 0:
        ser_ttyAMA4.read(ser_ttyAMA4.in_waiting)  # 读取并丢弃所有数据
    ser_ttyAMA4.reset_input_buffer()

    while ser_ttyAMA4.in_waiting == 0:
        ser_ttyAMA4.flushInput()
        ser_ttyAMA4.write(com)
        print("发送的数据", com)
        time.sleep(0.1)
        serial_cnt += 1

        if serial_cnt > 5:
            break
    time.sleep(0.1)
    try:
        while ser_ttyAMA4.in_waiting > 0:
            data_to_discard = ser_ttyAMA4.read()
            print("data_to_discard", data_to_discard)

    except Exception as e:
        # 如果解码失败，处理异常
        print("error")
    # if ser_ttyAMA4.in_waiting > 0:
    #     data_to_discard = ser_ttyAMA4.readline().decode('ascii').strip()
    #     print("data_to_discard", data_to_discard)
def open_serial(port, baudrate, timeout=None, retry_interval=1):
    """
    尝试打开串口，直到成功为止。

    :param port: 串口端口号，例如 'COM3' 或 '/dev/ttyUSB0'
    :param baudrate: 波特率，例如 9600
    :param timeout: 超时时间，默认 None
    :param retry_interval: 重试间隔时间（秒），默认 1 秒
    :return: 打开的 serial.Serial 对象
    """
    while True:
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Successfully opened serial port: {port}")
            return ser  # 返回成功打开的串口对象
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}. Retrying in {retry_interval} second(s)...")
            time.sleep(retry_interval)
def open_serial_command(port):
    start_time = time.time()
    while time.time()-start_time <10:
        try:
            ser = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"Successfully opened serial port: {port}")
            return ser  # 返回成功打开的串口对象
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}. Retrying in {0.1} second(s)...")
            time.sleep(0.1)



def serial_process(queue_receive,queue_transmit,queue_display_ser,queue_main_ser):
    #握手多次发送

    # while True:
    #     time.sleep(3)
    #     queue_display_ser.put("有害垃圾=!")
    #     print("a")
    # 创建串口对象
    # while True:
    #     if not queue_transmit.empty():
    #         data_to_send = queue_transmit.get()
    #
    #
    #     if not queue_main_ser.empty():
    #         data_to_send = queue_main_ser.get()
    #         queue_display_ser.put(data_to_send)
    #
    #     time.sleep(0.1)

    port = '/dev/ttyTHS1'  # 替换为你的串口号
    baudrate = 115200
    timeout = 1
    ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
    ser.reset_input_buffer()
    while ser.in_waiting > 0:
        data_to_discard = ser.read()
    buffer = ""
    received_data = ""
    # ser_command = serial.Serial(
    #     port="/dev/ttyUSB0",
    #     baudrate=115200,
    #     bytesize=serial.EIGHTBITS,
    #     parity=serial.PARITY_NONE,
    #     stopbits=serial.STOPBITS_ONE,
    # )
    # ser_command = open_serial_command("/dev/ttyUSB0")

    while True:
        while True:
            try:
                if ser.in_waiting > 0:  # 检查是否有数据等待读取
                    # 读取一行数据并解码
                    print("来自stm32有信息发送")
                    try:

                        received_data = ser.readline().decode('ascii').strip()
                        print("received_data", received_data)
                        buffer = received_data  # 将接收到的数据添加到缓冲区

                        # 假设数据以特定标识符结束（例如"!"）
                        if '!' in buffer:
                            messages = buffer.split('!')  # 根据标识符分割消息
                            # for message in messages:
                            if messages[-2]:  # 确保消息不为空
                                message = messages[-2]
                                data_to_send = ""
                                print(f"接收到的数据: {message}")
                                if message == "detect":  # 替换为实际的条件
                                    print("已发现有垃圾丢下，准备识别")
                                    queue_receive.put("detect")
                                    print("已放入 queue_receive: detect")
                                    # 延迟清串口
                                    time.sleep(2)
                                    while ser.in_waiting > 0:
                                        data_to_discard = ser.read()
                                        print("detect 后丢弃的数据",data_to_discard)
                                # 满载
                                elif message == "full":
                                    queue_display_ser.put("full=!")
                                # 动作完成
                                elif message == "success":
                                    queue_display_ser.put("success=!")

                                elif  message == "Com=q1" or message == "Com=q2" or message == "Com=q3" or message == "Com=q4":
                                    if message == "Com=q1":  # 替换为实际的条件
                                        data_to_send = "Tar=q1!"
                                        queue_display_ser.put("可回收垃圾=!")
                                        print(data_to_send,"可回收垃圾")
                                    elif message == "Com=q2":  # 替换为实际的条件
                                        data_to_send = "Tar=q2!"
                                        queue_display_ser.put("有害垃圾=!")
                                        print(data_to_send, "有害垃圾")
                                    elif message == "Com=q3":  # 替换为实际的条件
                                        data_to_send = "Tar=q3!"
                                        queue_display_ser.put("厨余垃圾=!")
                                        print(data_to_send, "厨余垃圾")
                                    elif message == "Com=q4":  # 替换为实际的条件
                                        data_to_send = "Tar=q4!"
                                        queue_display_ser.put("其他垃圾=!")
                                        print(data_to_send, "其他垃圾")
                                    uart_transition(data_to_send.encode('ascii'),ser)
                                    # 等待时间 清除main 发送的东西 避免二次

                                    time.sleep(0.01)
                                    start_time = time.time()
                                    while time.time()-start_time < 10 and  queue_transmit.empty():
                                        time.sleep(0.1)
                                    time.sleep(0.1)
                                    while not queue_transmit.empty():
                                        data_to_discard =queue_transmit.get()
                                        print("丢弃队列 queue_transmit", data_to_discard)
                                    while not queue_main_ser.empty():
                                        data_to_discard = queue_main_ser.get()
                                        print("丢弃队列 queue_main_ser", data_to_discard)
                                    print("队列已清空，是否为空：", queue_transmit.empty())  # 输出 True
                        buffer = ""  # 清空缓冲区
                        received_data = ""
                    except UnicodeDecodeError:
                        # 如果解码失败，处理异常
                        # queue_transmit.put("Tar=repeat!")
                        print("Decoding error: received data contains invalid ASCII characters.")

            except OSError as e:
                print(f"OSError occurred: {e}")
                ser.close()
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
                print("已重新打开")
            except serial.SerialException as e:
                print(f"SerialException occurred: {e}")
                print("Attempting to reinitialize the serial port...")
                ser.close()
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
                print("已重新打开")
            except Exception as e:
                print(f"Unexpected error: {e}")
                ser.close()
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
                print("已重新打开")



            # try:
            #     if ser_command.in_waiting > 0:  # 检查是否有数据等待读取
            #         # 读取一行数据并解码
            #         try:
            #             received_data = ser_command.readline().decode('ascii').strip()
            #             print("received_data", received_data)
            #             buffer += received_data  # 将接收到的数据添加到缓冲区
            #
            #             # 假设数据以特定标识符结束（例如"!"）
            #             if '!' in buffer:
            #                 messages = buffer.split('!')  # 根据标识符分割消息
            #                 for message in messages:
            #                     if message:  # 确保消息不为空
            #                         data_to_send = ""
            #                         print(f"command接收到的数据: {message}")
            #                         if message == "Com=q1":  # 替换为实际的条件
            #                             data_to_send = "Tar=q1!"
            #                             queue_display_ser.put("可回收垃圾=!")
            #                             print(data_to_send,"可回收垃圾")
            #                         elif message == "Com=q2":  # 替换为实际的条件
            #                             data_to_send = "Tar=q2!"
            #                             queue_display_ser.put("有害垃圾=!")
            #                             print(data_to_send, "有害垃圾")
            #                         elif message == "Com=q3":  # 替换为实际的条件
            #                             data_to_send = "Tar=q3!"
            #                             queue_display_ser.put("厨余垃圾=!")
            #                             print(data_to_send, "厨余垃圾")
            #                         elif message == "Com=q4":  # 替换为实际的条件
            #                             data_to_send = "Tar=q4!"
            #                             queue_display_ser.put("其他垃圾=!")
            #                             print(data_to_send, "其他垃圾")
            #                         uart_transition(data_to_send.encode('ascii'),ser)
            #                         # 等待时间 清除main 发送的东西 避免二次
            #
            #                         time.sleep(0.1)
            #                         start_time = time.time()
            #                         while time.time()-start_time < 10 and (not queue_transmit.empty()):
            #                             time.sleep(0.1)
            #                         time.sleep(0.1)
            #                         while not queue_transmit.empty():
            #                             queue_transmit.get()
            #                         while not queue_display_ser.empty():
            #                             queue_display_ser.get()
            #                         print("队列已清空，是否为空：", queue_transmit.empty())  # 输出 True
            #                 buffer = ""  # 清空缓冲区
            #         except UnicodeDecodeError:
            #             # 如果解码失败，处理异常
            #             # queue_transmit.put("Tar=repeat!")
            #             print("Decoding error: received data contains invalid ASCII characters.")
            #
            # except Exception as e:
            #     print(f"Unexpected error: {e}")
            #     ser_command.close()
            #     time.sleep(0.2)  # 程序暂停一秒后重试
            #     ser_command = open_serial_command(port="/dev/ttyUSB0")
            #     print("已重新打开")

            if is_gpio_low(7):
                queue_display_ser.put("full=!")
                time.sleep(1)


            if not queue_transmit.empty():
                data_to_send = queue_transmit.get()
                try:
                    uart_transition(data_to_send.encode('ascii'),ser)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    time.sleep(0.2)  # 程序暂停一秒后重试
                    ser.close()
                    ser = serial.Serial(port, baudrate, timeout=timeout)
                    uart_transition(data_to_send.encode('ascii'),ser)

            if not queue_main_ser.empty():
                data_to_send = queue_main_ser.get()
                queue_display_ser.put(data_to_send)

            time.sleep(0.1)


if __name__ == '__main__':

    queue_display = multiprocessing.Queue()
    queue_receive = multiprocessing.Queue()
    queue_transmit = multiprocessing.Queue()
    queue_display_ser = multiprocessing.Queue()
    queue_main_ser = multiprocessing.Queue()

    serial_proc = multiprocessing.Process(target=serial_process,
                                          args=(queue_receive, queue_transmit, queue_display_ser, queue_main_ser))
    display_proc = multiprocessing.Process(target=display_process, args=(queue_display, queue_display_ser,))
    main_proc = multiprocessing.Process(target=yolo_process, args=(queue_display, queue_receive, queue_transmit, queue_main_ser))

    display_proc.start()
    main_proc.start()
    serial_proc.start()

    display_proc.join()
    main_proc.join()
    serial_proc.join()
