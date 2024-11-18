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

import math

import math


def transform_point_to_rotated_coords_clockwise(point, angle=-45.0):
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
    x_new = x * math.cos(radians) - y * math.sin(radians)
    y_new = x * math.sin(radians) + y * math.cos(radians)

    return (x_new, y_new)


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
        # Create color palette
        self.classes = ["background", "dianchi", "jiaonang", "luobo", "shitou", "shuiping", "taoci", "xiaotudou", "yaobaozhuang","yilaguan"]
        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        cls_, confs, masks, angles, centers ,im1= self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )

        return cls_, confs, masks, angles, centers,im1

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

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
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
            x_centers = ((x[..., 0] + x[..., 1]) / 2).astype(int)
            y_centers = ((x[..., 2] + x[..., 3]) / 2).astype(int)
            centers = np.stack((x_centers, y_centers), axis=-1)
            print(centers)
            # Masks -> Segments(contours)
            segments, angles = self.masks2segments(masks)
            bboxes = x[..., :6]
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



            return x[..., 5], x[..., 4], masks, angles, centers,im0
        else:
            return [], [], [], [], [],im0

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
            else:
                point1, point2 = box[1], box[2]

            slope = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')
            # 计算角度（弧度转角度）
            angle = np.degrees(np.arctan(slope)) if slope != float('inf') else 90
            angles.append(angle)
            segments.append(c.astype("float32"))
        return segments,angles

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


def extract_region(image, points, output_size=(640, 640)):
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

model_path = "best.onnx"
model = YOLOv8Seg(model_path)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

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
port = '/dev/ttyTHS1'  # 替换为你的串口号
baudrate = 115200
timeout = 1
ser = serial.Serial(port, baudrate, timeout=timeout)
buffer =""
print("ok")
while True:
    if ser.in_waiting > 0:
            try:
                received_data = ser.readline().decode('ascii').strip()
                buffer += received_data  # 将接收到的数据添加到缓冲区

                # 假设数据以特定标识符结束（例如"\n"）
                if '\n' in buffer:
                    messages = buffer.split('\n')  # 根据标识符分割消息
                    for message in messages:
                        if message:  # 确保消息不为空
                            print(f"接收到的数据: {message}")
                            if message == "detect":  # 替换为实际的条件
                                print("已发现东西落下。")
                                start_time = time.time()
                                while time.time() - start_time <0.6:
                                    ret, frame = cap.read()
                                    time.sleep(0.1)
                                count_detect = 0
                                while count_detect < 5:
                                    ret, frame = cap.read()
                                    # 裁剪图像
                                    start_time = time.time()
                                    frame = extract_region(frame, points=[(446, 22), (824, 28), (818, 406), (446, 400)])
                                    cls_, confs, _, angles, centers, image = model(frame, conf_threshold=0.7 , iou_threshold=0.5)
                                    if(len(cls_)==1):
                                        centers = transform_point_to_rotated_coords_clockwise(centers[0])
                                        data_to_send = f"Tar=j1x{centers[0]}y{centers[1]}a105!"
                                        ser.write(data_to_send.encode('ascii'))
                                        break
                                    print(cls_, confs, angles, centers)
                                    print(time.time() - start_time)
                                    count_detect+= 1

                    buffer = ""  # 清空缓冲区
            except UnicodeDecodeError:
                # 如果解码失败，处理异常
                print("Decoding error: received data contains invalid ASCII characters.")



        # cv2.imshow('image', image)
        # cv2.waitKey(1)
