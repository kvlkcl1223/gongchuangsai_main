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

def extract_region(image, points=[(412,30),(786,24),(794,404),(418,406)], output_size=(640, 640)):
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

        # Pre-process
        im0 = extract_region(im0)
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        cls_, confs, masks, angles, centers,image,areas = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )

        return cls_, confs, masks, angles, centers, image, areas  # 类别 置信度 掩码 角度 中心 图像 面积比例

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
            areas = areas / self.total_image_area
            valid_indices = np.where(areas <= area_threshold)[0]  # Find indices of masks that meet the area condition
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
                return [], [], [], [], [], im0, []
            else:
                x_centers = ((x[valid_indices, 0] + x[valid_indices, 2]) / 2).astype(int)
                y_centers = ((x[valid_indices, 1] + x[valid_indices, 3]) / 2).astype(int)
                centers = np.stack((x_centers, y_centers), axis=-1)

                # Masks -> Segments(contours)
                segments, angles = self.masks2segments(masks[valid_indices])
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
                cls_ = np.array(x[valid_indices, 5], dtype=int).tolist()
                return cls_, x[valid_indices, 4], masks[valid_indices], angles, centers, im0, areas[valid_indices]

        else:
            return [], [], [], [], [], im0, []
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
        return segments, angles

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
        self.name = ""
        self.quantity = 1
        self.state = ""
        self.flag_start = 0
        self.quantity_harmful = 0
        self.quantity_recyclable = 0
        self.quantity_kitchen = 0
        self.quantity_other = 0
        self.names = ["harmful", "recyclable", "kitchen", "other"]
        self.last_frame_header = ''
        self.full_image_path = "full.png"
        self.video_path = "10643243.mp4"
        # 加载背景图片
        self.background_image = Image.open('background.jpg')  # 替换为你的背景图片路径
        self.background_image = self.background_image.resize(
            (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.bg_img = ImageTk.PhotoImage(self.background_image)

        # 创建背景标签
        self.background_label = Label(root, image=self.bg_img)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # 设置背景填满整个窗口

        #垃圾总数标签
        self.label_harmful = tk.Label(root, text="harmful:   ", font=("Arial", 32), bg='lightblue')
        self.label_harmful.place(relx=0.2, rely=0.2, anchor='center')

        self.label_recyclable = tk.Label(root, text="recyclable:   ", font=("Arial", 32), bg='lightblue')
        self.label_recyclable.place(relx=0.4, rely=0.2, anchor='center')

        self.label_kitchen = tk.Label(root, text="kitchen:   ", font=("Arial", 32), bg='lightblue')
        self.label_kitchen.place(relx=0.6, rely=0.2, anchor='center')

        self.label_other = tk.Label(root, text="other:   ", font=("Arial", 32), bg='lightblue')
        self.label_other.place(relx=0.8, rely=0.2, anchor='center')


        # 某次垃圾分类的标签 标号 名字 数量 是否成功
        self.label_index = tk.Label(root, text="index:   ", font=("Arial", 32), bg='lightblue')
        self.label_index.place(relx=0.2, rely=0.5, anchor='center')

        self.label_name = tk.Label(root, text="name:   ", font=("Arial", 32), bg='lightblue')
        self.label_name.place(relx=0.4, rely=0.5, anchor='center')

        self.label_quantity = tk.Label(root, text="quantity:   ", font=("Arial", 32), bg='lightblue')
        self.label_quantity.place(relx=0.6, rely=0.5, anchor='center')

        self.label_success = tk.Label(root, text="state:   ", font=("Arial", 32), bg='lightblue')
        self.label_success.place(relx=0.8, rely=0.5, anchor='center')

        # self.entry = tk.Entry(root, font=("Arial", 24))
        # self.entry.place(relx=0.5, rely=0.2, anchor='center')

        # self.button = tk.Button(root, text="更新参数", command=self.update_parameter, font=("Arial", 24))
        # self.button.place(relx=0.5, rely=0.3, anchor='center')

        self.exit_button = tk.Button(root, text="退出", command=self.exit_app, font=("Arial", 24))
        self.exit_button.place(relx=0.5, rely=0.4, anchor='center')

        self.video_label = Label(root)
        self.video_label.place(x=0, y=0, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())  # 设置为全屏
        self.play_video(self.video_path)  # 替换为你的视频文件路径

        # 显示图片
        self.image_label = Label(root)
        # self.display_image('E:\波的照片集\sum\DSC_2356.jpg')  # 替换为你的图片文件路径

        # # 调整层次
        # self.video_label.lower()  # 确保视频在图片下层
        # self.image_label.tkraise()  # 确保图片在视频上层
        self.background_label.lower()

    # 初始化参数
    def init_parameter(self):
        self.index = ""
        self.name = ""
        self.quantity = ""
        self.state = ""

    def update_display(self):
        self.label_harmful.config(text=f"harmful: {self.quantity_harmful}")
        self.label_recyclable.config(text=f"recyclable: {self.quantity_recyclable}")
        self.label_kitchen.config(text=f"kitchen: {self.quantity_kitchen}")
        self.label_other.config(text=f"other: {self.quantity_other}")
        self.label_index.config(text=f"index: {self.index}")
        self.label_name.config(text=f"name: {self.name}")
        self.label_quantity.config(text=f"quantity: {self.quantity}")
        self.label_success.config(text=f"state: {self.state}")

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
        self.root.after(10000, image_label.destroy)  # 10秒后销毁标签


def display_process(queue_display,queue_display_ser):
    flag_start = 0
    root = tk.Tk()
    app = SimpleApp(root)
    # 定期检查队列消息
    # 逻辑是 接收到垃圾种类存到last_frame_header再次收到success则把这个种类的垃圾加1
    def check_queue():
        try:
            # 来自main的命令
            message = queue_display.get_nowait()  # 尝试获取消息
            # app.update_label(message)  # 更新标签
            app.flag_start = 0
            app.video_label.destroy()
            header_match = re.match(r'^(.*?)=(.*?)!$', message)
            if header_match:
                frame_header = header_match.group(1).strip()
                data = header_match.group(2).strip()
                if frame_header == "fail":
                    app.state = "fail"
                else:
                    app.index += 1
                    app.name = frame_header
                    app.state = "classifying"
                    app.last_frame_header = frame_header
                app.update_display()

            # 来自串口的命令
            # 满载以及动作完成直接由串口发送
            message = queue_display_ser.get_nowait()  # 尝试获取消息
            header_match = re.match(r'^(.*?)=(.*?)!$', message)
            if header_match:
                frame_header = header_match.group(1).strip()
                data = header_match.group(2).strip()
                if frame_header == "full":
                    app.full_display()
                elif frame_header == "success":
                    app.state = "success"
                    if app.last_frame_header == "harmful":
                        app.quantity_harmful += 1
                    elif app.last_frame_header == "recyclable":
                        app.quantity_recyclable += 1
                    elif app.last_frame_header == "kitchen":
                        app.quantity_kitchen += 1
                    elif app.last_frame_header == "other":
                        app.quantity_other += 1
                app.update_display()

                # # 处理不同帧头的操作
                # if frame_header == "garbage":
                #     app.init_parameter()
                #     quantities = re.findall(r'i(\d+)\+q(\d+)', data)
                #     for index, quantity in quantities:
                #         print(f"Index: {index}, Quantity: {quantity}")
                #         app.index += str(index) + ","  # 将 index 转换为字符串并加上逗号
                #         app.name += app.names[int(index)] + ","
                #         app.quantity += str(quantity) + ","
                #         app.success = "OK"
                #     app.index = app.index[:-1]  # 移除最后的逗号
                #     print(app.index)
                #     app.name = app.name[:-1]
                #     app.quantity = app.quantity[:-1]
                #     app.update_display()
                # elif frame_header == "harmful":
                #     app.index += 1
                # elif frame_header == "full":
                #     app.full_display()
                # else:
                #     print(f"未知的帧头: {frame_header}")


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


import numpy as np


def group_coordinates_by_threshold(coords, threshold=20):
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
        cam.open(i)
        if cam.isOpened():
            return cam, i
    raise Exception("Camera not found")

def yolo_process(queue_display,queue_receive, queue_transmit):

    # time.sleep(10)
    # while True:
    #     # 该部分处理为进行视觉识别算法，得到目标，将信息显示在屏幕上
    #     print("运行 YOLO 算法...")
    #     queue_display.put('harmful=!')
    #     #queue.put("full=!")
    #     time.sleep(3)  # 模拟 YOLO 处理
    #     # queue_display.put('success=!')
    #     # time.sleep(3)
    #     # queue.put('garbage=i2+q2+i3+q19+i1+q7!')

    model_path = "best.onnx"
    model = YOLOv8Seg(model_path)
    model_large_path = "large.onnx"
    model_large = YOLOv8Seg(model_large_path)
    cap, i = open_camera()
    # cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # 角度偏差量
    angle_error = 0
    # 垃圾轮数计数
    index_garbage = 200
    # 预热模型
    start_time = time.time()
    while time.time() - start_time < 100:
        ret, frame = cap.read()
        cls_, confs, _, angles, centers, image, areas = model(frame, conf_threshold=0.7, iou_threshold=0.5)
        cls_, confs, _, angles, centers, image, areas = model_large(frame, conf_threshold=0.7, iou_threshold=0.5)

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
                command = f'Tar='
                command_display = f''

                # 单垃圾分类
                if index_garbage <= 100:
                    count = 0
                    while count < 5:
                        count += 1
                        ret, frame = cap.read()
                        # 置信度逐级递减
                        conf_threshold = 0.75-count*0.05
                        start_time = time.time()
                        cls_, confs, _, angles, centers, image, areas = model(frame, conf_threshold=conf_threshold, iou_threshold=0.5)
                        print("本次耗时：", time.time()-start_time)
                        print("第一次", cls_, confs, angles, centers, areas)

                        if len(cls_) == 1:
                            # 继续识别一次，与上次作比较
                            ret, frame = cap.read()
                            time.sleep(0.1)
                            ret, frame = cap.read()
                            new_cls_, new_confs, _, new_angles, new_centers, new_image, new_areas = model(frame, conf_threshold=conf_threshold,
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
                                ret, frame = cap.read()
                                time.sleep(0.1)
                                ret, frame = cap.read()
                                large_cls_, large_confs, _, large_angles, large_centers, large_image, large_areas = model_large(frame,
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


                # 双垃圾分类
                elif index_garbage > 10:
                    sum_cls_ = []
                    sum_angles = []
                    sum_centers = []
                    sum_confs = []
                    sum_areas = []
                    # 统计结果值
                    count = 0
                    while count < 5:
                        count += 1
                        ret, frame = cap.read()
                        time.sleep(0.1)
                        ret, frame = cap.read()
                        # 置信度逐级递减
                        conf_threshold = 0.75 - count * 0.05
                        start_time = time.time()
                        cls_, confs, _, angles, centers, image, areas = model(frame, conf_threshold=conf_threshold,
                                                                              iou_threshold=0.5)
                        print("本次耗时：", time.time() - start_time)
                        print(f"第{count}次", cls_, confs, angles, centers, areas)
                        # 加入序列之中
                        sum_cls_.extend(cls_)
                        sum_angles.extend(angles)
                        sum_confs.extend(confs)
                        sum_areas.extend(areas)
                        sum_centers.extend(centers)
                    print("sum:",sum_cls_,sum_confs,sum_angles,sum_centers,sum_areas)
                    # 对统计结果进行处理
                    group_count, grouped_indices = group_coordinates_by_threshold(sum_centers)
                    print("数量", group_count,"索引",grouped_indices)
                    if group_count == 2:

                        final_cls_.append(sum_cls_[grouped_indices[0][0]])
                        final_cls_.append(sum_cls_[grouped_indices[1][0]])
                        final_centers.append(sum_centers[grouped_indices[0][0]])
                        final_centers.append(sum_centers[grouped_indices[1][0]])
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
                            for index in grouped_indices[group_index]:
                                final_cls_.extend(sum_cls_[index])
                                final_centers.extend(sum_centers[index])
                                final_angles.extend(sum_angles[index])
                                final_areas.extend(sum_areas[index])

                    # 少于两个，尝试用大模型补充几次结果
                    elif group_count < 2:
                        print("少于两个")
                        while count < 2:
                            count += 1
                            ret, frame = cap.read()
                            # 置信度逐级递减
                            conf_threshold = 0.75 - count * 0.05
                            start_time = time.time()
                            cls_, confs, _, angles, centers, image, areas = model_large(frame, conf_threshold=conf_threshold,
                                                                                  iou_threshold=0.5)
                            print("本次耗时：", time.time() - start_time)
                            print(f"第{count}次", cls_, confs, angles, centers, areas)
                            # 加入序列之中
                            sum_cls_.extend(cls_)
                            sum_angles.extend(angles)
                            sum_confs.extend(confs)
                            sum_areas.extend(areas)
                            sum_centers.extend(centers)
                        # 根据添加后的结果进行上述操作
                        if group_count == 2:
                            print("增添后正好两个")
                            final_cls_.append(sum_cls_[grouped_indices[0][0]])
                            final_cls_.append(sum_cls_[grouped_indices[1][0]])
                            final_centers.append(sum_centers[grouped_indices[0][0]])
                            final_centers.append(sum_centers[grouped_indices[1][0]])
                        # 此情况出现概率较小，选择平均置信度最高的或者次数与概率积的和
                        elif group_count > 2:
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
                                for index in grouped_indices[group_index]:
                                    final_cls_.extend(sum_cls_[index])
                                    final_centers.extend(sum_centers[index])
                                    final_angles.extend(sum_angles[index])
                                    final_areas.extend(sum_areas[index])

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
                print("final",final_cls_,final_confs,final_angles,final_centers,final_areas)
                # 单垃圾
                if index_garbage <= 100:
                    if (len(final_cls_)==1):
                        # 有害垃圾
                        if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                            command += f'q2!'
                            command_display += 'harmful=!'
                        # 可回收垃圾
                        elif final_cls_[0] == 5 or final_cls_[0]== 9:
                            command += f'q1!'
                            command_display += 'recycle=!'
                        # 厨余垃圾
                        elif final_cls_[0] == 3 or final_cls_[0] == 7:
                            command += f'q3!'
                            command_display += 'kitchen=!'
                        # 其他垃圾
                        elif final_cls_[0] or final_cls_[0] == 6:
                            command += f'q4!'
                            command_display += 'other=!'
                    # 未成功识别
                    else:
                        print()
                #双垃圾
                else:
                    # 最好结果
                    if (len(final_cls_) == 2):
                         # 两个种类相同，倾倒即可
                        if final_cls_[0] ==final_cls_[1]:
                            # 有害垃圾
                            if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                                command += f'q2!'
                                command_display += 'harmful=!'
                            # 可回收垃圾
                            elif final_cls_[0] == 5 or final_cls_[0] == 9:
                                command += f'q1!'
                                command_display += 'recycle=!'
                            # 厨余垃圾
                            elif final_cls_[0] == 3 or final_cls_[0] == 7:
                                command += f'q3!'
                                command_display += 'kitchen=!'
                            # 其他垃圾
                            elif final_cls_[0] or final_cls_[0] == 6:
                                command += f'q4!'
                                command_display += 'other=!'
                        # 种类不同，先夹后倾倒
                        else:
                            if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                                command += f'j2x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                                command_display += 'harmful=!'
                            # 可回收垃圾
                            elif final_cls_[0] == 5 or final_cls_[0] == 9:
                                command += f'j1x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                                command_display += 'recycle=!'
                            # 厨余垃圾
                            elif final_cls_[0] == 3 or final_cls_[0] == 7:
                                command += f'j3x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                                command_display += 'kitchen=!'
                            # 其他垃圾
                            elif final_cls_[0] == 4 or final_cls_[0] == 6:
                                command += f'j4x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                                command_display += 'other=!'
                    # 夹取一个，剩下的 选择一个不同的垃圾倾倒
                    elif (len(final_cls_) == 1):
                        # 有害垃圾
                        if final_cls_[0] == 1 or final_cls_[0] == 2 or final_cls_[0] == 8:
                            command += f'j2x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                            command_display += 'harmful=!'
                        # 可回收垃圾
                        elif final_cls_[0] == 5 or final_cls_[0] == 9:
                            command += f'j1x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                            command_display += 'recycle=!'
                        # 厨余垃圾
                        elif final_cls_[0] == 3 or final_cls_[0] == 7:
                            command += f'j3x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                            command_display += 'kitchen=!'
                        # 其他垃圾
                        elif final_cls_[0] == 4 or final_cls_[0] == 6:
                            command += f'j4x{final_centers[0][0]}y{final_centers[0][1]}a{final_angles[0]-angle_error}!'
                            command_display += 'other=!'
                    # 完蛋，一个没识别出来，随机倾倒吧
                    elif (len(final_cls_) == 0):
                        command += f'q4!'
                        command_display += 'other=!'
                        print()
                # 处理未成功识的情识别
                if (command == ""):
                    print()

                # 信息发送到其他进程
                if (command != ""):
                    queue_transmit.put(command)
                    print(command)
                if (command_display != ""):
                    queue_display.put(command_display)
                    print(command_display)
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
    ser_ttyAMA4.flushInput()

    while ser_ttyAMA4.in_waiting == 0:
        ser_ttyAMA4.flushInput()
        ser_ttyAMA4.write(com.encode('ascii'))
        time.sleep(0.03)
        serial_cnt += 1

        if serial_cnt > 5:
            serial_cnt = 0
            print("serial_cnt=", serial_cnt)
            break


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

def serial_process(queue_receive,queue_transmit,queue_display_ser):
    #握手多次发送

    while True:
        time.sleep(5)
        queue_receive.put("detect")
    # 创建串口对象
    port = '/dev/ttyTHS1'  # 替换为你的串口号
    baudrate = 115200
    timeout = 1
    ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)

    buffer = ""

    while True:
        while True:
            try:
                if ser.in_waiting > 0:  # 检查是否有数据等待读取
                    # 读取一行数据并解码
                    try:
                        received_data = ser.readline().decode('ascii').strip()
                        print("received_data", received_data)
                        buffer += received_data  # 将接收到的数据添加到缓冲区

                        # 假设数据以特定标识符结束（例如"\n"）
                        if '!' in buffer:
                            messages = buffer.split('!')  # 根据标识符分割消息
                            for message in messages:
                                if message:  # 确保消息不为空
                                    print(f"接收到的数据: {message}")
                                    if message == "detect":  # 替换为实际的条件
                                        print("已发现有垃圾丢下，准备识别")
                                        queue_receive.put("detect")
                                        #延迟清串口
                                        time.sleep(0.2)
                                        ser.flush()
                                    # 满载
                                    elif message == "full":
                                        queue_display_ser.put("full=!")
                                    # 动作完成
                                    elif message == "success":
                                        queue_display_ser.put("success=!")
                            buffer = ""  # 清空缓冲区
                    except UnicodeDecodeError:
                        # 如果解码失败，处理异常
                        print("Decoding error: received data contains invalid ASCII characters.")

            except OSError as e:
                print(f"OSError occurred: {e}")
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser.close()
                ser = serial.Serial(port, baudrate, timeout=timeout)
            except serial.SerialException as e:
                print(f"SerialException occurred: {e}")
                print("Attempting to reinitialize the serial port...")
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser.close()
                ser = serial.Serial(port, baudrate, timeout=timeout)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser.close()
                ser = serial.Serial(port, baudrate, timeout=timeout)

        if not queue_transmit.empty():
            data_to_send = queue_transmit.get()
            try:
                uart_transition(data_to_send.encode('ascii'), ser)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(0.2)  # 程序暂停一秒后重试
                ser.close()
                ser = serial.Serial(port, baudrate, timeout=timeout)
                uart_transition(data_to_send.encode('ascii'), ser)
            print(f"发送的数据: {data_to_send}")
        time.sleep(0.1)


if __name__ == '__main__':

    queue_display = multiprocessing.Queue()
    queue_receive = multiprocessing.Queue()
    queue_transmit = multiprocessing.Queue()
    queue_display_ser = multiprocessing.Queue()

    display_proc = multiprocessing.Process(target=display_process, args=(queue_display, queue_display_ser))
    serial_proc = multiprocessing.Process(target=serial_process, args=(queue_receive, queue_transmit, queue_display_ser))
    main_proc = multiprocessing.Process(target=yolo_process, args=(queue_display, queue_receive, queue_transmit))

    display_proc.start()
    main_proc.start()
    serial_proc.start()

    display_proc.join()
    main_proc.join()
    serial_proc.join()
