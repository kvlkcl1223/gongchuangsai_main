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
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        cls_, confs, masks, angles, centers,image = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )

        return cls_, confs, masks, angles, centers, image  # 类别 置信度 掩码 角度 中心 图像

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

            return x[..., 5], x[..., 4], masks, angles, centers, im0
        else:
            return [], [], [], [], [], im0
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
            (self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.ANTIALIAS)
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
    def check_queue():
        try:
            message = queue_display.get_nowait()  # 尝试获取消息
            # app.update_label(message)  # 更新标签
            app.flag_start = 0
            app.video_label.destroy()
            header_match = re.match(r'^(.*?)=(.*?)!$', message)
            if header_match:
                frame_header = header_match.group(1).strip()
                data = header_match.group(2).strip()
                if frame_header == "ok":
                    app.state = "success"
                    if app.last_frame_header == "harmful":
                        app.quantity_harmful += 1
                    elif app.last_frame_header == "recyclable":
                        app.quantity_recyclable += 1
                    elif app.last_frame_header == "kitchen":
                        app.quantity_kitchen += 1
                    elif app.last_frame_header == "other":
                        app.quantity_other += 1
                else:
                    app.index += 1
                    app.name = frame_header
                    app.state = "classifying"
                    app.last_frame_header = frame_header
                app.update_display()

            message = queue_display_ser.get_nowait()  # 尝试获取消息
            header_match = re.match(r'^(.*?)=(.*?)!$', message)
            if header_match:
                frame_header = header_match.group(1).strip()
                data = header_match.group(2).strip()
                if frame_header == "full":
                    app.full_display()
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


def yolo_process(queue_display,queue_receive, queue_transmit):


    while True:
        # 该部分处理为进行视觉识别算法，得到目标，将信息显示在屏幕上
        print("运行 YOLO 算法...")
        queue_display.put('harmful=!')
        #queue.put("full=!")
        time.sleep(3)  # 模拟 YOLO 处理
        queue_display.put('ok=!')
        # queue.put('garbage=i2+q2+i3+q19+i1+q7!')

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
    while True:
        frame, ret = cap.read()
        # 裁剪图像
        frame = extract_region(frame, points=[(446, 22), (824, 28), (818, 406), (446, 400)])
        cls_, confs, _, angles, centers, image = model(frame, conf_threshold=0.7, iou_threshold=0.5)
        # 先用夹子丢需要压缩的垃圾，再用夹子丢其他，最后直接倾倒
        if cls_ is not None:
            # 先清队列
            data_to_receive = queue_receive.get()
            for i in range(len(cls_)):
                command = f'Tar='
                command_display = f''
                if len(cls_)-i == 1 and cls_[i] != 5 and cls_[i] != 9:
                    # 有害垃圾
                    if cls_[i] == 1 or cls_[i] == 2 or cls_[i] == 8:
                        command += f'q1!'
                        command_display += 'harmful=!'
                    # 厨余垃圾
                    elif cls_[i] == 3 or cls_[i] == 7:
                        command += f'q3!'
                        command_display += 'kitchen=!'
                    # 其他垃圾
                    elif cls_[i] == 4:
                        command += f'q4!'
                        command_display += 'other=!'
                else:
                    # 丢需压缩垃圾即可回收垃圾
                    if cls_[i] == 5 or cls_[i] == 9:
                        command += f'j2x{centers[i][0]}y{centers[i][1]}a{angels[i]-0.0}!'
                        command_display += 'recyclable=!'
                    # 有害垃圾
                    elif cls_[i] == 1 or cls_[i] == 2 or cls_[i] == 8:
                        command += f'j1x{centers[i][0]}y{centers[i][1]}a{angels[i] - 0.0}!'
                        command_display += 'harmful=!'
                    # 厨余垃圾
                    elif cls_[i] == 3 or cls_[i] == 7:
                        command += f'j3x{centers[i][0]}y{centers[i][1]}a{angels[i] - 0.0}!'
                        command_display += 'kitchen=!'
                    # 其他垃圾
                    elif cls_[i] == 4:
                        command += f'j4x{centers[i][0]}y{centers[i][1]}a{angels[i] - 0.0}!'
                        command_display += 'other=!'
                queue_display.put(command)
                queue_transmit.put(command)
                while(not queue_receive.empty()):
                    time.sleep(0.1)
                data_to_receive = queue_receive.get()
def serial_process(queue_receive,queue_transmit,queue_display_ser):
    def uart_transition(com, ser_ttyAMA4):
        serial_cnt = 1  # 调用一次该程序
        while ser_ttyAMA4.in_waiting > 0:
            ser_ttyAMA4.read(ser_ttyAMA4.in_waiting)  # 读取并丢弃所有数据
        ser_ttyAMA4.flushInput()
        while ser_ttyAMA4.in_waiting == 0:
            ser_ttyAMA4.flushInput()
            ser_ttyAMA4.write(com.encode('ascii'))
            time.sleep(0.01)
            serial_cnt += 1

            if serial_cnt > 5:
                serial_cnt = 0
                print("serial_cnt=", serial_cnt)
                break
    # 创建串口对象
    port = '/dev/ttyTHS1'  # 替换为你的串口号
    baudrate = 115200
    timeout = 1
    ser = serial.Serial(port, baudrate, timeout=timeout)

    buffer = ""

    while True:
        if ser.in_waiting > 0:
            # 读取一行数据并解码
            try:
                received_data = ser.readline().decode('ascii').strip()
                buffer += received_data  # 将接收到的数据添加到缓冲区

                # 假设数据以特定标识符结束（例如"\n"）
                if '\n' in buffer:
                    messages = buffer.split('\n')  # 根据标识符分割消息
                    for message in messages:
                        if message:  # 确保消息不为空
                            print(f"接收到的数据: {message}")
                            if message == "some_condition":  # 替换为实际的条件
                                print("满足条件，执行某些操作。")
                                queue_receive.put(message)
                            elif message == "full":
                                queue_display_ser.put("full=!")
                    buffer = ""  # 清空缓冲区
            except UnicodeDecodeError:
                # 如果解码失败，处理异常
                print("Decoding error: received data contains invalid ASCII characters.")

        if not queue_transmit.empty():
            data_to_send = queue_transmit.get()
            ser.write(data_to_send.encode('ascii'))
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
