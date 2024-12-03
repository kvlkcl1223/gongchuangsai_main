# import matplotlib.font_manager
#
# # 获取所有可用的字体
# font_paths = matplotlib.font_manager.findSystemFonts()
# print(font_paths)

import tkinter as tk
import tkinter.font as tkfont

root = tk.Tk()
font = tkfont.nametofont(root.option_get('font', 'TkDefaultFont'))
print(font.actual())  # 打印当前字体信息
