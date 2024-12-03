# import matplotlib.font_manager
#
# # 获取所有可用的字体
# font_paths = matplotlib.font_manager.findSystemFonts()
# print(font_paths)
# import tkinter as tk
# import tkinter.font as tkfont
#
# # 创建 Tkinter 窗口
# root = tk.Tk()
#
# # 手动指定一个字体，例如 'Arial'
# font = tkfont.Font(family='Arial', size=12)
# print(f"Font details: {font.actual()}")
#
# root.mainloop()



import tkinter.font as tkfont

# 获取所有字体
fonts = tkfont.families()
print(fonts)  # 输出所有可用的字体名称
