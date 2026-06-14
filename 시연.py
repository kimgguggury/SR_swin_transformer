import cv2
import numpy as np
import tkinter as tk

img_left = cv2.imread("/home/vclab2080ti/Downloads/projects/DRCT_new/results/compare/best_bicubic.png")
img_right = cv2.imread("/home/vclab2080ti/Downloads/projects/DRCT_new/results/compare/best_sr.png")

def resize_keep_aspect(img, target_h):
    h0, w0 = img.shape[:2]
    new_w = int(w0 * target_h / h0)
    return cv2.resize(img, (new_w, target_h))

# 실제 모니터 해상도 얻기
root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

# 높이 맞추기
target_h = min(img_left.shape[0], img_right.shape[0])
img_left = resize_keep_aspect(img_left, target_h)
img_right = resize_keep_aspect(img_right, target_h)

gap = 20  # 사이 간격, 10~30 정도로 조절
separator = np.zeros((target_h, gap, 3), dtype=np.uint8)

compare = np.hstack((img_left, separator, img_right))

# 비교 이미지를 적당히 크게
ch, cw = compare.shape[:2]
scale = min((screen_w * 0.8) / cw, (screen_h * 0.7) / ch)
new_w = int(cw * scale)
new_h = int(ch * scale)

compare_big = cv2.resize(compare, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# 모니터 전체 크기의 검은 캔버스
canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

# 가운데 배치
x_offset = (screen_w - new_w) // 2
y_offset = (screen_h - new_h) // 2 - 20

canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = compare_big

# 아래 라벨
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 3
label_y = y_offset + new_h + 50

left_center_x = x_offset + new_w // 4
right_center_x = x_offset + (3 * new_w) // 4

(text_w1, _), _ = cv2.getTextSize("Bicubic", font, font_scale, thickness)
(text_w2, _), _ = cv2.getTextSize("My Model", font, font_scale, thickness)

cv2.putText(canvas, "Bicubic",
            (left_center_x - text_w1 // 2, label_y),
            font, font_scale, (255, 255, 255), thickness)

cv2.putText(canvas, "My Model",
            (right_center_x - text_w2 // 2, label_y),
            font, font_scale, (255, 255, 255), thickness)

cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Comparison", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Comparison", canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()