import cv2
import numpy as np

# خواندن تصویر
image = cv2.imread('road.jpg')  # مطمئن شوید که فایل road.jpg در پوشه پروژه است
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# اعمال Gaussian Blur برای کاهش نویز
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# تشخیص لبه‌ها با استفاده از الگوریتم Canny
edges = cv2.Canny(blur, 50, 150)

# تعریف ناحیه موردنظر (ROI)
height, width = edges.shape
mask = np.zeros_like(edges)
triangle = np.array([
    [(0, height), (width, height), (width // 2, height // 2)]
])
cv2.fillPoly(mask, triangle, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# استفاده از Hough Transform برای شناسایی خطوط
lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)

# رسم خطوط روی تصویر اصلی
line_image = np.zeros_like(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

# ترکیب خطوط با تصویر اصلی
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

# نمایش تصویر
cv2.imshow('Detected Lines', combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()