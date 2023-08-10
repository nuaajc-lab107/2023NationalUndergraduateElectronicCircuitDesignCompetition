import cv2
import numpy as np

#打开摄像头，根据自己的设备改ID号！！！
cap = cv2.VideoCapture(0)
success,img = cap.read()
print(img.shape)

while True:
    success,img = cap.read()
    # 获取图片的宽度和高度
    height, width, _ = img.shape
    # 定义裁剪区域
    start_row, start_col = int(height * 0), int(width * 0.1)
    end_row, end_col = int(height * 0.8), int(width * 0.8)
    img=img[start_row:end_row, start_col:end_col]

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色在HSV空间中的范围42, 100, 23, 85, -20, 62
    # 请注意，这可能需要根据你的具体情况进行调整
    # 颜色识别(红色)，过滤红色区域
    lower_red1 = np.array([0, 43, 46])  # 红色阈值下界
    higher_red1 = np.array([10, 255, 255])  # 红色阈值上界
    mask_red1 = cv2.inRange(hsv, lower_red1, higher_red1)
    lower_red2 = np.array([156, 43, 46])  # 红色阈值下界
    higher_red2 = np.array([180, 255, 255])  # 红色阈值上界
    mask_red2 = cv2.inRange(hsv, lower_red2, higher_red2)
    mask = cv2.add(mask_red1, mask_red2)  # 拼接过滤后的mask

    # 查找掩膜中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选出最大的轮廓，我们假设这就是激光点
    # 你可以根据需要进行更复杂的筛选
    point=(5,5)
    if len(contours)>0:
        laser_contour = max(contours, key=cv2.contourArea)
        point=np.mean(laser_contour,axis=0)
        point=point[0]
    # print(con)
        # print("point:",point)
        # 画出激光点
        cv2.drawContours(img, [laser_contour], -1, (255,0,0), 3)


    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=cv2.medianBlur(img,15)
    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)
    # edges = cv2.GaussianBlur(edges, (5,5), 0)
    # 寻找轮廓
    contours, hierarchy  = cv2.findContours(edges.copy(), cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    # contours1 = max(contours, key=cv2.contourArea)
    # print(len(contours))
    # 存储所有的矩形和对应的面积
    area_threshold=2000
    rectangles = []
    areas = []
    for cnt in contours:
        # 计算轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 如果多边形有四个顶点，我们假定它是矩形
        # approx存储了四个顶点的坐标
        if len(approx) == 4:
            # 计算矩形的面积并存储
            area = cv2.contourArea(approx)
            if area>area_threshold:
                rectangles.append(approx)
                areas.append(area)

    
    filtered_rectangles = []
    filtered_areas = []


    # 将矩形按面积从大到小排序
    sorted_indices = np.argsort(areas)[::-1]
    # 找出面积第二大的矩形
    if len(rectangles)>=2:
        largest_rectangle = rectangles[sorted_indices[0]]
        second_largest_rectangle = rectangles[sorted_indices[-1]]
        # 在图像上绘制该矩形
        cv2.drawContours(img, [largest_rectangle], 0, (0, 0, 255), 2)
        cv2.drawContours(img, [second_largest_rectangle], 0, (0, 255, 0), 2)
    
        result = cv2.pointPolygonTest(largest_rectangle, tuple(point), True)
        result1 = cv2.pointPolygonTest(second_largest_rectangle, tuple(point), True)
        if result>0 and result1>0:
            print("res:",1)
            cv2.putText(img, "inside", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 1)
        elif result>0 and result1<0:
            cv2.putText(img, "between", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 1)
            print("res:",2)
        else:
            cv2.putText(img, "outside", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 1)
            print("res:",3)
    cv2.imshow('Image with rectangle contours', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()


