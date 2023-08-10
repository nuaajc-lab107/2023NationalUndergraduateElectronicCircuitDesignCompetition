import sensor, image, time
from pyb import Servo
from pyb import UART
red_threshold  = (25, 100, 53, 13, -6, 59)
uart = UART(3,9600)
sensor.reset() # Initialize the camera sensor.
sensor.set_pixformat(sensor.RGB565) # use RGB565.
sensor.set_framesize(sensor.QQVGA) # use QQVGA for speed.
sensor.skip_frames(10) # Let new settings take affect.
sensor.set_auto_whitebal(False) # turn this off.
sensor.set_vflip(True)  #因摄像头安装方向倒转
sensor.set_hmirror(True)#需要水平、垂直翻转
clock = time.clock() # Tracks FPS.
x_error=0.0; #X轴偏差,就是实际坐标减去中点坐标得到的误差，然后摄像头旋转去减少这个误差即可，以下变量同理
y_error=0.0; #Y轴偏差
pan_error=0.0; #X轴偏差
tilt_error=0.0; #Y轴偏差
def find_max(blobs): #寻找图像中最大的红色色块
    max_size=0
    for blob in blobs:
        if blob[2]*blob[3] > max_size:
            max_blob=blob
            max_size = blob[2]*blob[3]
    return max_blob


while(True):
    clock.tick() # Track elapsed milliseconds between snapshots().
    img = sensor.snapshot() # Take a picture and return the image.
    img.draw_circle(79, 59, 1, color=(255, 0, 0), thickness=3) #描点函数,描绘中心点坐标
    blobs = img.find_blobs([red_threshold]) #开始寻找色块
    if blobs: #找到了色块了
        max_blob = find_max(blobs)
        pan_error = max_blob.cx()-img.width()/2 #这里计算x轴偏差值
        tilt_error = max_blob.cy()-img.height()/2  #这里计算y轴偏差值
        if(pan_error>0 or pan_error==0): #这里因为偏差值有负数，但是呢负数直接通过串口发送给单片机是不行的，所以当为负数时我就+100，也就是+50发的就是50，-50发的就是150
            x_error=pan_error
        if(pan_error<0):
            x_error=abs(pan_error)+100
        if(tilt_error>0 or tilt_error==0):
            y_error=tilt_error
        if(tilt_error<0):
            y_error=abs(tilt_error)+100
        #这里是画出色块的框
        img.draw_rectangle(max_blob.rect()) # rect
        img.draw_cross(max_blob.cx(), max_blob.cy()) # cx, cy
   #下边是串口发送数据代码，参考正点原子STM32的以\r\n结尾的协议
    thousands=int(x_error/1000%10) #取千位
    hundreds =int(x_error/100%10) #取百位
    tens =int(x_error/10%10) #取十位
    ones =int(x_error%10) #取个位
    #将数字转化成ASCLL码发出去
    thousands_str=str(thousands)
    hundreds_str=str(hundreds)
    tens_str=str(tens)
    ones_str=str(ones)
   # 同上
    thousands_y=int(y_error/1000%10)
    hundreds_y =int(y_error/100%10)
    tens_y =int(y_error/10%10)
    ones_y =int(y_error%10)
    thousands_str_y=str(thousands_y)
    hundreds_str_y=str(hundreds_y)
    tens_str_y=str(tens_y)
    ones_str_y=str(ones_y)
   #这里整和数据x轴和y轴偏差发送过去
    combined_str=thousands_str+hundreds_str+ tens_str+ ones_str+thousands_str_y+hundreds_str_y+ tens_str_y+ ones_str_y
    uart.write('@')
    uart.write(combined_str.encode())
    uart.write('\r\n') #发送结尾必须以这个结尾,待会儿STM32端会说明
    print("x_error: ", x_error)
    print("y_error: ", y_error)

