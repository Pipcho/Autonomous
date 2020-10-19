# import requests,json
# api_key="AIzaSyCeMboCnYgBBu2exuCvhTtgicXyXidvrhw"
# source=input()
# dest=input()
# url="https://maps.googleapis.com/maps/api/distancematrix/json?"
# r=requests.get(url +"origins="+ source +"&destinations="+ dest +"&key="+ api_key)
# x=r.json()
# print(x)


# -------------------------------------------------------------
# for t in range(int(input())):
#     a=[i for i in range(int(input()))]
#     k=1
#     b=[0 for i in range(len(a))]
#     while len(a)!=0:
#         for i in range(k):
#             a.append(a.pop(0))
#         b[a[0]]=k
#         a.pop(0)
#         k=k+1
#     print(*b)
# ---------------------------------------------------------------------
# for i in range(int(input())):
#     n = int(input())
#     a = list(map(int, input().split()))
#     min_new = 0
#     if len(a)==1:
#         print(a[0]);8
#     else:
#         while len(a) != 1:
#             min1 = min(a)
#             a.remove(min1)
#             min2 = min(a)
#             a.remove(min2)
#             a.append(min1 + min2)
#             min_new = min1 + min2 + min_new
# print(min_new)
# -------------------------------------------------------------
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from useloop import ServoMotorControl
import math
import time
# import serial

#
# usbport = 'COM1'
# ser = serial.Serial(usbport, 9600, timeout=1)


servo_angle = 90

# image2=cv2.imread("test img 1.png",0)

# image=cv2.imread("solidWhiteCurve.jpg")
# -------------------------------------------
# cap = cv2.VideoCapture('solidWhiteRight.mp4')

# -------------------------------
cap = cv2.VideoCapture("project_video_output.mp4")
# --------------------------------------------
# cap = cv2.VideoCapture("testvideo2.mp4")
# ----------------------------------------
# loop through until entire video file is played
while (cap.isOpened()):

    # read video frame & show on screen
    ret, image = cap.read()


    # def object_brake(image1, image2):
    #     h, w = image2.shape
    #     image = cv2.cvtCol or(image1, cv2.COLOR_RGB2GRAY)
    #     result = cv2.matchTemplate(image, image2, cv2.TM_CCOEFF_NORMED)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    #     top_left = max_loc
    #     bottom_right = (top_left[0] + w, top_left[1] + h)
    #     cv2.rectangle(image1, top_left, bottom_right, [0,0,255], 2)
    #
    # object_brake(image,image2)

    # optional

    # def cannyreturn(frame):
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     lower_blue = np.array([0, 0, 225], dtype=np.uint8)
    #     upper_blue = np.array([255, 30, 255], dtype=np.uint8)
    #     mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #
    #     edges = cv2.Canny(mask, 200, 400)
    #
    #     return edges

    def region_interest(image):
        # ----------------------------
        # SOLIDWIGHTRIGHT
        # bottom_left = (130, image.shape[0])
        # bottom_right = (image.shape[1], image.shape[0])
        # top_middle=(490,300)
        # top_left = (373, 351)
        # top_right = (573, 351)
        # ---------------------------------------------------
        # CHALLENGE
        bottom_left=(110,image.shape[0])
        bottom_right =(image.shape[1],image.shape[0])
        top_left =(541,464)
        top_right =(805,464)
        # ------------------------------------------------------
        # testvideo
        # bottom_left = (234, image.shape[0])
        # bottom_right = (915, image.shape[0])
        # top_left = (505,585)
        # top_right = (716,585)

        # ----------------------------------------------------------
        triangle = np.array([[bottom_left, bottom_right, top_right, top_left]])
        # print(type(triangle))
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, triangle, [255, 255, 255])
        # cv2.imshow('mask_result',mask)##############################################
        return mask


    def cannyreturn(image):
        img_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_bw = cv2.GaussianBlur(img_bw, (7, 7), 0)
        cannyimg = cv2.Canny(img_bw, 120, 200)
        # print(image.shape)
        # cv2.imshow("canny_img",cannyimg)############################################################
        return cannyimg


    final_canny_img = cv2.bitwise_and(cannyreturn(image), region_interest(cannyreturn(image)))
    # cv2.imshow("final_canny",final_canny_img)########################################################q
    lines = cv2.HoughLinesP(final_canny_img, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=150)


    # for line in lines:
    #     x1,y1,x2,y2=line[0][:]
    #     cv2.line(image,(x1,y1),(x2,y2),[255,0,0],thickness=2)
    # cv2.imshow("original_image",image)

    def make_points(frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 0.5)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]


    def average_slope_intercept(frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1 / 2
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                cv2.line(frame, (x1, y1), (x2, y2), [255, 0, 0], thickness=2)

                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(make_points(frame, right_fit_average))

        # print(lane_lines)
        return lane_lines


    lane_lines = average_slope_intercept(image, lines)

    # cv2.imshow("original_image", image)

    height, width, _ = image.shape
    # cv2.imshow('IMAGE',image)
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height * 0.5)
    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        mid = int(width / 2)
        x_offset = x2 - x1
        y_offset = int(height / 2)
    else:
        print("break is applied")
        continue

    steering_angle_in_redian = math.atan(x_offset / y_offset)
    steering_angle_in_deg = int(steering_angle_in_redian * 180 / math.pi)
    servo_motor_input_angle = steering_angle_in_deg + 90
    print('steerin angle', steering_angle_in_deg)
    print("servo motor angle", servo_motor_input_angle)


    # def braking_control(steering_angle_deg):
    #     if abs(steering_angle_in_deg)>20:
    #         time.sleep(0.8)
    #     elif 15<abs(steering_angle_in_deg)<20:
    #         time.sleep(0.6)
    #     elif 10<abs(steering_angle_in_deg)<15:
    #         time.sleep(0.4)
    #     elif 5<abs(steering_angle_in_deg)<10:
    #         time.sleep(0.2)
    #     else:
    #         pass
    # braking_control(steering_angle_in_deg)

    def stabilize_steering_angle(
            curr_steering_angle,
            new_steering_angle,
            num_of_lane_lines,
            max_angle_deviation_two_lines=5,
            max_angle_deviation_one_lane=2):
        """
        Using last steering angle to stabilize the steering angle
        if new angle is too different from current angle,
        only turn by max_angle_deviation degrees
        """
        if num_of_lane_lines == 2:
            # if both lane lines detected, then we can deviate more
            max_angle_deviation = max_angle_deviation_two_lines
        else:
            # if only one lane detected, don't deviate too much
            max_angle_deviation = max_angle_deviation_one_lane

        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        return stabilized_steering_angle


    # ServoMotorControl(steering_angle_in_deg).func(steering_angle_in_deg)
    # x = str(servo_angle) + ","
    # ser.write(str.encode(x))
    # arduino.write(pack(">B", 180))
    servo_angle = stabilize_steering_angle(servo_angle, servo_motor_input_angle, len(lane_lines))
    print("stablized servo angle", servo_angle)
    if servo_angle < 90:
        cv2.putText(image, f"TURN LEFT BY {servo_angle} deg", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255])
    elif servo_angle > 90:
        cv2.putText(image, f"TURN RIGHT BY {servo_angle} deg", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255])
    else:
        cv2.putText(image, f"GO STRAIGHT", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255])

    cv2.line(image, (mid, int(height)), (mid, int(height * 0.75)), [255,0,0], thickness=2)
    cv2.imshow("original_image", image)

    # plt.imshow(image)
    # plt.show()
    # cv2.waitKey(0)
    # time.sleep(0.2)
    # time.sleep(0.3)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# plt.show()
cap.release()
cv2.destroyAllWindows()
