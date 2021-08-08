from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import argparse  
import time
from collections import deque
pts = deque(maxlen=36)




def pre_process(frame):
    frame = imutils.resize(frame, width=600,height=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv

def masking(frame):
    if frame is None:
        print("frame is None")
        raise IOError("Frame not valid")  
        exit()
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    mask = cv2.inRange(frame, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    return mask
    
    


def open_Video_Stream(file_path):
    
    vs = VideoStream(src=0).start()
    
    if(file_path==None):
        vs=cv2.VideoCapture(1)
        if not vs.isOpened():
            vs=cv2.VideoCapture(0)
        if not vs.isOpened():
            raise IOError("Cannot open video")
    else:
        vs=cv2.VideoCapture(file_path)
    time.sleep(2.0)
    return vs

def find_Contour(file_path=None):
    while True:
        vs=open_Video_Stream(file_path)
        frame = vs.read()
        frame = frame[1] #if args.get("video", False) else frame
        if frame is None:
            break
            
        frame_orig=frame.copy()
        frame=pre_process(frame)
        mask=masking(frame)
        
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
    
        p=None
        if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        #print(cnts)
        
        #p=None
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            p=cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 0), 2)
        # only proceed if the radius meets a minimum size
            if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
                p=cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 0), 2)
                cv2.circle(frame_orig, center, 5, (0, 0, 255), -1)
    
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
        
    
    
    
    
    

        cv2.imshow("Frame", frame_orig)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break
        
    vs.release()

    cv2.destroyAllWindows()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--Video", help = "Enter Path to Video",type=str)
    args = parser.parse_args()  
    print("Entered Main")
    find_Contour()
    
