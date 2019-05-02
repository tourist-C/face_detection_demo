#!/usr/bin/env python
# coding: utf-8

# # Face Detection Demo Code

# ## Core Logic:
# 1. Start Web Cam
# -  Detect faces using Haar Cascade
#     - return Region of Interest (ROI)
# -  Recognize attributes with pretrained CNN
#     - enlarge ROI for CNN input, since CNN is trained on the whole head, not just the face
#     - resize ROI into (200,200) to suit CNN input layers
# -  Display result in real time








import cv2 as cv
import numpy as np
import pandas as pd
import os
from keras.models import load_model





def get_info(cap):
    """
    info: get basic video info, e.g. lenght, FPS
    input: capture object from cv.Videocapture
    returns: basic info of the video file
    """
    info_to_get = [cv.CAP_PROP_FRAME_COUNT,
                    cv.CAP_PROP_FPS,
                    cv.CAP_PROP_FRAME_WIDTH,
                    cv.CAP_PROP_FRAME_HEIGHT,
                    cv.CAP_PROP_FOURCC,
                    cv.CAP_PROP_POS_AVI_RATIO,
                   ]
    
    info = [cap.get(i) for i in info_to_get]
    
    # duration = 1/FPS* nb frames
    duration = info[0]*1/info[1]
    info.append(duration)
    
    # define dictionary keys
    info_text = """FRAME,
    FPS,
    WIDTH,
    HEIGHT,
    FOURCC,
    POS,
    DURATION"""
    info_label = [i.strip() for i in info_text.split(",\n")]
    info_dict =  dict(zip(info_label,info))
    
    return info_dict





def view_video_plus(input_video,scale=0.5):
    """
    info: Display the video in a new frame
    input: file path of video file
    returns: basic info of video
    """
    cap = cv.VideoCapture(input_video)
    info = get_info(cap)
    
    # text style config
    color = (0,0,255) #BGR
    text_org = (0,int(info['HEIGHT']))
    font = cv.FONT_HERSHEY_SIMPLEX
    lineTHK = 2
    fontScale = 1
    
    # begining of loop
    c=1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            text = f"Frame:{c}/{int(info['FRAME'])}"
            cv.putText(frame,text,text_org,font,fontScale,color,lineTHK,cv.LINE_AA,bottomLeftOrigin=False)
            out = cv.resize(frame, None , None, scale, scale)
            cv.imshow(str(input_video),out)
            
            # wait key specify how many miliseconds per frame
            # e.g. 1000ms per frame => 1 frame per second
            # 30 fps = (1000/30**)-1
            c+=1
            if cv.waitKey(1000//FPS) & 0xFF == ord('q'): 
                break

    cap.release()
    cv.destroyAllWindows()
    return info





def export_to_jpgs(input_video,show_video=False,st_end=(100,200),export_dir="jpgs"):
    """
    info: simple utility to cut the video into frames, put those jpg into a new folder, start folder when completed
    input: filepath of the video
    output: jpgs
    """
    name_format = str(export_dir)+"\\"+str(input_video)+"_Frame_"
    try:
        os.makedirs(export_dir)
    except:
        pass

    # define start frame and end frame to cut
    frame_st, frame_fn = st_end

    c=0
    cap = cv.VideoCapture(input_video)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame_st<= c and c <frame_fn: 
            cv.imwrite(name_format+str(c)+".jpg",frame)
        c+=1
        
        if show_video:
            cv.imshow(str(video_raw_fp),frame)
            if cv.waitKey(1) and 0xFF == ord('q'):
                break
        if c > frame_fn:
            break
                
    cap.release()
    cv.destroyAllWindows()
    os.startfile(export_dir)





def save_video(input_video,output_video,shape=(1280,720),FPS=30):
    """
    info: save a video object
    input: filepath of input video
    output: saved video
    """
    
    # spent a lot of time figuring out the video encoding and format
    # output shape must be int and equal to the input video shape
    # working code is MJPG
    # working file extension is .avi
    
    codec = "MJPG"
    fourcc = cv.VideoWriter_fourcc(*codec)
    out = cv.VideoWriter(output_video,fourcc,FPS,shape)
    cap = cv.VideoCapture(input_video)
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break
        
    cap.release()
    out.release()





def save_cut_video(input_video,output_video,st_end=(10,20),shape=(1280,720),FPS=30):
    """
    info: save a video object between start time and end time
    input: filepath of input video start time and end time
    output: saved video
    """

    cap = cv.VideoCapture(input_video)
    info = get_info(cap)
    shape = (int(info['WIDTH']),int(info['HEIGHT']))
    FPS = info['FPS']
             
    codec = "MJPG"
    fourcc = cv.VideoWriter_fourcc(*codec)
    out = cv.VideoWriter(output_video,fourcc,FPS,shape)
    
    # convert time to frame index
    st_frame,end_frame = st_end[0]*FPS, st_end[1]*FPS
    
    c = 0
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            if st_frame < c and c <= end_frame:
                out.write(frame)
            if c > end_frame: 
                break
        else:
            break
        c+=1
        
    cap.release()
    out.release()





def show_pic2(img,window_name="DISPLAY",pos=(0,0),scale=1):
    """
    use this if you want to show picture OUTSIDE the cv video capture while lopp
    """
    img_out = cv.resize(img,None,None,scale,scale)
    cv.imshow(window_name,img_out)
    x,y = pos
    cv.moveWindow(window_name,x,y)
    if cv.waitKey(0) & 0xFF == ord('q'): 
        cv.destroyAllWindows()





def show_pic(img,window_name="DISPLAY",pos=(0,0),scale=1):
    """
    use this if you want to show picture INSIDE the cv video capture while lopp
    """
    img_out = cv.resize(img,None,None,scale,scale)
    cv.imshow(window_name,img_out)
    x,y = pos
    cv.moveWindow(window_name,x,y)


# --------------------------------------------------------------------------------

# # Main




# loading pretrained CNN model and labels detector
model =  load_model('CNN_model\\faceattr_model.h5')
df = pd.read_csv('CNN_model\\label_new.csv')
labels = df.columns[2:]

# load face detection haar cascade
detect_cascade = cv.CascadeClassifier('haar_cascades\\haarcascade_frontalface_default.xml')

# start webcam
cap = cv.VideoCapture(0)
info = get_info(cap)

# text style config
color = (0,0,255) #BGR
text_org = (0,int(info['HEIGHT']))
font = cv.FONT_HERSHEY_SIMPLEX
lineTHK = 2
fontScale = 1
FPS = int(info['FPS'])

c=1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    else:
        # convert frame to gray for faster calculation
        frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        # haar cascade block
        sample_rate = 1
        if c%sample_rate == 0: #detect once every 10 frames
            ROIs = detect_cascade.detectMultiScale(frame_gray,1.1,7)
                 
            # skip to next frame if no face is detected
            if len(ROIs)==0:
                pass
            else:
                # draw ROI boxes
                for (x,y,w,h) in ROIs:
                    # draw rectangle of detected face
                    cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    
                    # enlarge the ROI for input of the CNN detector
                    padding_ratio = 0.3
                    head = frame[y-int(padding_ratio*h):y+h+int(padding_ratio*h),
                                 x-int(padding_ratio*w):x+w+int(padding_ratio*w)]
                    
                    
                    # for debugging, show the head frame for CNN model input
#                     bbox = [x-padding_ratio*w , h+padding_ratio*h , x+(1+padding_ratio)*w , y+(1+padding_ratio)*h]
#                     bbox = [int(i) for i in bbox]
#                     pt1,pt2 = tuple(bbox[0:2]) , tuple(bbox[2:4])
#                     cv.rectangle(frame,pt1,pt2,(0,0,255),2)
                    
                    # exception handling to avoid video from crashing when no head is detected or the input dims are wrong
                    try:
                        head = cv.resize(head, (200,200), interpolation = cv.INTER_AREA)
                        
#                         for debugging, show the head frame for CNN model input
                        show_pic(head,window_name=str(padding_ratio),pos=(1200,0))

                        # reshape input to fit NN input
                        head_input = head.reshape(1,*head.shape)
                        # predict
                        y_hat = model.predict(head_input)

                        results = []
                        for i in range(len(labels)):
                            if y_hat[0][i] > 0.8:
                                text = f"{labels[i]}:{y_hat[0][i]}"
                                results.append(text)
                                
                        # displaying prediction result in the video
                        label_text = str(results).replace(",","\n")
                        y0, dy = 50, 10
                        for i, line in enumerate(label_text.split('\n')):
                            y_ = y0 + i*dy
                            cv.putText(frame, line, (x, y_ ), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))

                    except:
                        pass
            
        # text block for showing the progress of video
        text = f"Frame:{c}/{int(info['FRAME'])}"
        cv.putText(frame,text,text_org,font,fontScale,
                   color,lineTHK,cv.LINE_AA,bottomLeftOrigin=False)
        show_pic(frame,window_name="FRAME",pos=(0,0),scale=1.5)        
        
        c+=1
        if cv.waitKey(int(1000/FPS)) & 0xFF == ord('q'): 
            break

cap.release()
cv.destroyAllWindows()

