import numpy as np
import time
import cv2
import mss
import numpy
from PIL import ImageGrab
from grabscreen import grab_screen
import win32gui, win32ui, win32con, win32api

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
# Set monitor size to capture
mon = (0, 40, 800, 640)
def screen_recordPIL():
    # set variables as global, that we could change them
    global fps, start_time
    # begin our loop
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # add one to fps
        fps+=1
        # calculate time difference
        TIME = time.time() - start_time
        # check if our 2 seconds passed
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            # set fps again to zero
            fps = 0
            # set start time to current time again
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
def main():
    last_time = time.time()
    while True:
        # 1920 windowed mode
        screen = grab_screen(region=(0,40,1920,1120))
        img = cv2.resize(screen,None,fx=0.4,fy=0.3)
        height,width,channels = img.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label,(x,y+30),font,1,(255,255,255),2)
        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
    
output_layers= net.getLayerNames()
output_layers= [output_layers[i - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))
main()

