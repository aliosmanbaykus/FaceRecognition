import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
import face_recognition
import os
import numpy as np
 
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.ok=False
       
        
        path = VideoCapture().path
        images = []
        self.classNames = []
        mylist = os.listdir(path)
        print(mylist)

        for cl in mylist:
            curimg= cv2.imread(f'{path}/{cl}')
            images.append(curimg)
            self.classNames.append(os.path.splitext(cl)[0])
        print(self.classNames)

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

        self.encodeListKnown = findEncodings(images)

 
        #timer
        self.timer=ElapsedTimeClock(self.window)
 
        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)
 
        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
 
        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)
 
        #video control buttons
 
        self.btn_start=tk.Button(window, text='START', command=self.open_camera)
        self.btn_start.pack(side=tk.LEFT)
 
        self.btn_stop=tk.Button(window, text='STOP', command=self.close_camera)
        self.btn_stop.pack(side=tk.LEFT)
 
        # quit button
        self.btn_quit=tk.Button(window, text='QUIT', command=quit)
        self.btn_quit.pack(side=tk.LEFT)
 
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()
 
        self.window.mainloop()
 
    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()
 
        if ret:
            (h, w) = frame.shape[:2]
            cv2.rectangle(frame, (int(w*0.1), int(h*0.25)), (int(w-w*0.1), int(h-h*0.25) ),
                          (0, 255, 0))
            cv2.imwrite("frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
 
    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")
 
 
 
    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")
 
    
    def update(self):
 
        # Get a frame from the video source
        

        ret, frame = self.vid.get_frame()
        


        if self.ok:
            imgS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)    
            
          
            for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                
                 # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
 
        if ret:
            imgS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)    

            for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                
                 # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay,self.update)
 
 
class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        # Command Line Parser
        args=CommandLineParser().args
        
        #Take path of images
        self.path=args.path[0]

 
        #create videowriter
 
        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }
 
        self.fourcc=VIDEO_TYPE[args.type[0]]
 
        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,10,res)
 
        #set video sourec width and height
        self.vid.set(3,res[0])
        self.vid.set(4,res[1])
 
        # Get video source width and height
        self.width,self.height=res
 
 
    # To get frames
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)
    
   

 
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            cv2.destroyAllWindows()
    

 
 
class ElapsedTimeClock:
    def __init__(self,window):
        self.T=tk.Label(window,text='00:00:00',font=('times', 20, 'bold'), bg='green')
        self.T.pack(fill=tk.BOTH, expand=1)
        self.elapsedTime=dt.datetime(1,1,1)
        self.running=0
        self.lastTime=''
        t = time.localtime()
        self.zeroTime = dt.timedelta(hours=t[3], minutes=t[4], seconds=t[5])
        # self.tick()
 
    def tick(self):
        # get the current local time from the PC
        self.now = dt.datetime(1, 1, 1).now()
        self.elapsedTime = self.now - self.zeroTime
        self.time2 = self.elapsedTime.strftime('%H:%M:%S')
        # if time string has changed, update it
        if self.time2 != self.lastTime:
            self.lastTime = self.time2
            self.T.config(text=self.time2)
        # calls itself every 200 milliseconds
        # to update the time display as needed
        # could use >200 ms, but display gets jerky
        self.updwin=self.T.after(100, self.tick)
 
 
    def start(self):
            if not self.running:
                self.zeroTime=dt.datetime(1, 1, 1).now()-self.elapsedTime
                self.tick()
                self.running=1
 
    def stop(self):
            if self.running:
                self.T.after_cancel(self.updwin)
                self.elapsedTime=dt.datetime(1, 1, 1).now()-self.zeroTime
                self.time2=self.elapsedTime
                self.running=0
 
 
class CommandLineParser:
    
    def __init__(self):
 
        # Create object of the Argument Parser
        parser=argparse.ArgumentParser(description='Script to record videos')
 
        # Create a group for requirement
        # for now no required arguments
        # required_arguments=parser.add_argument_group('Required command line arguments')
 
        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')
 
        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['480p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')
 
        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # It will take path of file
        parser.add_argument('--path', nargs=1, type=str, help='Enter path for images')
 
        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()
 
 
def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(),'Video Recorder')
 
main()