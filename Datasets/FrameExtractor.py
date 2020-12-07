import cv2
import math
import os


class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_speed_fps(self):
        return self.fps, self.n_frames

    def isOpened(self):
        return self.vid_cap.isOpened() 
       
    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')
        
    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')
        
        frame_cnt = 0
        img_cnt = 52900

        while self.vid_cap.isOpened():
            print(frame_cnt)
            success,image = self.vid_cap.read() 
            
            if not success:
                break
            
            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)  
                img_cnt += 1
                
            frame_cnt += 1
        
        self.vid_cap.release()
        cv2.destroyAllWindows()


video_path = '/home/simon/Desktop/Scene_Analysis/Datasets/Scene-1-JacksonTownSquare/Videos/'
dest_path = '/home/simon/Desktop/Scene_Analysis/Datasets/Scene-1-JacksonTownSquare/Images/Img-1/'

if not os.path.exists(video_path):
    print('Video path not found.')

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

Frames = FrameExtractor(video_path + 'Video-9.webm')
#Frames.get_n_images(every_x_frame=1000)

if not Frames.isOpened():
    print('Video is not opened.')
    assert(0)

print('speed: ', Frames.get_speed_fps()[0], ' fps.')
print('number of frames: ', Frames.get_speed_fps()[1], ' frames.')
Frames.extract_frames(every_x_frame=30, img_name='JacksonTownSquare', dest_path=dest_path)

print('Finishing ...')








