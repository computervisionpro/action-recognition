
# importing libraries
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from functools import partial

import tensorflow as tf
from collections import deque
import numpy as np
import pickle
import cv2
import threading
import activity_predict_lite


# For managing multiple screens
class WindowManager(ScreenManager):
    pass


# first screen
class FirstWindow(Screen):

    def selected(self, filename):
        try:
            self.ids.display_txt.text = filename[0]
        except IndexError:
            # this will be executed on pressing go back
            self.ids.display_txt.text = ''


class SecondWindow(Screen):


    def on_enter(self, *args):
        """Triggered when you enter second screen"""

        self.stop = False
        s2 = self.manager.get_screen('first')
        print('Starting prediction! ')
        self.th = threading.Thread(target=self.recognize_action, args=[s2.ids.display_txt.text], daemon=True)
        self.th.start()

        #return super().on_enter(*args)

    def recognize_action(self, video_path):
        """Recognizing action by feeding it to model"""

        print("[INFO]: loading model and labels...")
        model = tf.lite.Interpreter(model_path = 'resnet-activity.tflite')
        model.allocate_tensors()
        lb = pickle.loads(open("activity-labels.pickle", "rb").read())


        # initialize the image mean for mean subtraction along with the
        # predictions queue
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        Q = deque(maxlen=10)
        video = video_path
        f = activity_predict_lite.pred(model, lb, video, mean, Q)

        vid = cv2.VideoCapture(video)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        
        for _ in range(total_frames):
            """Pass the encoded frames to browser"""
            if self.stop:
                break

            frame = next(f)
            
            # # call display_frame again as soon as it completes
            Clock.schedule_once(partial(self.display_frame, frame))

        cv2.destroyAllWindows()
        #print('Done')

    def display_frame(self, frame, dt):
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.vid.texture = image_texture


    def stop_thread(self):

        if self.th:
            self.stop = True
            s2 = self.manager.get_screen('first')
            s2.ids.select_file.selection = []
            #print('back pressed')



kv_file = Builder.load_file('activityrecognition.kv')



class ActivityRecognitionApp(App):
    def build(self):
        return kv_file


# launch app
ActivityRecognitionApp().run()
