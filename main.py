# -*- coding: utf-8 -*-
import time
import cv2
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.clock import Clock

from faceDetector import FaceDetector
import json

Builder.load_file("main.kv")

Window.size = (1000, 600)

JSON_DATA_PATH = "data/face_data.json"

'''
顔データ表示部
'''


class ShowData(BoxLayout):

    def __init__(self, **kwargs):
        super(ShowData, self).__init__(**kwargs)

    def set_text(self, t):
        self.ids.face_json.text = json.dumps(t)


'''
操作部
'''


class Editor(BoxLayout):
    def __init__(self, **kwargs):

        super(Editor, self).__init__(**kwargs)
        self.cam_show = True
        self.detector = FaceDetector()  # face detector
        self.img1 = Image(size_hint=(1.0, 1.0))
        self.showData = ShowData()  # データ表示部

    # カメラをスタートする
    def start_cam(self, src=0):
        self._cap = cv2.VideoCapture(src)
        while not self._cap.isOpened():
            pass
        self.cam_show = True
        print('cam started!')
        self.count = 0
        self.fps = 0
        self.start_time = time.time()

    # カメラを閉じる
    def close_cam(self):
        self.cam_show = False
        self._cap.release()

    # アップデート関数
    # 毎フレームごとに通る関数
    def update(self, dt):
        if self._cap.isOpened() is False:
            return

        _, img = self._cap.read()
        img = self.detect_face(img)
        img = cv2.flip(img, 0)
        # GUI（kivy)にtextureにセット
        texture1 = Texture.create(
            size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(img.tostring(),
                             colorfmt='bgr', bufferfmt='ubyte')
        # データ表示部にテキストを背とする
        self.showData.set_text(self.detector.known_face_names)
        # show the results (or not)
        self.img1.texture = texture1 if self.cam_show else None

        # counter, fps calculation
        self.count += 1
        if self.count == 10:
            self.calc_fps()

    # face detection
    def detect_face(self, img):
        faceData = self.detector.analyze_faces_in_image(img)
        show_img = cv2.flip(img, 0)
        show_img = self.detector.draw_rect(img, faceData)  # TODO: たぶんここが遅い
        return show_img

    def calc_fps(self):
        self.fps = 10 / (time.time() - self.start_time)
        self.count = 0
        self.start_time = time.time()

    def connec(self, ip, port):
        ip = ip.text
        port = int(port.text)

    def cam_load(self, src):
        print('cam load', src.text)
        self.close_cam()
        if src.text.find('mp4') != -1 or src.text.find('m4a') != -1:
            self.start_cam(src.text)
        else:
            self.start_cam(int(src.text))

    def toggle_cam_show(self, cb):
        print(cb.active)
        self.cam_show = cb.active


class CvCamera(App):
    def build(self):
        # UI
        layout1 = BoxLayout(orientation='horizontal', size_hint=(0.5, 0.5))
        self.editor = Editor()
        layout1.add_widget(self.editor)
        layout2 = BoxLayout(orientation='vertical', size_hint=(0.5, 1.0))
        layout2.add_widget(layout1)
        layout2.add_widget(self.editor.img1)
        layout = BoxLayout(orientation="vertical", size_hint=(1.0, 1.0))
        layout.add_widget(layout2)
        layout2.add_widget(self.editor.showData)

        self.editor.start_cam()
        self.editor.detector.load_face(JSON_DATA_PATH)

        Clock.schedule_interval(self.editor.update, 1.0/30.0)

        return layout

    def on_stop(self):
        self.editor.close_cam()
        self.editor.detector.save_to_json()


if __name__ == '__main__':
    CvCamera().run()
