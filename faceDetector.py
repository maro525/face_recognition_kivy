#!/usr/bin/env python
# -*- coding: utf-8 -*-

import face_recognition
import cv2
import os
import numpy as np
import json

FACE_NAME_JSON = "data/face_name.json"


class FaceDetector:
    def __init__(self):
        self.known_faces = {}
        self.known_faces["faces"] = []
        self.known_face_names = {}
        self.known_face_names["names"] = []
        self.tolerance = 0.6

    # jsonファイルから既知の顔のデータを読み込む
    def load_face(self, src):
        try:
            self.json_file_path = src
            f = open(src, 'r')
            json_data = json.load(f)
            self.known_faces.update(json_data)
            ff = open(FACE_NAME_JSON, 'r')
            name_json_data = json.load(ff)
            self.known_face_names.udate(name_json_data)
        except:
            pass

    # def load_image_from_folder(self, path):
    #     for file in face_recognition.image_files_in_folder(path):
    #         image = face_recognition.load_image_file(file)
    #         basename = os.path.splitext(os.path.basename(file))[0]
    #         self.record_face(image, basename)

    def record_new_face(self, encoding):
        name = self.get_new_name()
        info = self.get_info_dict(name, encoding)
        self.known_faces["faces"].append(info)
        self.known_face_names["names"].append(name)
        self.save_to_json()
        return name

    def get_info_dict(self, name, encoding):
        info = {
            "name": name,
            "encoding": encoding.tolist()
        }
        return info

    def save_to_json(self):
        dict_face = self.known_faces
        f = open(self.json_file_path, 'w')
        json.dump(dict_face, f, indent=4)
        dict_face_name = self.known_face_names
        ff = open(FACE_NAME_JSON, 'w')
        json.dump(dict_face_name, ff, indent=4)

    def get_new_name(self):
        ind = len(self.known_face_names["names"])
        c = "NO."
        name = c + str(ind+1)
        return name

    # 画像を渡すと、顔の場所と顔の名前が入ったdictを返す

    def analyze_faces_in_image(self, image):
        encodings = face_recognition.face_encodings(image)
        locations = face_recognition.face_locations(image)
        faces = {}
        faces["face"] = []

        if len(encodings) != 0:
            for encoding, location in zip(encodings, locations):
                face_info = self.compare_with_data(encoding, location)
                if face_info is None:
                    name = self.record_new_face(encoding)
                    face_info = self.get_draw_info(name, location, 1.00)
                faces["face"].append(face_info)
        print(faces)
        return faces

    def compare_with_data(self, encoding, location):
        if len(self.known_faces["faces"]) == 0:
            return None
        compare_encodings = [face["encoding"]
                             for face in self.known_faces["faces"]]
        distance = face_recognition.face_distance(
            compare_encodings, encoding)
        dist_results = list(distance <= self.tolerance)  # しきい値でふるいにかける
        if True in dist_results:
            assert isinstance(distance, np.ndarray)
            name = [face["name"]
                    for face in self.known_faces["faces"]][distance.argmin()]
            draw_info = self.get_draw_info(name, location, min(distance))
            return draw_info
        else:
            return None

    def get_draw_info(self, name, location, distance):
        draw_info = {
            "name": name,
            "info": {
                "location": location,
                "distance": distance
            }
        }
        return draw_info

    def draw_rect(self, image, faces):
        assert isinstance(faces, dict)
        if len(faces["face"]) == 0:
            return image

        for f in faces["face"]:
            location = f["info"]["location"]
            cv2.rectangle(
                image, (location[3], location[0]), (location[1], location[2]), (0, 0, 255), 2)
            cv2.rectangle(image, (location[3], location[2]-25),
                          (location[1], location[2]), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            prob = 100 - round(f["info"]["distance"], 3) * 100
            text = f["name"] + ":" + str(prob)
            cv2.putText(
                image, text, (location[3]+6, location[2]-6), font, 0.5, (255, 255, 255), 1)
        return image
