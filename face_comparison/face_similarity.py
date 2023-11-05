import io
import base64
import time
import numpy as np
from PIL import Image
import cv2
import face_recognition


class FaceSimilarity():

    def __init__(self, source_img, target_img, similarity_threshold, is_base64=True):
        super().__init__()

        if is_base64 == True:
            # convert base64 image data to image array(Please use for jpeg image)
            self.source_im = self.base64_to_RGB(source_img)
            self.target_im = self.base64_to_RGB(target_img)
        else:
            self.source_im = source_img
            self.target_im = target_img

        self.similarity_threshold = float(similarity_threshold)
        self.similarity_ary = []

        self.src_face_ary = None
        self.tgt_face_ary = None

        self.response_data = {
            "FaceMatches": [
                {
                    "Face": {
                        "BoundingBox": {
                            "Height": None,
                            "Left": None,
                            "Top": None,
                            "Width": None,
                        }
                    },
                    "Similarity": None
                }

            ],
            "SourceImageFace": {
                "BoundingBox": {
                    "Height": None,
                    "Left": None,
                    "Top": None,
                    "Width": None,
                }
            },
            "UnmatchedFaces": []
        }

        self.init()

    def init(self):
        # self.face_detect()
        #
        # if len(self.src_face_ary) == 0:
        #     return
        #
        # self.get_src_ratio()
        #
        # if len(self.tgt_face_ary) == 0:
        #     return
        #
        # self.get_similarity()
        # if self.source_im.ndim == 2:
        #     self.source_im = cv2.cvtColor(self.source_im, cv2.COLOR_GRAY2RGB)
        #
        # if self.target_im.ndim == 2:
        #     self.target_im = cv2.cvtColor(self.target_im, cv2.COLOR_GRAY2RGB)

        source_im_arr = [np.array(Image.open(self.source_im)),
                         np.array(Image.open(self.source_im).rotate(90, expand=True)),
                         np.array(Image.open(self.source_im).rotate(180, expand=True)),
                         np.array(Image.open(self.source_im).rotate(270, expand=True))]
        target_im_arr = [np.array(Image.open(self.target_im)),
                         np.array(Image.open(self.target_im).rotate(90, expand=True)),
                         np.array(Image.open(self.target_im).rotate(180, expand=True)),
                         np.array(Image.open(self.target_im).rotate(270, expand=True))]

        source_face_encoding = None
        target_face_encoding = None
        for i in range(len(source_im_arr)):
            source_face_encoding = face_recognition.face_encodings(source_im_arr[i])
            if len(source_face_encoding) != 0:
                break

        for i in range(len(target_im_arr)):
            target_face_encoding = face_recognition.face_encodings(target_im_arr[i])
            if len(target_face_encoding) != 0:
                break

        if source_face_encoding is None:
            self.response_data = {'Warning': 'Incorrect source image'}
            return
        if target_face_encoding is None:
            self.response_data = {'Warning': 'Incorrect target image'}
            return

        face_distance = face_recognition.face_distance(target_face_encoding, source_face_encoding[0])[0]
        self.response_data = {'Similarity': round((1 - face_distance) * 100, 2)}

    def get_src_ratio(self):
        src_ratio = self.src_face_ary.astype(float)
        HH, WW = self.source_im.shape[0], self.source_im.shape[1]
        x, y, w, h = self.src_face_ary[0]
        src_ratio[0] = [x / WW, y / HH, w / WW, h / HH]

        self.response_data["SourceImageFace"]["BoundingBox"]["Height"] = src_ratio[0][3]
        self.response_data["SourceImageFace"]["BoundingBox"]["Left"] = src_ratio[0][0]
        self.response_data["SourceImageFace"]["BoundingBox"]["Top"] = src_ratio[0][1]
        self.response_data["SourceImageFace"]["BoundingBox"]["Width"] = src_ratio[0][2]

    def base64_to_RGB(self, base64_string):
        imgdata = base64.b64decode(str(base64_string))
        img = io.BytesIO(imgdata)
        # img = Image.open(io.BytesIO(imgdata))
        # img = np.array(img)
        return img

    def face_detect(self):

        if self.source_im.ndim == 2:
            self.source_im = cv2.cvtColor(self.source_im, cv2.COLOR_GRAY2RGB)

        gray_src = cv2.cvtColor(self.source_im, cv2.COLOR_BGR2GRAY)

        if self.target_im.ndim == 2:
            self.target_im = cv2.cvtColor(self.target_im, cv2.COLOR_GRAY2RGB)

        gray_tgt = cv2.cvtColor(self.target_im, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt_tree.xml')
        self.src_face_ary = face_cascade.detectMultiScale(gray_src, scaleFactor=1.5, minNeighbors=3)
        self.tgt_face_ary = face_cascade.detectMultiScale(gray_tgt, scaleFactor=1.5, minNeighbors=3)

    def get_face_match_ratio(self):
        tgt_ratio = self.tgt_face_ary.astype(float)
        HH, WW = self.target_im.shape[0], self.target_im.shape[1]
        x, y, w, h = self.tgt_face_ary[0]
        tgt_ratio[0] = [x / WW, y / HH, w / WW, h / HH]
        self.response_data["FaceMatches"][0]["Face"]["BoundingBox"]["Height"] = tgt_ratio[0][3]
        self.response_data["FaceMatches"][0]["Face"]["BoundingBox"]["Left"] = tgt_ratio[0][0]
        self.response_data["FaceMatches"][0]["Face"]["BoundingBox"]["Top"] = tgt_ratio[0][1]
        self.response_data["FaceMatches"][0]["Face"]["BoundingBox"]["Width"] = tgt_ratio[0][2]
        self.response_data["FaceMatches"][0]["Similarity"] = self.ordered_similarity[0]

    def get_face_unmatch_ratio(self, indexes):
        HH, WW = self.target_im.shape[0], self.target_im.shape[1]
        for idx in indexes:
            x, y, w, h = self.tgt_face_ary[idx].astype(float)
            ratio = [x / WW, y / HH, w / WW, h / HH]

            self.response_data["UnmatchedFaces"].append(
                {
                    "Face": {
                        "BoundingBox": {
                            "Height": ratio[3],
                            "Left": ratio[0],
                            "Top": ratio[1],
                            "Width": ratio[2],
                        }
                    },
                    "Similarity": self.similarity_ary[idx]
                }
            )

    def get_similarity(self):
        # Encode the known image
        x, y, w, h = self.src_face_ary[0]
        img = self.source_im[y:y + h, x:x + w]
        source_image_encoding = face_recognition.face_encodings(img)[0]

        # Loop over all the images we want to check for similar people
        i = 0
        for item in self.tgt_face_ary:
            # Load an image to check
            x, y, w, h = item
            img = self.target_im[y:y + h, x:x + w]

            # Get the location of faces and face encodings for the current image
            time.sleep(0.010)
            face_encodings = face_recognition.face_encodings(img)
            if not face_encodings:
                break

            # Get the face distance between the known person and all the faces in this image
            face_distance = face_recognition.face_distance(face_encodings, source_image_encoding)[0]
            self.similarity_ary.append((1 - face_distance) * 100)

        order = np.flip(np.argsort(self.similarity_ary))
        self.ordered_similarity = np.array(self.similarity_ary)[order]

        count = np.sum(self.ordered_similarity >= self.similarity_threshold)
        if count >= 1:
            self.get_face_match_ratio()

        indexes = order[self.ordered_similarity < self.similarity_threshold]
        if len(indexes) >= 1:
            self.get_face_unmatch_ratio(indexes)

        # x, y, w, h = self.tgt_face_ary[order[0]]
        # img = self.target_im[y:y + h, x:x + w]
        # # Display the face image that we found to be the best match!
        # pil_image = Image.fromarray(img)
        # pil_image.show()
