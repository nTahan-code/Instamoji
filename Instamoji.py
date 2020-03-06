from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
import time

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
emoji_array = {0: 'angerEmoji.png', 5: 'sadEmoji.png', 4: 'neutralEmoji.png', 1: 'disgustEmoji.png', 6: 'surpriseEmoji.png', 2: 'fearEmoji.png', 3: 'happyEmoji.png'}
model = load_model("model_v6_23.hdf5")

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, oimg = cam.read()
        img=oimg
        face_locations = face_recognition.face_locations(img)
        print(len(face_locations))
        
        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            if mirror: 
                img = cv2.flip(img, 1)
            
            width =right - left
            height = bottom-top
            dsize = (width, height)
            center=(int((left+right)/2),int((top+bottom)/2))

            face_image = img[top:bottom, left:right]
            face_image = cv2.resize(face_image, (48,48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
            predicted_class = np.argmax(model.predict(face_image))
            label_map = dict((v,k) for k,v in emotion_dict.items())
            predicted_label = label_map[predicted_class]
            print(predicted_label) 
            #cv2.rectangle(oimg,(left,top),(right,bottom),2,10)
            cv2.circle(oimg,center,int(width/2),(255, 128, 0))
            # emoji overlay without alpha
            emogi=cv2.imread("images/"+emoji_array[predicted_class])
            output = cv2.resize(emogi, dsize)
            x_offset=left
            y_offset=top
            oimg[y_offset:y_offset+output.shape[0], x_offset:x_offset+output.shape[1]] = output
           
            output = cv2.resize(emogi, dsize)
            



        #except:
         #   print("no face detected")
        cv2.imshow('my webcam', oimg)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
