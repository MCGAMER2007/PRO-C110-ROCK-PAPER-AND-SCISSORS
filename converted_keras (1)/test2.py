import cv2
import tensorflow as tf 
import numpy as np
model = tf.keras.models.load_model("keras_model.h5")
video = cv2.VideoCapture(0)
while True:
    check,frame = video.read()
    image = cv2.resize(frame, (224,224))
    test_image = np.array(image, dtype = np.float32)
    test_image = np.expand_dims(test_image, axis = 0)
    normalized_image = test_image/255.0
    prediction = model.predict(normalized_image)
    print(prediction)
    cv2.imshow("result", frame)
    key = cv2.waitKey(1)
    if key == 64:
        print("closing")
        break
video.release()