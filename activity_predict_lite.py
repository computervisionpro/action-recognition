import tensorflow as tf
import numpy as np
import cv2
#import pickle
#from collections import deque


def pred(interpreter, labels, input_video, mean, Q):
    """Inferencing the frames"""
    cap = cv2.VideoCapture(input_video)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224,224)).astype('float32')
        frame -= mean

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))

        # run inference
        interpreter.invoke()

        output_pred = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data[0])
        Q.append(output_pred)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = labels.classes_[i]

        text = "action: {}".format(label)
        cv2.putText(output, text, (2,20), cv2.FONT_HERSHEY_SIMPLEX, .38,
                        (0,255,0),1)

        yield output

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()

