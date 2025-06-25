
import cv2
import numpy as np
import onnxruntime as ort

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
emotion_model = ort.InferenceSession("emotion-ferplus-8.onnx")
emotion_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
           
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            face_roi = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (64, 64))
            processed_face = resized_face.reshape(1, 1, 64, 64).astype(np.float32)
            
            inputs = {emotion_model.get_inputs()[0].name: processed_face}
            emotion_preds = emotion_model.run(None, inputs)
            emotion_label = emotion_labels[np.argmax(emotion_preds[0])]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()