import cv2

video = cv2.VideoCapture(
    'rtsp://admin:nam781999@192.168.1.206:554/cam/realmonitor?channel=1&subtype=0')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_3.avi', fourcc, 20.0, (512, 512))

while video.isOpened():
    ret, frame = video.read()
    if ret:
        frame = cv2.resize(frame, (512, 512))
        cv2.imshow('test', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
