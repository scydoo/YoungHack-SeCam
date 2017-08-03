import numpy as np
import cv2

cap = cv2.VideoCapture('video/3.mp4')
fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
fcnt = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print(fps, fcnt)

fourcc = cv2.cv.CV_FOURCC(*'a\0\0\0')
out = cv2.VideoWriter('video/output.mp4',fourcc, fps, (480,640))
print(cap,out)
i = 0
while(cap.isOpened()):
    i += 1
    ret, frame = cap.read()
    if i < 1:
        continue
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(i)
    # out.write(frame)
    # out.release()
    if(i%fps == 0):
        cv2.imwrite('tmp/%04d.jpg'%(i/fps), frame)
    if i > 600:
        break
cap.release()
out.release()
cv2.destroyAllWindows()