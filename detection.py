import cv2 
import torch
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.img =360
cap = cv2.VideoCapture(1)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Baca sampai selesai
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    results = model(frame)
    DetectedBoxes =results.pandas().xyxy[0].values.tolist()
    #---Start  for Box...
    for Box in DetectedBoxes:
      xmin,ymin,xmax,ymax,conf,cl,name = Box
      pmin =(int(xmin),int(ymin))
      pmax =(int(xmax),int(ymax))
      frame=cv2.rectangle(frame ,pmin,pmax,(0,255,255),2)
      pPos = (int(xmin),int(ymin))
      # fontScale
      fontScale = 0.8
      color = (255, 255, 255)
      thickness = 2
      font = cv2.FONT_HERSHEY_SIMPLEX
      frame = cv2.putText(frame,name, pPos, font,fontScale, color,thickness, cv2.LINE_AA)
    #---end for Box...

    cv2.imshow('Frame',frame)
    # Tekan Tombol Q Untuk Keluar
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Keluar dari loop
  else: 
    break
cap.release()
# Hapus Semua frame
cv2.destroyAllWindows()