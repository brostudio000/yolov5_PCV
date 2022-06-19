import cv2 as cv
import torch
import numpy as np 

# Model yang digunakan untuk deteksi
path = "best_result14.pt"
model = torch.hub.load('', 'custom', path=path, source='local')
model.img = 480
# Source yang digunakan untuk deteksi
cap = cv.VideoCapture(1)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Baca source sampai selesai
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    result = model(frame) # Inference

    DetectionBoxes = result.pandas().xyxy[0].values.tolist()
    for Box in DetectionBoxes:
      # Bounding Box
      xmin,ymin,xmax,ymax,conf,cl,name = Box
      pmin =(int(xmin),int(ymin))
      pmax =(int(xmax),int(ymax))
      frame = cv.rectangle(frame, pmin, pmax,(255,255,0),3)
      pPos = (int(xmin),int(ymin))
      # Font
      fontScale = 2
      color = (255, 255, 255)
      thickness = 2
      font = cv.FONT_HERSHEY_DUPLEX
      frame = cv.putText(frame,name, pPos, font,fontScale, color,thickness, cv.LINE_AA)
    # Menampilkan window capture
    cv.imshow('Object Detection',frame)
    # Tombol Q untuk keluar dari window capture
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  # Keluar dari loop
  else: 
    break
cap.release()
# Hapus Semua frame
cv.destroyAllWindows()