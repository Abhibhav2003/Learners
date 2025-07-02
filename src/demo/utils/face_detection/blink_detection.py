import cv2 as cv
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


cap = cv.VideoCapture('Learners/src/demo/files/video.mp4')

detector = FaceMeshDetector(maxFaces=1)

plotY = LivePlot(640,360,[20,50])
idList = [22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []
blinkCount = 0
counter = 0

while True:
    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
        cap.set(cv.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read() 
    if not success:
        break
    img = cv.resize(img,(640,360))
    
    img, faces = detector.findFaceMesh(img,draw=False)
    
    if faces:
        face = faces[0]
        for id in idList:
            cv.circle(img,face[id],2,(255,0,255),cv.FILLED)
        
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        vertical_len,_ = detector.findDistance(leftUp,leftDown)
        horizontal_len,_ = detector.findDistance(leftLeft,leftRight)
        cv.line(img,leftUp,leftDown,(0,200,0),2)
        cv.line(img,leftLeft,leftRight,(0,200,0),2)

        ratio = (vertical_len/horizontal_len)*100
        ratioList.append(ratio)
        if len(ratioList) > 5:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 36 and counter == 0:
            blinkCount +=1
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
        
        cvzone.putTextRect(img,f'Blink Count : {blinkCount}',(50,50))
        imgPlot = plotY.update(ratioAvg)
        cv.imshow("ImagePlot",imgPlot)

    cv.imshow("Image",img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()