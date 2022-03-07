import cv2
import numpy as np
from keras import models
cascade = cv2.CascadeClassifier("custom.xml")
img=cv2.imread('Main2/36.jpg')
classifier=models.load_model("../CNN/custommodel3.p")
dic = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','Q','R','S','T','U','V','X','Y','Z']
def hauxuly(img):
    img2 = cv2.resize(img, (32, 32))
    images = np.array(img2)
    images = images / 255
    k = images.reshape(1, 32, 32, 1)
    return k

def loaibongoaicanh(rects, plate_image):
    copy_plate = plate_image.copy()
    Height, Width = plate_image.shape[:2]
    mask = np.zeros((Height + 2, Width + 2), np.uint8)
   #cv2.imshow('matna',mask)
    for i, rect in enumerate(rects):
        cx, cy = (int(x) for x in rect[0])
        cv2.circle(plate_image, (cx, cy), 3, (0, 255, 0), -1)
        w, h = (int(x) for x in rect[1])
        minSize = w if w < h else h
        minSize = minSize - minSize * 0.5
        mask = np.zeros((Height + 2, Width + 2), np.uint8)
        seed_pt = None
        connectivity = 4
        loDiff = 30;
        upDiff = 30
        newMaskVal = 255
        numSeeds = 250
        flags = connectivity + (newMaskVal << 8) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY
        for j in range(numSeeds):
            minX = rect[0][0] - minSize / 2 - 25 if rect[0][0] - minSize / 2 - 25 >= 0 else rect[0][0] - minSize / 2
            maxX = rect[0][0] + (minSize / 2) + 25 if rect[0][0] + (minSize / 2) + 25 < Width else rect[0][
                                                                                                       0] + minSize / 2
            minY = rect[0][1] - minSize / 2 - 10 if rect[0][1] - minSize / 2 - 10 >= 0 else rect[0][1] - minSize / 2
            maxY = rect[0][1] + (minSize / 2) + 10 if rect[0][1] + (minSize / 2) + 10 < Height else rect[0][1] + (
                        minSize / 2)
            seed_ptX = np.random.randint(minX, maxX)
            seed_ptY = np.random.randint(minY, maxY)
            seed_pt = (seed_ptX, seed_ptY)
            if seed_ptX >= Width or seed_ptY >= Height or copy_plate[seed_ptY, seed_ptX] != 0:
                continue
            cv2.circle(plate_image, seed_pt, 1, (0, 255, 255), -1)
            cv2.floodFill(plate_image, mask, seed_pt, 255, loDiff, upDiff, flags)

        #cv2.imshow("Mask", mask)
    return mask


def phantachkitu(mask, contours, plate_img):
    character_img = []

    for i, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        character_roi = np.copy(plate_img[y:y + h, x:x + w])
        character_roi = cv2.resize(character_roi, None, fx = 0.75, fy = 0.75, interpolation = cv2.INTER_AREA)
        character_roi = cv2.copyMakeBorder(character_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
        charImg = cv2.GaussianBlur(character_roi, (3, 3), 0)
        charImg = cv2.adaptiveThreshold(charImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        openingImg = cv2.morphologyEx(charImg, cv2.MORPH_OPEN, kernel)
        character_img.append(openingImg)
        cv2.imshow("Character {:d}".format(i), openingImg)
    return character_img

def tienxuly(found_plate, plate_box):
    found_plate = cv2.resize(found_plate, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #cv2.imshow("Scale down a half", found_plate)
    copy_plate = found_plate.copy()
    grayPlate = cv2.cvtColor(found_plate, cv2.COLOR_BGR2GRAY)
    grayPlate = cv2.equalizeHist(grayPlate)
    blurPlate = cv2.GaussianBlur(grayPlate, (5, 5), 0)
    #cv2.imshow("anhxam", blurPlate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closingImg = cv2.morphologyEx(blurPlate, cv2.MORPH_CLOSE, kernel) #Mở cửa chỉ là một tên khác của xói mòn tiếp theo là giãn nở
    cv2.imshow("Closing image", closingImg)
    threshPlate = cv2.adaptiveThreshold(closingImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 22)
    cv2.imshow("Threshold", threshPlate)
    contours, _ = cv2.findContours(threshPlate.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        if rect[1][0] == 0 or rect[1][1] == 0 or rect[1][0] < 80 or rect[1][1] < 80:
            continue
        rects.append(rect)

    cv2.drawContours(found_plate, contours, -1, (0, 255, 0), 2)
    #cv2.imshow("Contours", found_plate)
    mask = loaibongoaicanh(rects, threshPlate)
    mask = cv2.bitwise_not(mask)
    #cv2.imshow("Mask inverse", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get only corners of each contour
    refine_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if 300 <= w * h <= 1250 and w <= h:
            refine_contours.append(cnt)
    cv2.drawContours(copy_plate, contours, -1, (0, 255, 0), 2)
    #cv2.imshow("Mask contour", copy_plate)
    return refine_contours, mask, threshPlate

def nhandangkitu(found_plate, plate_box, img):
    contours, mask, threshPlate = tienxuly(found_plate, plate_box)
    #cv2.imshow("Mask contour2", mask)
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    for i in range(0, len(contours) - 1, 2):
        x1, y1, _, _ = cv2.boundingRect(contours[i])
        j = i + 1
        x2, y2, _, _ = cv2.boundingRect(contours[j])
        if not (x1 < x2 and y1 < y2):
            t = contours[i]
            contours[i] = contours[j]
            contours[j] = t
    character_img = phantachkitu(mask, contours, threshPlate)
    plate_text = ""
    hangtren=""
    hangduoi=""
    for i, charImg in enumerate(character_img):
        text = str(classifier.predict_classes(hauxuly(charImg)))

        #custom_config = r'-l eng --oem 3 --psm 6'
        #text = pytesseract.image_to_string(charImg,config=custom_config)
        c = "".join(i for i in text if i.isalnum())
        print(int(c))
        plate_text += str(dic[int(c)])
    if(len(plate_text)<=8):
        for i in range(len(plate_text)):
            if i%2==0:
                hangtren+=plate_text[i]
            else:
                hangduoi+=plate_text[i]
    else:
        for i in range(8):
            if i%2==0:
                hangtren+=plate_text[i]
            else:
                hangduoi+=plate_text[i]
        hangduoi=hangduoi+plate_text[8]
    print(hangtren)
    print(hangduoi)
    x, y, w, h = plate_box
    copy_img = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(copy_img, plate_text, (x, y + h + 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(copy_img, hangtren, (x, y + h + 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(copy_img, hangduoi, (x, y + h + 80), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("License plate number recognition", copy_img)
    print(plate_text)
while 1:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bienso = cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in bienso:
        img2=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        image = np.copy(img[y:y+h, x:x + w])
        nhandangkitu(image, (x, y, w, h), img2)
    cv2.imshow('img2', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
