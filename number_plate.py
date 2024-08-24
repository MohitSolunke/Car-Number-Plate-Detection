import cv2


harcascade = "E:\PROJECTS\Computer Vision( CV) Project\Car Number Plate Detection\model\haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0) # Capture webcam


cap.set(3, 640)  # set width
cap.set(4, 480)  # set height

min_area = 500   # min area for a detected region to be considered as a licence plate
count = 0

while True:     # Infinite loop
    # Reading the frame from webcam
    sucess, img = cap.read()
    
    # creating a licence plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)
    
    # converting the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    # Iterating through the detected plates
    for(x, y, w, h) in plates:
        area = w * h
        # if the area is greater than minimum area, draw a
        # rectangle around the plate and display the
        # text "LICENSE Plate" on the top left corner of the rectangle.
        # Also display the region of interest (ROI) of the license plate.
        
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # diplay the text License plate on the top left corner of the 
            # rectangle
            # also display the region of interest (ROI) of the license plate
            cv2.putText(img, "License plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            # display the region of interest (ROI) of the license plate
            img_roi = img[y:y + h, x:x + w] 
            # display the region of interest of the licence platein seperate Window
            cv2.imshow("ROI", img_roi)
            
        cv2.imshow("Result", img)
        
        # Saving the plate when 's' key is pressed
        # waitkey(1) means that the programe will wait for q milisecond
        # and tehn check if the "s" key is pressed. if it is pressed, the
        # program will save the plate and display the text "Plate Saved.
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Plate_" + str(count) + ".jpg", img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", img)
            cv2.waitKey(500)
            count += 1
            
            
            
            
            
    