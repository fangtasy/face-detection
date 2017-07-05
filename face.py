import cv
import cv2
import datetime,time


# load all the necessary training sets
face_cade  = cv.Load("data/haarcascade_frontalface_alt.xml")
eye_cade   = cv.Load("data/parojosG.xml")
nose_cade  = cv.Load("data/Nariz_nuevo_20stages.xml")
hand_cade  = cv.Load("data/palm.xml")


# images to overlay on features
moustache = cv.LoadImage("data/moustache.png")
eyeball   = cv.LoadImage("data/glasses.png")
tophat    = cv.LoadImage("data/hat.png")
handPic   = cv.LoadImage("data/hand.png")

#button part
button_eye    = cv.LoadImage("data/eye_button.png")
button_hat   = cv.LoadImage("data/hat_button.png")
button_nose   = cv.LoadImage("data/moustache_button.png")


# drawn bottom up rather than top down
cv.Flip(tophat, tophat, flipMode=0)

# bools for which features to detect, set via number keys
face_on, eye_on, nose_on = False, False, False
hand_on = True


starttime = datetime.datetime.now()
endtime = datetime.datetime.now()


def overlay_image(frame, image, x, y, w, h):
    # resize the image to fit the detected feature
    new_feature = cv.CreateImage((w, h), 8, 3)
    cv.Resize(image, new_feature, interpolation=cv.CV_INTER_AREA)
    # overlay the image on the frame
    for py in xrange(h):
        for px in xrange(w):
            pixel = cv.Get2D(new_feature, py, px)
            # don't map the whitespace surrounding the image
            if pixel != (255.0, 255.0, 255.0, 0.0):
                if image is tophat:
                    # above feature
                    new_y = y - py
                elif image is moustache:
                    # bottom half of feature
                    new_y = (h / 2) + y + py
                else:
                    # over feature
                    new_y = y + py
                new_x = x + px

                # make sure the image is in the frame
                if 0 < new_x < frame.width and 0 < new_y < frame.height:
                    cv.Set2D(frame, new_y, new_x, pixel)


def detect_feature(frame, cade, min_size=(30, 30)):
    """detect a single feature and draw a box around the matching area

    By using Canny Pruning, we disregard certain image regions to boost the
    performance. Tweaking the parameters passed to HaarDetectObjects greatly
    affects performance as well.
    """
    # detect feature in image
    objs = cv.HaarDetectObjects(frame, cade, cv.CreateMemStorage(),
                                scale_factor=1.2, min_neighbors=3,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                min_size=min_size)

    # pass the proper image to overlay for each match
    for (x, y, w, h), n in objs:
        if cade == face_cade:
            overlay_image(frame, tophat, x, y, w, h)
        elif cade == eye_cade:
            overlay_image(frame, eyeball, x, y, w, h)
        elif cade == nose_cade:
            overlay_image(frame, moustache, x, y, w, h)


def detect_features(frame):
    """detect features based on loaded cascades and render the frame"""
    if face_on:
        detect_feature(frame, face_cade)
    if eye_on:
        detect_feature(frame, eye_cade)
    if nose_on:
        detect_feature(frame, nose_cade)


def detect_hand_Control(frame,cade, min_size=(30, 30)):#use hand to control the face change
    global starttime, endtime,face_on,eye_on,nose_on



    objs = cv.HaarDetectObjects(frame, cade, cv.CreateMemStorage(),
                                scale_factor=1.2, min_neighbors=3,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                min_size=min_size)
   
    for (x, y, w, h), n in objs:
        if cade == hand_cade:
            overlay_image(frame ,handPic,x,y,w,h)
            x_mid = x+(w/2);
            y_mid = y+(h/2);
            
            if x_mid<210 and x_mid > 150 and y_mid <190 and y_mid>150:
                
                if eye_on == True:
                    endtime=datetime.datetime.now()
                    if (endtime-starttime).seconds>2.0:
                        eye_on = False
                else:
                    starttime=datetime.datetime.now()
                    if (starttime-endtime).seconds>2.0:
                        eye_on = True
            if x_mid<210 and x_mid > 150 and y_mid <390 and y_mid>350:
                
                if face_on == True:
                    endtime=datetime.datetime.now()
                    if (endtime-starttime).seconds>2.0:
                        face_on = False
                else:
                    starttime=datetime.datetime.now()
                    if (starttime-endtime).seconds>2.0:
                        face_on = True
            if x_mid<210 and x_mid > 150 and y_mid <540 and y_mid>500:
                
                if nose_on == True:
                    endtime=datetime.datetime.now()
                    if (endtime-starttime).seconds>2.0:
                        nose_on = False
                else:
                    starttime=datetime.datetime.now()
                    if (starttime-endtime).seconds>2.0:
                        nose_on = True  




def add_button(frame):
  
    #eye button location (100,150)
    new_feature_eye = cv.CreateImage((60, 40), 8, 3)
    cv.Resize(button_eye, new_feature_eye, interpolation=cv.CV_INTER_AREA)
    for py in xrange(40):
        for px in xrange(60):
            pixel=cv.Get2D(new_feature_eye, py, px)
            if pixel != (255.0, 255.0, 255.0, 0.0):
                cv.Set2D(frame, py+100, px+150, pixel)
    # cv.Flip(frame,frame,1)
    window=cv.CreateImage((1080,600),8,3)
    cv.Resize(frame,window,interpolation=cv.CV_INTER_AREA)
    #end eye button
    
    #hat button location (300,150)
    new_feature_hat = cv.CreateImage((60, 40), 8, 3)
    cv.Resize(button_hat, new_feature_hat, interpolation=cv.CV_INTER_AREA)
    for py in xrange(40):
        for px in xrange(60):
            pixel=cv.Get2D(new_feature_hat, py, px)
            if pixel != (255.0, 255.0, 255.0, 0.0):
                cv.Set2D(frame, py+300, px+150, pixel)
    window=cv.CreateImage((1080,600),8,3)
    cv.Resize(frame,window,interpolation=cv.CV_INTER_AREA)
    #end hat button


    #nose button location (550,150)
    new_feature_nose = cv.CreateImage((60, 40), 8, 3)
    cv.Resize(button_nose, new_feature_nose, interpolation=cv.CV_INTER_AREA)
    for py in xrange(40):
        for px in xrange(60):
            pixel=cv.Get2D(new_feature_nose, py, px)
            if pixel != (255.0, 255.0, 255.0, 0.0):
                cv.Set2D(frame, py+550, px+150, pixel)
    cv.Flip(frame,frame,1)
    window=cv.CreateImage((1080,600),8,3)
    cv.Resize(frame,window,interpolation=cv.CV_INTER_AREA)
    #end nose button



def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect_features on each
    frame to look for True features (based on keys 1-3).
    """
    global face_on, eye_on, nose_on, hand_on

    # set up window and camera
    win_name = "faceDetector"
    source = cv.CaptureFromCAM(0)
    cv.NamedWindow(win_name)

    while True:
        # grab the key press to delegate action
        key = cv.WaitKey(15)
        frame = cv.QueryFrame(source)
        detect_hand_Control(frame,hand_cade)
        # escape key to exit
        if key == 27:
            break
        

    
    

        # get the frame, detect selected features, and show frame
        
        detect_features(frame)
        
        #add button () trigger to start the detection
        window= add_button(frame)
        #end add button


        cv.ShowImage(win_name, frame)

    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
