import cv2
import os



def parseVideo():
    local_directory = 'videos'
    dest_directory = 'frames'
    frame_no = 0
    spacer = 4

    for file in os.listdir(local_directory):
        success = True
        count = 0

        if file.endswith('.mp4') == False:
            continue

        vidcap = cv2.VideoCapture(os.path.join(local_directory, file))

        while success:
            try:
                success,image = vidcap.read()
                if count % spacer == 0:
                    print('Save frame: ', frame_no)
                    height,width,channels = image.shape
                    cv2.imwrite(os.path.join(dest_directory, "frame_"+str(frame_no).zfill(6)+".jpg"), image) # save frame as JPEG file
                    frame_no += 1
                count += 1

            except:
                continue

        vidcap.release()

parseVideo()
    