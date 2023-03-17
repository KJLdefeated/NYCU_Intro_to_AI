import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # Load the txt file
    cords1 = []
    cords2 = []
    img_path = []
    with open(dataPath) as file:
        tmp = file.readline()
        tmp = tmp.split(' ')
        img_path.append(tmp[0])
        n_cord = int(tmp[1])
        for _ in range(n_cord):
            cor = file.readline()
            cor = cor.split(' ')
            res = tuple(map(int, cor))
            cords1.append(res)
        tmp = file.readline()
        tmp = tmp.split(" ")
        img_path.append(tmp[0])
        n_cord = int(tmp[1])
        for _ in range(n_cord):
            cor = file.readline()
            cor = cor.split(" ")
            res = tuple(map(int, cor))
            cords2.append(res)

    # Process on first image
    img1 = cv2.imread(os.path.join('data/detect/', img_path[0]))
    for (x, y ,w, h) in cords1:
        face_img = img1[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (19, 19),interpolation=cv2.INTER_NEAREST)
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        res = clf.classify(face_img_gray)
        if res == 1:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Process on second image
    img2 = cv2.imread(os.path.join('data/detect/', img_path[1]))
    for (x, y ,w, h) in cords2:
        face_img = img2[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (19, 19),interpolation=cv2.INTER_NEAREST)
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        res = clf.classify(face_img_gray)
        if res == 1:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the images
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # End your code (Part 4)


def detect_selfdata(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # Load the txt file
    cords1 = []
    img_path = []
    with open(dataPath) as file:
        tmp = file.readline()
        tmp = tmp.split(' ')
        img_path.append(tmp[0])
        n_cord = int(tmp[1])
        for _ in range(n_cord):
            cor = file.readline()
            cor = cor.split(' ')
            res = tuple(map(int, cor))
            cords1.append(res)

    img1 = cv2.imread(os.path.join('data/Self_made_detect', img_path[0]))
    for (x, y ,w, h) in cords1:
        face_img = img1[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (19, 19),interpolation=cv2.INTER_NEAREST)
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        res = clf.classify(face_img_gray)
        if res == 1:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the images
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # End your code (Part 4)