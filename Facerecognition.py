import cv2

# 读取图像
img = cv2.imread('test.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 检测人脸
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 提取特征并存储在数据库中
features_db = []
for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    features = extract_features(face_roi)
    features_db.append(features)

# 读取另一张图像
img2 = cv2.imread('test2.jpg')

# 转换为灰度图像
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

# 对于每个检测到的人脸，提取特征并将其与数据库中的特征进行比较
for (x, y, w, h) in faces2:
    face_roi = gray2[y:y+h, x:x+w]
    features = extract_features(face_roi)
    for i, features_db_i in enumerate(features_db):
        similarity = compare_features(features, features_db_i)
        if similarity > threshold:
            print("This is person ", i+1)
            break

# 显示结果
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
