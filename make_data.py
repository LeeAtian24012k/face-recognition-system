import cv2
import time
import os


def main():
    # Tên thư mục chứa ảnh train
    lable = "Hung"
    cap = cv2.VideoCapture(0)
    preTime = 0
    # Thay đổi kích thước Camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # Số ảnh tối đa được chụp
    img_counter = 300
    print("Press ESC to close the program, SPACE to crop the area containing the face!")
    while cap.isOpened():
        success, img = cap.read()
        try:
            key = cv2.waitKey(1)
            # Tạo thư mục theo biến lable nếu chưa có
            if not os.path.exists('train_img/' + str(lable)):
                os.mkdir('train_img/' + str(lable))
            if key % 256 == 32:
                if img_counter != 0:
                    # SPACE pressed
                    img_name = lable + "_{}.jpg".format(img_counter)
                    save_path = "train_img/" + lable + "/" + img_name
                    cv2.imwrite(save_path, img)
                    print("Output: {}".format(save_path))
                    img_counter -= 1
                elif img_counter == 0:
                    break
            if key % 256 == 27:
                # ESC pressed
                break
        except Exception as e:
            print(e)

        curTime = time.time()
        fps = 1 / (curTime - preTime)
        preTime = curTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Make data", img)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
