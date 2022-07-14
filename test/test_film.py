import cv2
from test.test_watermark import Reveal_one_pic

def film_detect(film_path):
    cap = cv2.VideoCapture(film_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./test_picture/out.avi',fourcc ,fps, size)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = Reveal_one_pic(frame,is_cuda=False)
            print(type(frame))
            frame =cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            out.write(frame)
            cv2.imshow('frame',frame)
            cv2.waitKey(100)
            cv2.imwrite("./test_picture/11.png",frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    film_detect(r"C:\Users\brighten\Desktop\软件服务外包\测试视频1.mp4")
