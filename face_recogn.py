from cProfile import label
from email import message_from_binary_file
from logging import root
from msilib.schema import RadioButton
from tkinter import *
from tkinter import ttk
from tkinter import font
from turtle import width
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
from time import strftime
from datetime import datetime


class FaceRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")  # setting windows screen size
        self.root.title("Face Recognition System")

        # Title of the system
        title_lbl = Label(
            self.root,
            text="FACE RECOGNITION",
            font=("times new roman", 35, "bold"),
            bg="white",
            fg="green",
        )
        title_lbl.place(x=0, y=0, width=1530, height=45)

        imgface = Image.open(r"C:\Users\Dell\Desktop\Minor-Project\Images\faace.jpg")
        imgface = imgface.resize((1530, 800))
        self.photoimg2 = ImageTk.PhotoImage(imgface)

        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=0, y=55, width=1530, height=800)

        bt1_l = Button(
            self.root,
            text="Face Detector",
            cursor="hand2",
            command=self.face_recog,
            font=("times new roman", 25, "bold"),
            bg="blue",
            fg="white",
        )
        bt1_l.place(x=500, y=730, width=500, height=40)

    ####### Attendance @######33

    # def mark_attendance(self, i, r, n, d):
    #     with open("attend.csv", "r+", newline="\n") as f:
    #         myDatalist = f.readlines()
    #         name_list = []
    #         for line in myDatalist:
    #             entry = line.split((","))
    #             name_list.append(entry[0])

    #         # ek choti aayepaxi fere naaaos
    #         if (
    #             (i not in name_list)
    #             and (r not in name_list)
    #             and (n not in name_list)
    #             and (d not in name_list)
    #         ):
    #             now = datetime.now()
    #             d1 = now.strftime("%d/%m/%Y")
    #             dtString = now.strftime("%H:%M:%S")
    #             f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")

    ######### Face Recognition #######
    def face_recog(self):
        def draw_boundray(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(
                gray_image, scaleFactor, minNeighbors
            )

            coord = []

            for x, y, w, h in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                id, conf = clf.predict(gray_image[y : y + h, x : x + w])
                confidence = int((100 * (1 - conf / 300)))

                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="qwerty123",
                    database="admin_minorproject",
                )
                my_cursor = conn.cursor()

                my_cursor.execute(
                    "select Name from student where Student_id=" + str(id)
                )
                n = my_cursor.fetchone()
                n = "+".join(n)

                my_cursor.execute(
                    "select Roll from student where Student_id=" + str(id)
                )
                r = my_cursor.fetchone()
                r = "+".join(r)

                my_cursor.execute(
                    "select Class from student where Student_id=" + str(id)
                )
                d = my_cursor.fetchone()
                d = "+".join(d)

                my_cursor.execute(
                    "select Section from student where Student_id=" + str(id)
                )
                s = my_cursor.fetchone()
                s = "+".join(s)

                my_cursor.execute(
                    "select Batch from student where Student_id=" + str(id)
                )
                b = my_cursor.fetchone()
                b = "+".join(b)
                if conf < 55:
                    cv2.putText(
                        img,
                        f"Roll:{r}",
                        (x, y - 75),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        img,
                        f"Name:{n}",
                        (x, y - 55),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        img,
                        f"Section:{s}",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        img,
                        f"Class:{d}" + " " + str(conf),
                        (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        img,
                        f"Batch:{b}" + " " + str(conf),
                        (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )

                    self.mark_attendance(
                        b, r, n, d, s
                    )  # Attendance wala function lai call gareko attendance lina ko lagi
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(
                        img,
                        "Unknown Face",
                        (x, y - 55),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        3,
                    )

                coord = [x, y, w, h]

            return coord

        def recognize(img, clf, faceCascade):
            coord = draw_boundray(
                img, faceCascade, 1.1, 10, (255, 25, 255), "Face", clf
            )
            return img

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap = cv2.VideoCapture(0)

        while True:
            ret, img = video_cap.read()
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome to Face Recognition", img)

            if cv2.waitKey(1) == 13:
                break

        video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = FaceRecognizer(root)
    root.mainloop()
