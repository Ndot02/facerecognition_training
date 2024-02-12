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
import os
import numpy as np


class Training:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")  # setting windows screen size
        self.root.title("Face Recognition System")

        title_lbl = Label(
            self.root,
            text="Train Dataset",
            font=("times new roman", 35, "bold"),
            bg="white",
            fg="black",
        )
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # button
        b1_l = Button(
            self.root,
            text="Train Data",
            command=self.train_image,
            cursor="hand2",
            font=("times new roman", 30, "bold"),
            bg="maroon",
            fg="white",
        )
        b1_l.place(x=0, y=620, width=1530, height=60)

    ########## Training images LBPH algorithm ################
    def train_image(self):
        data_dir = "data"
        path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert("L")  ## Converting to grayscale
            imageNp = np.array(img, "uint8")
            id = int(os.path.split(image)[1].split(".")[1])

            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training", imageNp)
            cv2.waitKey(1) == 13
        ids = np.array(ids)

        ########### Training classifier ##########3
        clf = cv2.face.LBPHFaceRecognizer_create()

        clf.train(faces, ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("Result", "Training dataset completed!!")


if __name__ == "__main__":
    root = Tk()
    obj = Training(root)
    root.mainloop()
