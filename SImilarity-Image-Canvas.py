from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image, ImageTk
import io
import torch
from torchvision import transforms as T
from tkinter import filedialog
import numpy as np
import torchvision.utils as vutils
from model import SiameseNetwork, BasicBlock, LambdaLayer, _weights_init
import torch.nn.functional as F
from matplotlib import pyplot as plt

CIFAR = False

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
W, H = 32, 32
if CIFAR:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
else:
    transform = T.Compose([
        T.ToTensor(),
    ])

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):

        #riferimenti dell'immagine
        self.c1_image_tk = None
        self.c2_image_tk = None

        self.device = torch.device("cuda")
        self.model = SiameseNetwork(num_blocks = [9, 9, 9]).to(self.device) #ResNet56
        if CIFAR:
            self.model.load_state_dict(torch.load('cifar100_150.pt'))
        else:
            self.model.load_state_dict(torch.load('siamese_network_MNIST_8.pt'))

        
        self.root = Tk()
        self.c1_tensor = None
        self.c2_tensor = None

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=50, to=80, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.choose_size_button.set(10)

        self.save_button = Button(self.root, text='save', command=self.save_image)
        self.save_button.grid(row=0, column=5)

        self.clear_button1 = Button(self.root, text='clear', command=self.clear_canvas1)
        self.clear_button1.grid(row=2, column=0)

        self.clear_button2 = Button(self.root, text='clear', command=self.clear_canvas2)
        self.clear_button2.grid(row=2, column=3)

        self.label = Label(self.root, text="Draw a picture or upload one", font=("Helvetica", 30))
        self.label.grid(row=3, column=0, columnspan=6)

        self.load_image_button1 = Button(self.root, text='load image', command=self.load_image1)
        self.load_image_button1.grid(row=2, column=1)

        self.load_image_button2 = Button(self.root, text='load image', command=self.load_image2)
        self.load_image_button2.grid(row=2, column=4)

        self.hello_world_button = Button(self.root, text='CONFRONTA', command=self.compare)
        self.hello_world_button.grid(row=4, column=0, columnspan=6)


        self.c1 = Canvas(self.root, bg='white', width=512, height=512)
        self.c1.grid(row=1, column=0, columnspan=2)

        self.c2 = Canvas(self.root, bg='white', width=512, height=512)
        self.c2.grid(row=1, column=3, columnspan=2)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c1.bind('<B1-Motion>', self.paint)
        self.c1.bind('<ButtonRelease-1>', self.reset)
        self.c2.bind('<B1-Motion>', self.paint)
        self.c2.bind('<ButtonRelease-1>', self.reset)

    def clear_canvas1(self):
        self.c1.delete("all")

    def clear_canvas2(self):
        self.c2.delete("all")

    def use_pen(self):
        self.activate_button(self.pen_button)

    def load_image1(self):
        self.load_image(self.c1)

    def load_image2(self):
        self.load_image(self.c2)

    def load_image(self, canvas):
            image_path = filedialog.askopenfilename(filetypes=[('All Files', '*.*')])
            if image_path:
                print(image_path)

                # Resize image
                image = Image.open(image_path)
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                image_tk = ImageTk.PhotoImage(image)

                # Save pointer of immage to avoid deleting from garbage collector
                if canvas == self.c1:
                    self.c1_image_tk = image_tk
                else:  # canvas == self.c2
                    self.c2_image_tk = image_tk

                # Crea l'immagine nel canvas
                canvas.create_image(0, 0, anchor='nw', image=image_tk)
            
    def compare(self):

        c1_image = self.convert_to_image(self.c1)
        c2_image = self.convert_to_image(self.c2)

        c1_image = c1_image.resize((H, W), Image.Resampling.LANCZOS)
        c2_image = c2_image.resize((H, W), Image.Resampling.LANCZOS)

        self.c1_tensor = transform(c1_image)
        self.c2_tensor = transform(c2_image)
        '''
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(self.c1_tensor.to(self.device)[:], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        '''
        self.c1_tensor = self.c1_tensor.unsqueeze(0)
        self.c2_tensor = self.c2_tensor.unsqueeze(0)
        self.c1_tensor, self.c2_tensor = self.c1_tensor.to(self.device), self.c2_tensor.to(self.device)
        output1 = self.model(self.c1_tensor)
        output2 = self.model(self.c2_tensor)
        distance = F.cosine_similarity(output1, output2, dim=1)
        #pred = (distance > 0.9945).to(torch.float32) #forse la distance è 0.4
        if CIFAR:
            pred = (distance >= 0.85).to(torch.float32) #forse la distance è 0.4
        else:
            pred = (distance >= 0.994).to(torch.float32) #forse la distance è 0.4

        print(distance)

        if pred[0] == 1.0:
            self.label.config(text="SAME CLASS!")
        else:
            self.label.config(text="DIFFERENT CLASS!")



    def save_image(self):
        transform = T.ToTensor()

        c1_image = self.convert_to_image(self.c1)
        c2_image = self.convert_to_image(self.c2)

        c1_image = c1_image.resize((H, W), Image.Resampling.LANCZOS)
        c2_image = c2_image.resize((H, W), Image.Resampling.LANCZOS)

        self.c1_tensor = transform(c1_image)
        self.c2_tensor = transform(c2_image)

        print(self.c1_tensor.shape)

        c1_image.save("input1.png")
        c2_image.save("input2.png")
        print("saved")
        
    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            event.widget.create_line(self.old_x, self.old_y, event.x, event.y,
                                     width=self.line_width, fill=paint_color,
                                     capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def convert_to_image(self, canvas):
        postscript = canvas.postscript(colormode='color')
        image = Image.open(io.BytesIO(postscript.encode('utf-8')))
        return image

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    paint_app = Paint()
