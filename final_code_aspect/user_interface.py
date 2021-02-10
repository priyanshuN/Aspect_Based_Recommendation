'''import tkinter
import tkinter.messagebox

top = tkinter.Tk()
def helloCallBack2():
   tkinter.messagebox.showinfo( "Hello Python", "Hello World")

def helloCallBack():
    C= tkinter.Button(top, text ="Hello", command = helloCallBack2)
    C.pack()
    

B = tkinter.Button(top, text ="Hello", command = helloCallBack)

B.pack()
top.mainloop()

from tkinter import *
import tkinter as tk
from tkinter import messagebox


firstwindow = Tk()
#Creating my first window
firstwindow.title("  this is my first window  ")
firstwindow.geometry("970x690")

#Adding image to window
#imagen = PhotoImage(file="imagen1.gif")
#etiqueta = Label(firstwindow, image = imagen).place(x = 370, y = 75)

#Adding exit button and title button
button1 = tk.Button(master = firstwindow, text = " DISCLAIMER ", font=('Arial', 18, 'bold'), height = 3).pack(side = tk.TOP)
button2 = tk.Button(master = firstwindow, text = "exit", font=('Courier New', 16), height = 3, command = exit).pack(side = tk.BOTTOM)


#Creating second window
def create_secondwindow():
 create_secondwindow1 = Tk()
 create_secondwindow1.geometry("370x290")
 create_secondwindow1.config(background="#FF9999")
 create_secondwindow1.title("this is my new window!!!")


# Create button that will open second window
button4 = tk.Button(master=firstwindow, text=" open new window ", font=('Courier New', 16), command=create_secondwindow).pack(side=tk.BOTTOM)


#Trying to open a new message when using a button in second window
def mensaje ():
 tk.Button(master=create_secondwindow, text=" Click here Ill give you a surprise! ", font=('Courier New', 16, "bold"), height=3, command=exit)
 tk.Button.grid(row=370, column=70)
 msg = messagebox.showinfo("Hello Python", "Hello World")
 create_secondwindow = Button(TOP, text="Hello", command=exit)
 create_secondwindow.place(x=350, y=50)


#Open message button
button5 = tk.Button(master=create_secondwindow, text=" open message ", font=('Courier New', 16), command=mensaje).pack(side=tk.TOP)

#Last exit button
tk.Button(master=create_secondwindow, text=" exit ", font=('Courier New', 16), command=exit).pack(side=tk.BOTTOM)


firstwindow.mainloop()
'''
liquids_s=["DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"]
food_variety_s=["FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"]
food_timing_s=["FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION"]
restaurant_features_s=["GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]
cat=['liquids','food_variety','food_timing','restaurant_features']
category={'liquids':liquids_s,
          'food_variety': food_variety_s,                                                                                                                                                                                        
          'food_timing':food_timing_s,
        'restaurant_features':restaurant_features_s}

            #print("Following are the categories of aspects");
for n,i in enumerate(cat):
    print(i);
#print("Enter the category name");
#user_cat=input("Enter the category name : ");
#print("Following are the aspects in this category");
#print(category[user_cat]);
#user_aspect=input("Enter the aspect : ");
#z=-1;
#for n,i in enumerate(aspects):
#    if(user_aspect==i):
#        z=n
import tkinter as tk

class Demo1:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr1=['liquids','food_variety','food_timing','restaurant_features']
        self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i,st in enumerate(cat):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()
        
        #self.arr.append(tk.Button(self.frame, text = 'click here to view graph', width = 25, command = self.new))
        self.var1 = tk.StringVar()
        self.label1 = tk.Label(self.frame, textvariable=self.var1)

        self.var1.set('enter user id here')
        self.label1.pack()
        self.textBox1=tk.Text(self.frame, height=2, width=10)
        self.textBox1.pack()
        self.var2 = tk.StringVar()
        self.label2 = tk.Label(self.frame, textvariable=self.var2)

        self.var2.set('enter item id here')
        self.label2.pack()
        self.textBox2=tk.Text(self.frame, height=2, width=10)
        self.textBox2.pack()
        self.buttonCommit=tk.Button(self.frame, text="graph",  width = 25,
                            command=lambda: self.retrieve_input())
        self.buttonCommit.pack()
        self.frame.pack()
#        self.button1 = tk.Button(self.frame, text = 'New Window', width = 25, command = self.new_window1)
#        self.button1.pack()
#        self.frame.pack()
#        self.button2 = tk.Button(self.frame, text = 'New Window', width = 25, command = self.new_window1)
#        self.button2.pack()
#        self.frame.pack()
    def retrieve_input(self):
        inputValue1=self.textBox1.get("1.0","end-1c")
        inputValue2=self.textBox2.get("1.0","end-1c")
        print(inputValue1,inputValue2)
    def new_window1(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo3(self.newWindow,self.txte,self.mdl)
    def new_window2(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo7(self.newWindow,self.txte,self.mdl)
    def new_window3(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo8(self.newWindow,self.txte,self.mdl)
    def new_window4(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo9(self.newWindow,self.txte,self.mdl)

class Demo2:
    def __init__(self, master,text,mdl):
        #self.txt=''
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        #txt=self.fn()
        self.texte=mdl.add()
        self.var = tk.StringVar()
        self.label = tk.Label(self.frame, textvariable=self.var)

        self.var.set(self.texte)
        self.label.pack()
        self.show_review=tk.Button(self.frame, text ='show reviews', width = 25, command = self.new_window4)
        self.show_review.pack()
        self.quitButton = tk.Button(self.frame, text ='exit', width = 25, command = self.close_windows)
        self.quitButton.pack()

        self.frame.pack()
        
    def fn(self):
        return str(1+2)
    def close_windows(self):
        self.master.destroy()
    def new_window4(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo10(self.newWindow,self.texte,self.mdl)


class Demo10:
    def __init__(self, master,text,mdl):
        #self.txt=''
        self.master = master
        self.frame = tk.Frame(self.master)
        #txt=self.fn()
        #self.texte=mdl.add()


        S = tk.Scrollbar(self.frame)
        T = tk.Text(self.frame, height=40, width=50)
        S.pack(side='right', fill="y")
        T.pack(side='left', fill="y")
        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)
        quote = 'fvdghdgdasf'
        T.insert('end', quote)


        
        self.quitButton = tk.Button(self.frame, text ='exit', width = 15, command = self.close_windows)
        self.quitButton.pack()
        self.frame.pack()
        
    def fn(self):
        return str(1+2)
    def close_windows(self):
        self.master.destroy() 

class Demo3:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr2=[]
        self.arr1=["DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"]
        
        #self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i in range(len(self.arr1)):
            self.arr2.append(self.new_w(i))
        for i,st in enumerate(self.arr1):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()


    def new_w(self,i):
        def new_window1():
            self.newWindow = tk.Toplevel(self.master)
            self.app = Demo2(self.newWindow,i,self.mdl)
        return new_window1



class Demo7:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr2=[]
        self.arr1=["FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"]
        #self.geometry("970x690")
        #self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i in range(len(self.arr1)):
            self.arr2.append(self.new_w(i))
        for i,st in enumerate(self.arr1):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()


    def new_w(self,i):
        def new_window1():
            self.newWindow = tk.Toplevel(self.master)
            self.app = Demo2(self.newWindow,i,self.mdl)
        return new_window1


class Demo8:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr2=[]
        self.arr1=["GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]

        #self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i in range(len(self.arr1)):
            self.arr2.append(self.new_w(i))
        for i,st in enumerate(self.arr1):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()


    def new_w(self,i):
        def new_window1():
            self.newWindow = tk.Toplevel(self.master)
            self.app = Demo2(self.newWindow,i,self.mdl)
        return new_window1


class Demo9:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr2=[]
        self.arr1=["GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]

        #self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i in range(len(self.arr1)):
            self.arr2.append(self.new_w(i))
        for i,st in enumerate(self.arr1):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()


    def new_w(self,i):
        def new_window1():
            self.newWindow = tk.Toplevel(self.master)
            self.app = Demo2(self.newWindow,i,self.mdl)
        return new_window1


class Demo4:
    def __init__(self, master,txte,mdl):
        self.txte=txte
        self.textee=mdl.add()
        self.master = master
        self.mdl=mdl
        self.frame = tk.Frame(self.master)
        self.arr=[]
        self.arr1=['liquids','food_variety','food_timing','restaurant_features']
        
        self.arr2=[self.new_window1,self.new_window2,self.new_window3,self.new_window4]
        for i,st in enumerate(cat):
            self.arr.append(tk.Button(self.frame, text = st, width = 25, command = self.arr2[i]))
            self.arr[i].pack()
            self.frame.pack()



    def new_window1(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow,self.txte,self.mdl)
        
    def new_window2(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow,self.txte,self.mdl)
    def new_window3(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow,self.txte,self.mdl)
    def new_window4(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow,self.txte,self.mdl)

class model:
    def add(self):
        return '3'

def main(): 
    root = tk.Tk()
    root.geometry("970x690")
    mdl=model()
    app = Demo1(root,'dfg',mdl)
    root.mainloop()

if __name__ == '__main__':
    main()


from tkinter import *

top = Tk()
L1 = Label(top, text="User Name")
L1.pack( side = LEFT)
E1 = Entry(top, bd =5)

E1.pack(side = RIGHT)

top.mainloop()



#from tkinter import *
#root=Tk()
#def retrieve_input():
#    inputValue1=textBox1.get("1.0","end-1c")
#    inputValue2=textBox2.get("1.0","end-1c")
#    print(inputValue1,inputValue2)
#textBox1=Text(root, height=2, width=10)
#textBox1.pack()
#textBox2=Text(root, height=2, width=10)
#textBox2.pack()
#buttonCommit=Button(root, height=1, width=10, text="Commit", 
#                    command=lambda: retrieve_input())
#command=lambda: retrieve_input() >>> just means do this when i press the button
#buttonCommit.pack()

mainloop()
