import re
import json
import csv
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common import exceptions
from selenium.webdriver.support.ui import WebDriverWait
browser=webdriver.Chrome("C:\\seleniumdriver\\chromedriver.exe")
type(browser) 
browser.get('https://www.zomato.com/ncr/pa-pa-ya-saket-new-delhi')
browser.implicitly_wait(9000)


import time
import threading
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode


delay = 100.0
button = Button.left
start_stop_key = KeyCode(char='s')
exit_key = KeyCode(char='e')


class ClickMouse(threading.Thread):
    def __init__(self, delay, button):
        super(ClickMouse, self).__init__()
        self.delay = delay
        self.button = button
        self.running = False
        self.program_running = True

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                mouse.click(self.button)
                time.sleep(self.delay)
            time.sleep(0.1)


mouse = Controller()
click_thread = ClickMouse(delay, button)
click_thread.start()


def on_press(key):
    if key == start_stop_key:
        if click_thread.running:
            click_thread.stop_clicking()
        else:
            click_thread.start_clicking()
    elif key == exit_key:
        click_thread.exit()
        listener.stop()


with Listener(on_press=on_press) as listener:
    listener.join()

'''rev=browser.find_element_by_css_selector("#reviews-container > div.notifications-content > div.res-reviews-container.res-reviews-area > div > div > div.mt0.ui.segment.res-page-load-more.zs-load-more > div.load-more.bold.ttupper.tac.cursor-pointer.fontsize2 > span.zred")
       

x=rev.location['x']
y=rev.location['y']


x=rev.location['x']
y=rev.location['y']
print(rev.location['x'])
print(rev.location['y'])'''
def find(browser):
    rev1=browser.find_element_by_css_selector("#reviews-container > div.notifications-content > div.res-reviews-container.res-reviews-area > div > div > div.mt0.ui.segment.res-page-load-more.zs-load-more > div.load-more.bold.ttupper.tac.cursor-pointer.fontsize2 > span.zred")
    if rev1:
        return rev1
    else:
        return False
for i in range(250):
    print(i)
    '''try:
        u=WebDriverWait(browser,50).until(find)
        u.click()
    except:
        break'''
    u=WebDriverWait(browser,50).until(find)
    u.click()


count=13010




def only_string(name):
	only_name=re.sub('[^a-zA-Z0-9,. ]+','', name)
	only_name=(only_name).strip()
	#only_name=re.sub("/ +/"," ",name)
	#only_name=re.sub("/^ /","",name)
	#only_name=re.sub("/ $/","",name)
	return(only_name)

def only_num(text):
	t=re.sub('[^0-9.]+','',text)
	return(t)

indian_review=[]
#indian_review+=[['user_id','user_name','business_name','local_add','address','rating','review']]

rest_name=browser.find_element_by_css_selector("#mainframe > div.wrapper.mtop > div > div.res-info-left.col-l-11 > div.ui.segment.res-header-overlay.vr > div > div.res-header-overlay.brbot > div:nth-child(1) > div.col-l-12 > h1 > a").get_attribute('innerHTML')
rest_name=only_string(rest_name)
#print(rest_name)

rest_local_add=browser.find_element_by_css_selector("#mainframe > div.wrapper.mtop > div > div.res-info-left.col-l-11 > div.ui.segment.res-header-overlay.vr > div > div.res-header-overlay.brbot > div:nth-child(1) > div.col-l-12 > div.mb5.pt5.clear > a").text
rest_local_add=only_string(rest_local_add)
#print(rest_local_add)

rest_add=browser.find_element_by_css_selector("#mainframe > div.wrapper.mtop > div > div.res-info-left.col-l-11 > div.ui.segments.mbot > div.row.ui.segment > div:nth-child(2) > div:nth-child(3) > div.borderless.res-main-address > div > span").text
rest_add=only_string(rest_add)
#print(rest_add) 


box=browser.find_elements_by_css_selector(".ui.segments.res-review-body.res-review")

for i in box:
	user_name=i.find_element_by_css_selector(".header.nowrap.ui.left").text
	user_name=only_string(user_name)
	#print(user_name)

	user_review=i.find_element_by_css_selector(".rev-text.mbot0").text	
	user_review=only_string(user_review)
	if(user_review != ''):
		#print(user_review)
		rev=1
	else:
		user_review=i.find_element_by_css_selector(".rev-text.mbot0.hidden").get_attribute("textContent")
		user_review=only_string(user_review)
		#print(user_review)

	user_rating=i.find_element_by_css_selector(".ttupper.fs12px.left.bold.zdhl2.tooltip").get_attribute('aria-label')
	user_rating=int(float(only_num(user_rating)))
	#print(user_rating)

	indian_review+=[[count,user_name,rest_name,rest_local_add,rest_add,user_rating,user_review]]
	count=count+1
	print(count)

#print(indian_review)

csv.register_dialect('myDialect',lineterminator='\n',quoting=csv.QUOTE_ALL,skipinitialspace=True)

with open('indian_review_delhi.csv', 'a') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in indian_review:
        writer.writerow(row)
f.close()


browser.quit()
