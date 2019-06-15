"""
Team Id: HC#3266
Author List: Hemanth Kumar K.L , Aman Bhat T, Mahantesh R, Nischith B.O.
File Name: python_to_robot.py
Theme: Homecoming
"""


import serial
import io
from datetime import datetime

f = open(r"example_1.txt", "r")
str=f.readline().replace('\n', ' ')
str+=r'\n'
str+=f.readline()
str+="#"
print("start time")
print(datetime.now().time())
print("\nserial data transmitted")
print(str)

s=serial.Serial(port="COM4", baudrate=9600)
if(s.isOpen):
    s.write(str.encode('utf-8'))
    s.close()
print("\nend time")
print(datetime.now().time())