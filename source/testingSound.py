# import required module
from playsound import playsound
import os
x = ''
f = os.path.split(os.getcwd())[0]+ "/sound"
for file in os.listdir(f):
    if file.endswith(".mp3"):
        x = os.path.join(f, file)
playsound(x)
print('playing sound using  playsound')