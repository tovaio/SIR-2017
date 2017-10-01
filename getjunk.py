import os
from PIL import Image
import urllib.request
from bs4 import BeautifulSoup

files = []
for i in range(1, 925):
    i = str(i)
    i = list(i)
    z = 5 - len(i)
    for k in range(z):
        i.insert(0,'0')
    i = ''.join(i)
    i += ".jpg"
    files.append(i)
    
URL = "https://unsplash.it/64/128/?random"
for i in range(924):
    urllib.request.urlretrieve(URL, files[i])
