import nltk
from urllib import request
from bs4 import BeautifulSoup

url = "http://nonciclopedia.wikia.com/wiki/Ges%C3%B9"
html = request.urlopen(url).read()
print(html)
soup = BeautifulSoup(html)
raw = soup.get_text()
# raw = nltk.clean_html(html)
print(raw)

output_file = open('text.txt','wb')
output_file.write(raw.encode('utf8'))
