import bs4
import requests
import os
from parseEvents import getEvents

bursts = getEvents()

MAINPATH = 'http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/2023/'

monthResponse = requests.get(MAINPATH)
monthData = bs4.BeautifulSoup(monthResponse.text, 'html.parser')
monthLinks = monthData.find_all('a')[5:]


# Download duds from 2023
# for i in range(12):
#     dayResponse = requests.get(MAINPATH + monthLinks[i].get_text('href'))
#     dayData = bs4.BeautifulSoup(dayResponse.text, 'html.parser')
#     dayLinks = dayData.find_all('a')[5:]
#     for day in dayLinks:
#         response = requests.get(MAINPATH + monthLinks[i].get_text('href') + day.get_text('href'))
#         data = bs4.BeautifulSoup(response.text, 'html.parser')
#         links = data.find_all('a')[5:]
#         for link in links:
#             filename = link.get_text('href')
#             file = requests.get(MAINPATH + monthLinks[i].get_text('href') + day.get_text('href') + filename)
#             with open(f'data/duds/{filename}', mode='wb') as f1:
#                 f1.write(file.content)
#                 print(filename, 'downloaded')

# Download bursts from 2023
for burst in bursts:
    month = burst[0][1]
    day = burst[0][2]
    hour = burst[0][3]
    minute = burst[0][4]
    stations = burst[3]
    
    if month < 10:
        if day<10:
            url = f'http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/2023/0{month}/0{day}/'
        else:
            url = f'http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/2023/0{month}/{day}/'
    else:
        if day<10:
            url = f'http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/2023/{month}/0{day}/'
        else:
            url = f'http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/2023/{month}/{day}/'
            
    response = requests.get(url)
    data = bs4.BeautifulSoup(response.text, 'html.parser')
    
    # for i in range(5, len(data.find_all('a'))):
    links = data.find_all('a')[5:]
    for link in links:
        filename = link.get_text('href')
        for station in burst[3]:
            checkHour = int(filename.split('_')[2][0:2])
            checkMinute = int(filename.split('_')[2][2:4])
            # print(f'checking at time: {checkHour}:{checkMinute}', )
            if station in filename and int(hour) == checkHour and int(minute)//15 * 15 == checkMinute:
                response = requests.get(url+link['href'])
                with open(f'data/bursts/{filename}', mode='wb') as f1:
                    f1.write(response.content)
                    print(filename, 'downloaded')
