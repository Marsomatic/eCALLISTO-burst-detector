import os

burstPath = 'data/validation/bursts'
dudsPath = 'data/validation/duds'
burstFilenames = os.listdir(burstPath)
dudsFilenames = os.listdir(dudsPath)

count = 0
for dud in dudsFilenames:
    if dud in burstFilenames:
        os.remove(dudsPath + '/' + dud)
        print(dudsPath + '/' + dud)
        count += 1
        
print(count)