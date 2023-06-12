import json
file = open('test.json','r')
cons = json.load(file)
print(cons['2'])