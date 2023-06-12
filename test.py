import json
file = open('test.json','r')
cons = json.load(file)
print("name:{}\nage:{}".format(cons['name'],cons['age']))