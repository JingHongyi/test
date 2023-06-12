import json
file = open('test.json','r')
cons = json.load(file)
print("name:{}\nage:{}school:{}\n".format(cons['name'],cons['age'],cons['school']))
