import json
file = open('test.json','r')
cons = json.load(file)
print("name:{}\nage:{}\nschool:{}\n".format(cons['name'],cons['age'],cons['school']))
