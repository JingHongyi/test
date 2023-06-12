import json
file = open('test.json','r')
cons = json.load(file)
print("name:{}\nage:{}school:{}\ncity:{}".format(cons['name'],cons['age'],cons['school'],cons['city']))
