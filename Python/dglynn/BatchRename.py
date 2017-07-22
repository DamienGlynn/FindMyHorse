import os

class BatchRename(object):
	"""docstring for BatchRename"""
	def process(path, prefix):
		i     = 1
		files = os.listdir(path)
		for file in files:
			os.rename(os.path.join(path, file), os.path.join(path, prefix+str(i)+'.jpg'))
			i = i+1
		print('%d "%s" files renamed.' % (i-1, path))


BatchRename.process('../images/horses/new', '')
# BatchRename.process('pos', '')


# BatchRename.process('test/cows',   'cow')
# BatchRename.process('test/dogs',   'dog')
# BatchRename.process('test/horses', 'horse')
# BatchRename.process('test/people', 'person')
# BatchRename.process('test/pigs',   'pig')
# BatchRename.process('test/sheep',  'sheep')
# BatchRename.process('test/trees',  'tree')



# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Horse/horse', 'horse')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Horse/horse_mirror', 'horse')


# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Cows/cows', 'cow')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Cows/cows_mirror', 'cow')

# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Dogs/GermanShepard/german_shepard', 'german_shepard')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Dogs/GermanShepard/german_shepard_mirror', 'german_shepard')

# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/People/people', 'people')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/People/people_mirror', 'people')

# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Pigs/pigs', 'pig')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Pigs/pigs_mirror', 'pig')

# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Sheep/sheep', 'sheep')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Sheep/sheep_mirror', 'sheep')

# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Trees/trees', 'tree')
# BatchRename.process('D:/Documents/DIT/DT228-4/Final Year Project/images/Facebook/Images/Other/Trees/trees_mirror', 'tree')

