#!/usr/bin/env python
import sys
import math
from random import randint

train_features = []
train_data = []
train_class = []
train_labels = []
feature_vec_length = 0
m = 0

class Node(object):
	def __init__(self, data, feature, label, majority):
		self.data = data
		self.feature = feature
		self.label = label
		self.majority = majority
		self.children = []

	def add_child(self, obj):
		self.children.append(obj)

def DetermineCandidateSplits(data):
	candidate_splits = []
	for i in range(len(train_features)):
		feature = train_features[i]
		split = []
		tlist = [item[i] for item in data]
		ulist = list(set(tlist))
		
		if feature[1] == None: #For numeric features
			tlist = [float(item[i]) for item in data]
			ulist = list(set(tlist))
			if len(ulist) == 1:
				pass
			else:
				ulist.sort()
				means = []
				for j in range(len(ulist)-1):
					means.append((float(ulist[j]) + float(ulist[j+1]))/2)
				split.append(feature[0])
				split.append(i)
				split.append(means)
			if len(split) > 0:
				candidate_splits.append(split)
		else: #For nominal features
			tlist = [item[i] for item in data]
			ulist = list(set(tlist))
			if len(ulist) == 1:
				pass
			else:
				split.append(feature[0]) #name of the nominal feature
				split.append(i)		#occurrence in the ARFF file
				split.append(feature[1])	#feature values
			if len(split) > 0 :
				candidate_splits.append(split)
	return candidate_splits


def FindBestSplit(data, C):
	y_labels = [item[feature_vec_length] for item in data]
	count = [ y_labels.count(train_labels[0]), y_labels.count(train_labels[1]) ]
	p = [float(count[0])/(count[0] + count[1]), float(count[1])/(count[0] + count[1]) ]
	H_y = -1*p[0]* math.log(p[0]) -1*p[1]*math.log(p[1])
	
	info_gain = float(0)
	info_gain_feature = None
	for i in range(len(C)): #iterate over every candidate feature
		can_values = [ [item[C[i][1]], item[feature_vec_length]] for item in data] #values for the candidate feature
		if isinstance(C[i][2][0], float): #for numeric feature
			for j in C[i][2]: # iterate over all the thresholds
				t1 = len([k for k in can_values if float(k[0]) <=j and k[1] == train_labels[0]]) 
				t2 = len([k for k in can_values if float(k[0]) <=j and k[1] == train_labels[1]])
				t3 = len([k for k in can_values if float(k[0]) >j and k[1] == train_labels[0]]) 
				t4 = len([k for k in can_values if float(k[0]) >j and k[1] == train_labels[1]])

				probs12 = [float(t1)/(t1+t2), float(t2)/(t1+t2)]
				entropy12 = 0.0
				for item in probs12 :
					if item != 0:
						entropy12 += -1*item*math.log(item)
				probs34 = [float(t3)/(t3+t4), float(t4)/(t3+t4)]
				entropy34 = 0.0
				for item in probs34 :
					if item != 0:
						entropy34 += -1*item*math.log(item)
				H_y_feature = (entropy12*(t1+t2) + entropy34*(t3+t4))/(t1+t2+t3+t4)
				
				gain = H_y - H_y_feature
				if gain > info_gain:
					info_gain = gain
					info_gain_feature = [C[i][0], C[i][1], j] #return the threshold
			

		else: # for nominal feature
			conds_count = []
			for j in C[i][2]:
				t1 = len([k for k in can_values if k[0] == j and k[1] == train_labels[0]]) 
				t2 = len([k for k in can_values if k[0] == j and k[1] == train_labels[1]])
				conds_count.append([t1,t2])

			probs = []
			for item in conds_count:
				if item[0] != 0 or item[1] != 0:
					probs.append([float(item[0])/(item[0] + item[1]), float(item[1])/(item[0] + item[1])])
				else :
					probs.append([0.0,0.0])
			
			entropies = []
			for item in probs:
				e = float(0)
				if item[0] != 0:
					e += -1* item[0]* math.log(item[0])
				if item[1] != 0:
					e += -1* item[1]*math.log(item[1])
				entropies.append(e)

			H_y_feature = float(0)
			for j in range(len(conds_count)):
				H_y_feature += (float(conds_count[j][0] + conds_count[j][1])/len(data))*entropies[j]
			gain = H_y - H_y_feature
			if gain > info_gain:
				info_gain = gain
				info_gain_feature = C[i]
			
	if info_gain == 0.0:
		return None
	return info_gain_feature
				

def MakeSubTree(data, n):
	global m
	C = DetermineCandidateSplits(data)
	data_labels = [item[feature_vec_length] for item in data]

	#Stopping criterion
	if (len(data) < m or len(list(set(data_labels))) == 1 or len(C) == 0):

		count_labels = [data_labels.count(train_labels[0]), data_labels.count(train_labels[1])]
		if len(data) == 0:
			n.label = n.majority
		if count_labels[0] > count_labels[1]:
			n.label = train_labels[0]
		elif count_labels[0] < count_labels[1]:
			n.label = train_labels[1]
		else:
			n.label = n.majority
		return

	else:
		S = FindBestSplit(data, C) # S = [feature, index in ARFF, list of values for nominal feature or threshold for numeric feature]
				
		if S == None: #yet another stopping criterion
			count_labels = [data_labels.count(train_labels[0]), data_labels.count(train_labels[1])]
			if len(data) == 0:
				n.label = n.majority
			if count_labels[0] > count_labels[1]:
				n.label = train_labels[0]
			elif count_labels[0] < count_labels[1]:
				n.label = train_labels[1]
			else:
				n.label = n.majority
		elif isinstance(S[2], float): #for numeric feature with threshold			
			data1 = [item for item in data if float(item[S[1]]) <= S[2]]
			data2 = [item for item in data if float(item[S[1]]) > S[2]]
			c1 = len([item for item in data1 if item[feature_vec_length] == train_labels[0]])
			c2 = len([item for item in data1 if item[feature_vec_length] == train_labels[1]])
			c3 = len([item for item in data2 if item[feature_vec_length] == train_labels[0]])
			c4 = len([item for item in data2 if item[feature_vec_length] == train_labels[1]])
			if c1 > c2:
				majority1 = train_labels[0]
			elif c2 > c1:
				majority1 = train_labels[1]
			else:
				majority1 = n.majority
			n1 = Node( [c1, c2], S[0] + " <= " + str('%.6f' % round(S[2], 6)), None, majority1)
			if c3 > c4:
				majority2 = train_labels[0]
			elif c4 > c3:
				majority2 = train_labels[1]
			else:
				majority2 = n.majority
			n2 = Node( [c3, c4], S[0] + " > " + str('%.6f' % round(S[2], 6)), None, majority2)
			n.add_child(n1)
			n.add_child(n2)
			#n.feature = S[0]
			#print "generating sub tree", n1.data
			MakeSubTree(data1, n1)
			#print "generating sub tree", n2.data
			MakeSubTree(data2, n2)
		else : # for nominal feature
			for j in S[2]:
				data_sub = [item for item in data if item[S[1]] == j]
				c1 = len([item for item in data_sub if item[feature_vec_length] == train_labels[0]])
				c2 = len([item for item in data_sub if item[feature_vec_length] == train_labels[1]])
				if c1 > c2:
					majority1 = train_labels[0]
				elif c2 > c1:
					majority1 = train_labels[1]
				else:
					majority1 = n.majority
				n_sub = Node([c1, c2], S[0] + " = " + j, None, majority1)
				n.add_child(n_sub)
				#print "generating sub tree", n_sub.data
				MakeSubTree(data_sub, n_sub)
			#n.feature = S[0]
				
		return

def traverse(node, depth, flag, f):
	indent = ""
	if flag != 0:
		for i in range(depth):
			indent += "|" + "\t"
		if node.label != None:
			f.write(indent + node.feature + " [" + str(node.data[0]) + " " + str(node.data[1]) + "]: "+ node.label + "\r\n")
		else:
			f.write(indent + node.feature + " [" + str(node.data[0]) + " " + str(node.data[1]) + "]"+"\r\n")
	
		depth += 1
	for child in node.children:
		traverse(child, depth, 1, f)
	
def PredictOutcome(sample, node):
	while node.label == None:
		for child in node.children:
			vec = child.feature.split(" ")
			feature_names = [item[0] for item in train_features]
			i = feature_names.index(vec[0])
			if vec[1] == "=": # for nominal feature
				if vec[2] == sample[i]:
					node = child
					
			else: # for numeric feature
				if vec[1] == "<=":
					if float(sample[i]) <= float(vec[2]):
						node = child
				elif vec[1] == ">":
					if float(sample[i]) > float(vec[2]):
						node = child
	return node.label
	
				
		

def main():
	global m
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	m = int(sys.argv[3])

	file = open(train_file, 'r')
	train_arff = file.readlines()

	global train_features
	global train_data
	global train_labels
	global train_class
	global feature_vec_length

	#Parse the ARFF file to get the training data, features
	for line in train_arff:
		if line.lower().startswith("@relation") or line.lower().startswith("@data"):
			pass
		elif line.lower().startswith("@attribute"):
			section = line.split('  ', 2)
			attr = []
			attr.append(section[1].replace("'", ""))
			if section[2].startswith('{'):
				attr_values = section[2].replace("{","").replace("}","").split(",")
				for i in range(len(attr_values)):
					attr_values[i] = attr_values[i].replace(" ", "").replace("\n","").replace("\r","")
				attr.append(attr_values)
			else:
				attr.append(None) # for numeric features
			train_features.append(attr)
	
		elif line.startswith("%"):
			pass
		else :
			vec=line.split(",")
			for i in range(len(vec)):
				vec[i] = vec[i].replace(" ", "").replace("\n", "").replace("\r","")
			if len(vec) == len(train_features):
				train_data.append(vec)
			#train_data.append(line)

	#Get the class label out from the list of features
	train_class = train_features[len(train_features)-1]
	train_labels = train_class[1]
	train_features = train_features[:-1]
	feature_vec_length = len(train_features)

	#print train_class, train_labels, train_features

	#print train_data

	c1 = len([item for item in train_data if item[feature_vec_length] == train_labels[0]])
	c2 = len([item for item in train_data if item[feature_vec_length] == train_labels[1]])
	if c1 > c2:
		majority = train_labels[0]
	elif c1 < c2:
		majority = train_labels[1]
	else:
		majority = None
	root = Node([c1,c2], None, None, majority)
	MakeSubTree(train_data, root)

	# -----------------------------Predict class labels for the test data----------------------------------------------
	file = open(test_file, 'r')
	test_arff = file.readlines()
	test_data = []
	for line in test_arff:
		if line.startswith("@"):
			pass
		else:
			vec=line.split(",")
			for i in range(len(vec)):
				vec[i] = vec[i].replace(" ", "").replace("\n", "").replace("\r","")
			test_data.append(vec)

	#print test_data

	
	predicted_labels = []
	count_pred = 0
	for item in test_data:
		label = PredictOutcome(item, root)
		if label == item[feature_vec_length]:
			count_pred += 1
		predicted_labels.append(label)
		
	f = open('m='+str(m)+'.txt', 'w')
	traverse(root, 0, 0, f)
	f.write("<Predictions for the Test Set Instances>\r\n")
	for i in range(len(test_data)):
		f.write(str(i+1) + ": Actual: " + test_data[i][feature_vec_length]+ " Predicted: "+predicted_labels[i] +"\r\n")
	f.write("Number of correctly classified: "+str(count_pred)+" Total number of test instances: "+str(len(test_data))+"\r\n")
	f.close()

"""
	#What follows is supporting code for Parts 2 and 3 and hence is commented out "
	accuracies = []	
	for i in range(10):
		train_data_subset = []
		size = int(0.5 * len(train_data))
		track_list = []
		for j in range(size):
			flag = 0
			while flag == 0:
				r = randint(0,len(train_data)-1)
				if r not in track_list:
					track_list.append(r)
					train_data_subset.append(train_data[r])
					flag = 1
		#print "train_data for round ", i, "is :", train_data_subset

		c1 = len([item for item in train_data_subset if item[feature_vec_length] == train_labels[0]])
		c2 = len([item for item in train_data_subset if item[feature_vec_length] == train_labels[1]])
		if c1 > c2:
			majority = train_labels[0]
		elif c1 < c2:
			majority = train_labels[1]
		else:
			majority = None
		root = Node([c1,c2], None, None, majority)
		MakeSubTree(train_data_subset, root)

		predicted_labels = []
		count_pred = 0
		for item in test_data:
			label = PredictOutcome(item, root)
			if label == item[feature_vec_length]:
				count_pred += 1
			predicted_labels.append(label)
		accuracies.append (float(count_pred)/len(test_data))

	print " accuracies : ", accuracies
 	print min(float(s) for s in accuracies), max(float(s) for s in accuracies), sum(float(s) for s in accuracies)/10
"""		 
main()



		
		

		
		
	
