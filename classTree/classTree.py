import sys
import random
import csv

MAX_HEIGHT = 5
MAX_DATA = 10
FOLDS = 5
DATACSV = 'data_banknote_authentication.txt'

def load_csv(fileName):
	# open file and read data
	file = open(fileName, "rb")
	lines = csv.reader(file)
	data = list(lines)
	# convert data from strings to floats
	for column in range(len(data[0])):
		for row in data:
			row[column] = float(row[column])
	#return list(lines)
	return data

# Print a decision tree
def print_classTree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['feature']+1), node['value'])))
		print_classTree(node['left'], depth+1)
		print_classTree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

# create and evaluate a classification decision tree
def class_tree(trainData, testData):
	# make the tree
	tree = decision_tree(trainData)
	predictions = list()
	print_classTree(tree)
	# get predictions on new data
	for element in testData:
		predictions.append(predict(tree, element))

	return predictions

# create a decision tree using the training data
def decision_tree(trainData):
	# create root of tree
	root = max_split(trainData)
	# recursive splitting
	split_node(1, root)
	return root

# recursive splitting of nodes in decision tree
# stopping point is data on one side only, MAX_DEPTH, or less than MAX_DATA
def split_node(currHeight, currNode):
	left, right = currNode['split']

#	del(node['data'])
	# no split required, no data 
	if not left or not right:
		currNode['left'] = make_leaf(left+right)
		currNode['right'] = make_leaf(left+right)
		return

	# no split required, at MAX HEIGHT of tree
	if currHeight >= MAX_HEIGHT:
		currNode['left'] = make_leaf(left)
		currNode['right'] = make_leaf(right)
		return

	# recurse left
	if len(left) <= MAX_DATA:
		currNode['left'] = make_leaf(left)
	else:
		currNode['left'] = max_split(left)
		split_node(currHeight + 1, currNode['left'])
	
	# recurse right
	if len(right) <= MAX_DATA:
		currNode['right'] = make_leaf(right)
	else:
		currNode['right'] = max_split(right)
		split_node(currHeight + 1, currNode['right'])

def make_leaf(group):
	# classification of data
	classes = [row[-1] for row in group]
	return max(set(classes), key=classes.count)

def check_split(feature, value, data):
	leftSplit  = list()
	rightSplit = list()
	# determine data split by feature and value
	for row in data:
		if row[feature] < value:
			leftSplit.append(row)
		else:
			rightSplit.append(row)

	return leftSplit, rightSplit

def max_split(data):
	classes = list(set(row[-1] for row in data))
#	splitFeature = 999
#	splitValue = 999
	splitScore = sys.maxint
#	splitData = None

	for feature in range(len(data[0])-1):
		for row in data:
			testSplit = check_split(feature, row[feature], data)
			giniScore = eval_gini(testSplit, classes)
#			print('X%d < %.3f GiniScore=%.3f'% ((feature+1), row[feature], giniScore))
			if giniScore < splitScore:
				splitFeature = feature
				splitValue = row[feature]
				splitScore = giniScore
				splitData = testSplit

#	print splitFeature, splitValue		
	return {'feature':splitFeature, 'value':splitValue, 'split':splitData}

def eval_gini(split, vals):
    # number of samples
    samples = float(sum([len(data) for data in split]))

    #sum for each class
    score = 0.0
    for data in split:
        size = float(len(data))
        if size > 0:
            count = 0.0
            # get proportions for each class
            for value in vals:
                proportion = [row[-1] for row in data].count(value)/size
                count += proportion * proportion
            # calc score for gini evaluation
            score += (1.0-count)*(size/samples)
    
    return score

# Make a prediction on data using the decision tree
# prediction is at the leaf recursed to
def predict(example, node):
	if example[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(example, node['left'])
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(example, node['right'])
		else:
			return node['right']

# split data into folds for validation
def k_folds(data):
	folds = list()
	tmpData = list(data)
	foldSize = len(data)/FOLDS
	# split data into k folds
	for cnt in range(FOLDS):
		fold = list()
		while len(fold) < foldSize:
			selection = random.randrange(len(tmpData))
			fold.append(tmpData.pop(selection))
		folds.append(fold)
	return folds

def evaluate(predictions, actual):
	num_correct = 0
	for cnt in range(len(actual)):
		if actual[cnt] == predictions[cnt]:
			num_correct += 1
	return num_correct / len(actual) 

def eval_tree(data):
	folds = k_folds(data)
	foldsAccuracy = list()
	
	for fold in folds:
		trainSet = list(folds)
		trainSet.remove(fold)
		trainSet = sum(trainSet, [])
		testSet = list()
		
		for element in fold:
			tmpEle = list(element)
			testSet.append(tmpEle)
			tmpEle[-1] = None
		predictions = class_tree(trainSet, testSet)
		actual = [element[-1] for element in fold ]
		accuracy = evaluate(actual, predictions)
		foldsAccuracy.append(accuracy)
	
	return foldsAccuracy


# MAIN PROGRAM

random.seed(1)
# load data
data = load_csv(DATACSV)
# create and evaluate decision tree
evaluation = eval_tree(data)
# print results
#for foldNum in range(FOLDS)
print ('Accuracys: %s', evaluation)
print ('Mean accuracy: %.3f%%' % (sum(evaluation)/(len(evaluation))))

### TESTING ###
# test eval_gini
#print(eval_gini([[[1,1],[1,0]],[[1,1], [1,0]]], [0,1]))
#print(eval_gini([[[1,0],[1,0]],[[1,1], [1,1]]], [0,1]))

# eval data
#data = [[2.771244718,1.784783929,0],
#	[1.728571309,1.169761413,0],
#	[3.678319846,2.81281357,0],
#	[7.497545867,3.162953546,1],
#	[9.00220326,3.339047188,1],
#	[7.444542326,0.476683375,1],
#	[10.12493903,3.234550982,1],
#	[6.642287351,3.319983761,1]]

# test splitting
#split = max_split(data)
#print('Split: [X%d < %.3f]' % ((split['feature']+1), split['value']))

# test decision tree
#tree = decision_tree(data)
#print_classTree(tree)

# test predictor
#  predict with a stump
#stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
#for row in data:
#	print type(stump), type(row)
#	prediction = predict(row, stump)
#	print('Expected=%d, Got=%d' % (row[-1], prediction))

# temp testing!!!


