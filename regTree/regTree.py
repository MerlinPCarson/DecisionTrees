import sys
import random
import csv
import numpy as np

MAX_HEIGHT = 5
MAX_DATA = 10
FOLDS = 2
DATACSV = 'Carseats.csv'
SHELVELOC = 6
URBAN = 9
US = 10

def load_csv(fileName):
	# open file and read data
    file = open(fileName, "rb")
    lines = csv.reader(file)
    data = list(lines)[1:]          # leave out first row/headers
	# convert data from strings to floats
    for column in range(len(data[0])):
        if column != SHELVELOC and column != URBAN and column != US:
            for row in data:
                row[column] = float(row[column])
	
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
		predictions.append(predict(element, tree))

	return predictions

# create a decision tree using the training data
def decision_tree(trainData):
	# create root of tree
    root = max_split(trainData)
    print 'root', len(root['split'])
    print root['split'][0]
    print root['split'][1]
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
    sales = list(set(row[0] for row in data))
#    splitFeature = 999
#    splitValue = 999
    splitRSS = sys.maxint
#    splitData = None
    for feature in range(1, len(data[0])):
        if feature == SHELVELOC or feature == URBAN or feature == US:
 #           print "ignoring categorical data"
            continue
        for row in data:
            testSplit = check_split(feature, row[feature], data)
            rss = eval_RSS(testSplit, row[0])

#            print('X%d < %.3f RSS=%.3f'% ((feature+1), row[feature], rss))
            if rss < splitRSS:
				splitFeature = feature
				splitValue = row[feature]
				splitRSS = rss
				splitData = testSplit

    print splitFeature, splitValue		
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

def eval_RSS(split, actualScore):
    # RSS for the split
    rss = 0.0
    # summ RSS for left side and right side
    for data in split:
        # scores from data for median
        scores = list()
        for row in data:
            scores.append(row[0])
        if len(data) > 0:
            rss += (np.median(scores)-actualScore)**2

    return rss/2        # average of left and right side error

# Make a prediction on data using the decision tree
# prediction is at the leaf recursed to
def predict(example, node):
	if example[node['feature']] < node['value']:
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
#		print (actual[cnt], '?=', predictions[cnt])
		if actual[cnt] == predictions[cnt]:
			num_correct += 1
#	print num_correct		
	return num_correct / float(len(actual)) 

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
#		print predictions
		actual = [element[-1] for element in fold ]
#		print actual
		accuracy = evaluate(predictions, actual)
		foldsAccuracy.append(accuracy)
	
	return foldsAccuracy


# MAIN PROGRAM

random.seed(1)

# load data
data = load_csv(DATACSV)        

# create and evaluate decision tree
evaluation = eval_tree(data)
# print results

print ('Fold Accuracies: ', evaluation)
print ('Mean accuracy: %.3f' % (sum(evaluation)/(len(evaluation))))