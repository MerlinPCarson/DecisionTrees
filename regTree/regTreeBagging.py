import sys
import random
import csv
import numpy as np

MAX_HEIGHT = 5
MAX_DATA = 10
FOLDS = 2
NUM_TREES = 10
BAG_RATIO = .67
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
        print('%s[X%d < %.3f]' % ((depth*' ', (node['feature']), node['value'])))
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
    # recursive splitting, creating the tree
    split_node(1, root)
    return root

# recursive splitting of nodes in decision tree
# stopping point is data on one side only, MAX_DEPTH, or less than MAX_DATA
def split_node(currHeight, currNode):
    left, right = currNode['split']

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

def make_leaf(data):
	# classification of data
    scores = [row[0] for row in data]
    return np.median(scores)

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

    splitRSS = sys.maxint

    for feature in range(1, len(data[0])):
        if feature == SHELVELOC or feature == URBAN or feature == US:
            continue
        for row in data:
            testSplit = check_split(feature, row[feature], data)
            rss = eval_RSS(testSplit, row[0], feature, row[feature])
#            print('X%d < %.3f RSS=%.3f'% ((feature+1), row[feature], rss))
            if rss < splitRSS:
                splitFeature = feature
                splitValue = row[feature]
                splitRSS = rss
                splitData = testSplit

#    print splitFeature, splitValue, splitRSS		
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

def eval_RSS(split, actualScore, feature, splitVal):
    # RSS for the split
    rss = 0.0
    # summ RSS for left side and right side
    for data in split:
        if len(data) > 0:
            prices = [row[0] for row in data]
            mean = np.mean(prices)
            for price in prices:
                rss += (price-mean)**2

#    print 'RSS with: ', rss, actualScore, feature, splitVal
    return rss        # average of left and right side error

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
    preds = np.array(predictions)
    targets = np.array(actual)
    targetMean = np.mean(actual)
    SStotal = 0.0
    for target in targets:
        SStotal += (target-targetMean)**2

    SSres = 0.0
    for cnt in range(len(actual)):
        SSres += (targets[cnt]-preds[cnt])**2
    return SSres/len(actual)    #RMSE

def eval_tree(data):
    folds = k_folds(data)
    foldsAccuracy = list()
	
    for fold in folds:
        trainSet = list(folds)
        trainSet.remove(fold)
        trainSet = sum(trainSet, [])
        testSet = list()
		
        for data in fold:
            tmpEle = list(data)
            testSet.append(tmpEle)

        # bagging
        predictions = bagging(trainSet,testSet)
 #       for tree in range(NUM_TREES):
 #           baggedSet, valSet = bag(trainSet)
 #           predictions = class_tree(baggedSet, valSet)
        actual = [data[0] for data in fold]
        accuracy = evaluate(predictions, actual)
 #       accuracy /= NUM_TREES

        foldsAccuracy.append(accuracy)
	
    return foldsAccuracy

# train NUM_TREES of trees using bagging
def bagging(trainSet, testSet):
    trees = list()
    for cnt in range (NUM_TREES):
        trainSubset = bag(trainSet)
        tree = decision_tree(trainSubset)
        trees.append(tree)
    predictions = [bagged_predict(trees, data) for data in testSet]
    return predictions

# predict values using ensemble of bagged trees
def bagged_predict(trees, data):
    predictions = [predict(data, tree) for tree in trees]
    return np.median(predictions)

def bag(trainSet):
    bagSet = list()
    num_in_bag = round(len(trainSet)*BAG_RATIO)
    # build subset of training data with replacment
    while len(bagSet) < num_in_bag:
        selection = random.randrange(len(trainSet))
        bagSet.append(trainSet[selection])

    return bagSet


# MAIN PROGRAM

random.seed(1)

# load data
data = load_csv(DATACSV)    
  

# create and evaluate decision tree
evaluation = eval_tree(data)

# print results
print ('Fold Accuracies: ', evaluation)
print ('Mean accuracy: %.3f' % (sum(evaluation)/(len(evaluation))))