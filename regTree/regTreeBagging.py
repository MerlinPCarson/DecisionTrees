import sys
import time
import random
import csv
import numpy as np

MAX_HEIGHT = 5
MAX_DATA = 10
FOLDS = 2
BAGGING = False
RANDOM_FOREST = False
NUM_TREES = 10
BAG_RATIO = .67
NORMALIZE = True
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
            if NORMALIZE == True:
                featureSet = [float(example[column]) for example in data] 
                featureMin = np.min(featureSet)
                featureMax = np.max(featureSet)
                #featureMean = np.mean(featureSet)
                #featureStd = np.std(featureSet)
            for row in data:
                if NORMALIZE == True:
                    # min/max normalization
                    row[column] = (float(row[column])-featureMin)/(featureMax-featureMin)
                    # Standardization    
                    #row[column] = (float(row[column])-featureMean)/featureStd
                else:    
                    row[column] = float(row[column])
	
    return data

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        if node['feature'] != SHELVELOC and node['feature'] != URBAN and node['feature'] != US:
            print('%s(X%d < %.3f)' % ((depth*' ', (node['feature']), node['value'])))
        else:
            print('%s(X%s = %s)' % ((depth*' ', (node['feature']), node['value']))) 
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# create and evaluate a classification decision tree
def single_tree(trainData, testData, foldCnt):
	# make the tree
    tree = decision_tree(trainData)
    predictions = list()
    print '\nDecision Tree (fold {}): \n'.format(foldCnt)
    print_tree(tree)
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
        # categorical feature
        if feature == SHELVELOC or feature == URBAN or feature == US:
            if row[feature] == value:
                leftSplit.append(row)
            else:
                rightSplit.append(row)
        # numerical feature
        else:
            if row[feature] < value:
                leftSplit.append(row)
            else:
                rightSplit.append(row)

    return leftSplit, rightSplit

def max_split(data):

    splitRSS = sys.maxint
    total_features = len(data[0])
    features = list()
    if RANDOM_FOREST == True:
        num_features = int(np.sqrt(total_features))
        # add num_features amount of features to feature set to maximize split
        while len(features) < num_features:
            selection = random.randrange(1, total_features)         # don't select first feature, sales
            # add features without replacement
            if selection not in features:
 #               if selection != SHELVELOC and selection != URBAN and selection != US:
                features.append(selection)
    # add all features to feature set to maximize split
    else:
        for feature in range(1, total_features):                    # don't select first feature, sales
 #           if feature != SHELVELOC and feature != URBAN and feature != US:
            features.append(feature)                             

    for feature in features:
#        if feature == SHELVELOC or feature == URBAN or feature == US:
#            continue
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
    for _ in range(FOLDS):
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
	
    foldCnt = 0
    for fold in folds:
        foldCnt += 1
        trainSet = list(folds)
        trainSet.remove(fold)
        trainSet = sum(trainSet, [])
        testSet = list()
		
        for data in fold:
            tmpEle = list(data)
            testSet.append(tmpEle)

        # bagging
        if BAGGING == True:
            predictions = ensemble_trees(trainSet,testSet, foldCnt)
        else:
            predictions = single_tree(trainSet, testSet, foldCnt)

        # determine accuracy of tree(s)
        actual = [data[0] for data in fold]
        accuracy = evaluate(predictions, actual)

        # save accuracy of fold
        foldsAccuracy.append(accuracy)
	
    return foldsAccuracy

# train NUM_TREES of trees using bagging
def ensemble_trees(trainSet, testSet, foldCnt):
    trees = list()
    print 'fold ', foldCnt
    for treeNum in range (NUM_TREES):
        print 'Building tree ', treeNum
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

random.seed(time.time())

# load data
data = load_csv(DATACSV)    

# create and evaluate decision tree
evaluation = eval_tree(data)

# print results
print '\nFold RMSE: ', evaluation
print 'Mean RMSE: %.3f \n' % (sum(evaluation)/(len(evaluation)))