import sys
import time
import random
import csv
import numpy as np
from copy import deepcopy

MAX_HEIGHT = 2
MAX_DATA = 5
FOLDS = 4
TUNING = False
BAGGING = True
RANDOM_FOREST = True
NUM_TREES = 10
BAG_RATIO = .67
NORMALIZE = True
DATACSV = 'Carseats.csv'
# feature number for CATEGORICAL data
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
#            if NORMALIZE == True:
#                featureSet = [float(example[column]) for example in data] 
#                featureMin = np.min(featureSet)
#                featureMax = np.max(featureSet)
                #featureMean = np.mean(featureSet)
                #featureStd = np.std(featureSet)
            for row in data:
 #           if NORMALIZE == True:
                    # min/max normalization
 #                   row[column] = (float(row[column])-featureMin)/(featureMax-featureMin)
                    # Standardization    
                    #row[column] = (float(row[column])-featureMean)/featureStd
 #               else:    
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

# makes a leaf at the bottom of the decision tree, i.e. a prediction
def make_leaf(data):
	# classification of data
    scores = [row[0] for row in data]
    return np.median(scores)

# splits data in to left and right by feature/value
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

# checks all possible splits for training example and selects one with lowest RSS
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

# calculate the RSS of a split
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

	# split the rest of the data into k folds for training
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
        trainSet = deepcopy(folds)
        trainSet.remove(fold)
        trainSet = sum(trainSet, [])
        testSet = [example for example in deepcopy(fold)] #list()
        
        # normalize numerical data using training set min and max values  
        for feature in range(1, len(fold[0])):
            if feature != SHELVELOC and feature != URBAN and feature != US:
                featureSet = [example[feature] for example in trainSet] 
                featureMin = np.min(featureSet)
                featureMax = np.max(featureSet)
                for example in trainSet:
                    example[feature] = (example[feature]-featureMin)/(featureMax-featureMin)
                for example in testSet:
                    example[feature] = (example[feature]-featureMin)/(featureMax-featureMin)      
		
#        for example in fold:
#            tmpEle = list(example)
#            testSet.append(tmpEle)

        # bagging
        if BAGGING == True:
            predictions = ensemble_trees(trainSet,testSet, foldCnt)
        else:
            predictions = single_tree(trainSet, testSet, foldCnt)

        # determine accuracy of tree(s)
        actual = [example[0] for example in testSet]
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

# create and evaluate decision trees with hyperparameter search
def tune_decisionTree(data):
    global MAX_HEIGHT, MAX_DATA, BAGGING, RANDOM_FOREST
    BAGGING = RANDOM_FOREST = False
    
    #hyperparameter search space
    maxheight   = [2,5,10,20]
    maxdata     = [2,5,10,20]
    
    for height in maxheight:
        for numdata in maxdata:
            MAX_HEIGHT = height
            MAX_DATA   = numdata
            print 'max tree height: ', height
            print 'max data: ', numdata
            # test decision tree
            evaluation = eval_tree(data)
            # print results
            print '\nFold RMSE: ', evaluation
            print 'Mean RMSE: %.3f \n' % (sum(evaluation)/(len(evaluation)))

# create and evaluate bagging, decision trees with hyperparameter search
def tune_bagging(data):
    global NUM_TREES, BAGGING, RANDOM_FOREST
    BAGGING = True
    RANDOM_FOREST = False
    
    #hyperparameter search space
    maxtrees    = [2, 4, 6, 8, 10, 12, 14, 16]

    for numtrees in maxtrees:
        print 'tree max : ', numtrees
        NUM_TREES = numtrees
        evaluation = eval_tree(data)

        # print results
        print '\nFold RMSE: ', evaluation
        print 'Mean RMSE: %.3f \n' % (sum(evaluation)/(len(evaluation)))

# create and evaluate random forest, decision trees with hyperparameter search
def tune_randomForest(data):
    global NUM_TREES, BAGGING, RANDOM_FOREST
    BAGGING = RANDOM_FOREST = True

    #hyperparameter search space
    maxtrees    = [20, 40, 60, 80, 100]
    maxheight   = [1,2,3,5,8]

    for height in maxheight:
        for numtrees in maxtrees:
            print 'tree max : ', numtrees
            NUM_TREES = numtrees
            evaluation = eval_tree(data)

            # print results
            print '\nFold RMSE: ', evaluation
            print 'Mean RMSE: %.3f \n' % (sum(evaluation)/(len(evaluation)))

def default_fit(data):
    evaluation = eval_tree(data)

    # print results
    print '\nFold RMSE: ', evaluation
    print 'Mean RMSE: %.3f \n' % (sum(evaluation)/(len(evaluation)))


# MAIN PROGRAM

random.seed(time.time())

# load data
data = load_csv(DATACSV)    

# Training and Testing
if TUNING:
    # tune decision trees
    tune_decisionTree(data)
    tune_bagging(data)
    tune_randomForest(data)
else:
    # run with default constants
    default_fit(data)