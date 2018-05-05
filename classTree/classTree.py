def eval_gini(classes, vals):
    # number of samples
    samples = float(sum([len(aClass) for aClass in classes]))

    #sum for each class
    score = 0.0
    for aClass in classes:
        size = float(len(aClass))
        if size > 0:
            count = 0.0
            # get proportions for each class
            for value in vals:
                proportion = [row[-1] for row in aClass].count(value)/size
                count += proportion * proportion
            # calc score for gini evaluation
            score += (1.0-count)*(size/samples)
    
    return score


print(eval_gini([[[1,1],[1,0]],[[1,1], [1,0]]], [0,1]))
print(eval_gini([[[1,0],[1,0]],[[1,1], [1,1]]], [0,1]))
