def class_wise_accuracy(y_true,y_pred,average=None):
    # y_true and y_pred should have same dimensions
    levels = set(y_true).intersection(set(y_pred))
    accuracy_by_class = []
    levels_order = []
    for each_level in levels:
        print("each_level:",each_level)
        relevant_cases = [each_true_label == each_level for each_true_label in y_true]
        correct_cases = [y_pred[i] == each_level for i,relevant_case in enumerate(relevant_cases) if relevant_case ]
        print("relevant_cases: ",relevant_cases)
        print("correct_cases: ",correct_cases)
        accuracy_by_class.append(sum(correct_cases)/sum(relevant_cases))
        levels_order.append(each_level)
    return accuracy_by_class

class_wise_accuracy(y_true,y_pred)
y_true
y_pred