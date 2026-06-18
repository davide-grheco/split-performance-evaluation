using MLJ
using DataFrames
using Distances

export train_classification_random_forest, predict_classification, evaluate_classification, ClassificationResult

struct ClassificationResult
    test_metrics::DataFrame
    ipss_metrics::DataFrame
    test_confusion::Matrix
    ips_confusion::Matrix
end

function train_classification_random_forest(X_train, y_train; n_trees=100)
    TreeClass = MLJ.@load RandomForestClassifier pkg = DecisionTree
    model = TreeClass(n_trees=n_trees)
    mach = machine(model, X_train, y_train)
    fit!(mach)
    return mach
end

function predict_classification(mach, X_test)
    ŷ_class = predict(mach, X_test)
    ŷ_labels = mode.(ŷ_class)
    return ŷ_class, ŷ_labels
end



function evaluate_classification(y_pred, y_test, y_ips, y_pred_ips)
    acc_test = accuracy(y_pred, y_test)
    matrix_of_confusion_test = confusion_matrix(y_test, y_pred)
    mcc_test = mcc(y_pred, y_test)

    acc_ips = accuracy(y_pred_ips, y_ips)
    matrix_of_confusion_ips = confusion_matrix(y_ips, y_pred_ips)
    mcc_ips = mcc(y_pred_ips, y_ips)

    evaluation_results_test = DataFrame(
        Metric=["Test Accuracy", "Test MCC"],
        Value=[round(acc_test, sigdigits=4), round(mcc_test, sigdigits=4)]
    )

    evaluation_results_ips = DataFrame(
        Metric=["IPS Accuracy", "IPS MCC"],
        Value=[round(acc_ips, sigdigits=4), round(mcc_ips, sigdigits=4)]
    )


    return ClassificationResult(evaluation_results_test, evaluation_results_ips, matrix_of_confusion_test, matrix_of_confusion_ips)
end
