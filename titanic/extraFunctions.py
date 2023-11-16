
import pandas
import sklearn as sk

def saveSubmission(model, X_test, test, string):
    '''
    Saves the submission file.

    Parameters
    ----------
    model  : current trained model
    X_test : data for prediction
    test   : dataframe containing the passengerId 
    string : name of the saved file
        
    Returns
    -------
    None
        '''
    predictions = model.predict(X_test).astype(int)
    output = pandas.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('submissions/{}.csv'.format(string), index=False)
    print("Your submission was successfully saved!")

def confusionMatrix(model,X_cv,y_cv):
    '''
    Outputs a confusion matrix

    Parameters
    ----------
    model  : current trained model
    X_cv   : data for prediction
    y_cv   : target
        
    Returns
    -------
    None
    '''
    y_cv_hat = model.predict(X_cv)
    CM = sk.metrics.confusion_matrix( y_cv, y_cv_hat )
    sk.metrics.ConfusionMatrixDisplay(CM).plot()
    print("The fraction of correct predictions is {0:.2}".format((CM[0][0]+CM[1][1])/(CM[0][0]+CM[1][1]+CM[0][1]+CM[1][0])))
    print("The precision is {0:.2}".format((CM[1][1])/(CM[1][1]+CM[0][1])))
    print("The recall is {0:.2}".format((CM[1][1])/(CM[1][1]+CM[1][0])))
    print("The specificity is {0:.2}".format(CM[0][0]/(CM[0][0]+CM[1][0])))