from decisionmodule import *

####This is an example of how to use decisionmodule
#For practicality, we're not taking user priority matrix (i.e. asking the user for it)
#So we'll use the example arr as below
#This example is exact from Caterino et al. 2008 (https://www.tandfonline.com/doi/full/10.1080/13632460701572872)

criteria = ['Installation cost', 'Maintenance cost', 'Duration of works', 'Functional compatibility', 'Skilled labour requirement', 'Impact on the foundations', 'Significant damage risk', 'Damage limitation risk']

alternatives = ['GFRP', 'Steel Bracing', 'RC Jacketing', 'Base Isolation']

prioritymatrix = np.array([
    [1, 1/3, 1, 1/5, 4, 1/3, 4, 1/3], 
    [3, 1, 3, 1/2, 6, 1, 6, 1], 
    [1, 1/3, 1, 1/5, 4, 1/3, 4, 1/3], 
    [5, 2, 5, 1, 6, 2, 5, 2], 
    [1/4, 1/6, 1/4, 1/6, 1, 1/6, 1/2, 1/5], 
    [3, 1, 3, 1/2, 6, 1, 5, 3], 
    [1/4, 1/6, 1/4, 1/5, 2, 1/5, 1, 1/3], 
    [3, 1, 3, 1/2, 5, 1/3, 3, 1]])

decisionmatrix = np.array([
    [23096, 23206, 33, 0.538, 0.414, 2.90, 0.022, 0.291], 
    [53979, 115037, 122, 0.074, 0.120, 15.18, 0.024, 0.002], 
    [11175, 40353, 34, 0.274, 0.052, 2.97, 0.040, 0.172], 
    [74675, 97884, 119, 0.114, 0.414, 2.65, 0.020, 0.000]])

typecriteria = [0, 0, 0, 1, 0, 0, 0, 0]

weights, consistent, CR, prioritymatrix, CI, adoptedRCI = AHP(criteria, prioritymatrix, printstuff=True)

df = Topsis(decisionmatrix, weights,
            TypeCriteria=typecriteria, AlternativeNames=alternatives,
            CriteriaNames=criteria, printstuff=True, plotspider=True, plotstuff=False)


