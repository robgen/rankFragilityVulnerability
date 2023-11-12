# -*- coding: utf-8 -*-
# Doesn't support user input unless user inputs exact priority matrix
import numpy as np
import pandas as pd

def UserPriorityMatrix(criteria: list):
    pmatrix = np.zeros(shape=(len(criteria), len(criteria)))
    for i in range(len(criteria)):
        for j in range(len(criteria)):
            if i == j:
                pmatrix[i][j] = 1
            elif j > i:
                print('menu: ')
                print(f'(9) {criteria[i]} is extremely more important than {criteria[j]}')
                print('(8)')
                print(f'(7) {criteria[i]} is demonstrably more important than {criteria[j]}')
                print('(6)')
                print(f'(5) {criteria[i]} is essentially more important than {criteria[j]}')
                print('(4)')
                print(f'(3) {criteria[i]} is moderately more important than {criteria[j]}')
                print('(2)')
                print(f'(1) {criteria[i]} is equally as important as {criteria[j]}')
                print('(1/2)')
                print(f'(1/3) {criteria[j]} is moderately more important than {criteria[i]}')
                print('(1/4)')
                print(f'(1/5) {criteria[j]} is essentially more important than {criteria[i]}')
                print('(1/6)')
                print(f'(1/7) {criteria[j]} is demonstrably more important than {criteria[i]}')
                print('(1/8)')
                print(f'(1/9) {criteria[j]} is extremely more important than {criteria[i]}')
                pmatrix[i][j] = float(input('Type the choice that applies (type what\'s in the brackets): '))
            elif j < i:
                pmatrix[i][j] = 1 / pmatrix[j][i]
    return pmatrix




def AHP(criteria: list, prioritymatrix: np.array=None, crthreshold: float=0.1, printstuff: bool=False):
    defaultmatrix = np.zeros(shape=(len(criteria), len(criteria)))
    for i in range(1, len(criteria)+1):
        for j in range(1, len(criteria)+1):
            if i == j:
                defaultmatrix[i-1][j-1] = 1
            elif j > i:
                defaultmatrix[i-1][j-1] = 2*(i-1) + j
            elif j < i:
                defaultmatrix[i-1][j-1] = 1 / defaultmatrix[j-1][i-1]
    try:
        _ = prioritymatrix.shape
        if np.shape(prioritymatrix) != (len(criteria), len(criteria)):
            print('amount of criteria doesn\'t match dimensions of priority matrix - using default matrix instead')
            prioritymatrix = defaultmatrix
    except:
        print('priority matrix not given - using default matrix')
        prioritymatrix = defaultmatrix
        
    
    rank = np.linalg.matrix_rank(prioritymatrix)
    eigvals, eigvecs = np.linalg.eig(prioritymatrix)
    
    weightscalc = eigvecs / sum(eigvecs)
    weights = []
    for x in range(len(weightscalc)):
        weights.append(np.real(weightscalc[x][0]))
   
    #consistency index
    CI = (eigvals[0] - rank) / (rank - 1)
    #random consistency index depending on n criteria
    RCI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.53, 1.56, 1.57, 1.59]

    #consistency ratio (ncriteria = rank)
    CR = CI / RCI[rank]
    if CR > crthreshold:
        consistent = False
    else:
        consistent = True
    adoptedRCI = RCI[rank]
    
    
    if printstuff:
        print(f'Weights are: {weights}')
        print(f'This comparison is consistent: {consistent} with Consistency Ratio: {CR}')
    return weights, consistent, CR, prioritymatrix, CI, adoptedRCI

def Topsis(DecisionMatrix: np.ndarray, weights: list, TypeCriteria:list=None, AlternativeNames:list=None, CriteriaNames:list=None, printstuff: bool=False, plotstuff: bool=False, plotspider: bool=False):
    assert np.ndim(DecisionMatrix) == 2, 'Decision Matrix is not the correct dimension - should be (number of alternatives, number of criteria) i.e. of dimension 2'    
    assert np.ndim(weights) == 1, 'Weights list is not the correct dimension - should be (number of criteria) i.e. of dimension 1'
    assert np.shape(DecisionMatrix)[1] == len(weights), 'Decision Matrix and Weights list have different dimensions'
    if TypeCriteria is None:
        TypeCriteria = np.zeros((np.shape(DecisionMatrix)[1]))
    if AlternativeNames is None:
        AlternativeNames = []
        for i in range(np.shape(DecisionMatrix)[0]):
            AlternativeNames.append(f'Alt {i+1}')
    elif np.ndim(AlternativeNames) != 1 or len(AlternativeNames) != np.shape(DecisionMatrix)[0]:
        print('error using custom alternative names - check if dimensions are correct')
        print('using default names - Alt 1, Alt2, etc.')
        AlternativeNames = []
        for i in range(np.shape(DecisionMatrix)[0]):
            AlternativeNames.append(f'Alt {i+1}')
    if CriteriaNames is None:
        CriteriaNames = []
        for i in range(np.shape(DecisionMatrix)[1]):
            CriteriaNames.append(f'Crit {i+1}')
    elif np.ndim(CriteriaNames) != 1 or len(CriteriaNames) != np.shape(DecisionMatrix)[1]:
        print('error using custom criteria  names - check if dimensions are correct')
        print('using default names - Crit 1, Crit 2, etc.')
        CriteriaNames = []
        for i in range(np.shape(DecisionMatrix)[1]):
            CriteriaNames.append(f'Crit {i+1}')
    #normalise decision matrix
    normaliser = np.meshgrid(np.sqrt(np.sum(np.square(DecisionMatrix), axis=0)), np.ones(((np.shape(DecisionMatrix)[0]), 1)))[0]
    normDecisionMat = DecisionMatrix / normaliser
    #weight matrix
    weighter = np.meshgrid(weights, np.ones(((np.shape(DecisionMatrix)[0]), 1)))[0]
    weightednormDecisionMat = normDecisionMat * weighter
    #calculate ideal solutions
    #if criteria is a cost, best is minimum, otherwise maximum
    maxima = weightednormDecisionMat.max(axis=0)
    minima = weightednormDecisionMat.min(axis=0)
    #positive ideal solution (BEST)
    idealpos = TypeCriteria.copy()
    for i in range(len(idealpos)):
        if idealpos[i] == 0:
            idealpos[i] = minima[i]
        elif idealpos[i] == 1:
            idealpos[i] = maxima[i]
    # negative ideal solution (WORST)
    idealneg = TypeCriteria.copy()
    for i in range(len(idealneg)):
        if idealneg[i] == 0:
            idealneg[i] = maxima[i]
        elif idealneg[i] == 1:
            idealneg[i] = minima[i]
    #get distances
    distidealpos = (np.sum(np.square(weightednormDecisionMat - np.meshgrid(idealpos, np.ones(((np.shape(DecisionMatrix)[0]), 1)))[0]), axis=1)) ** 0.5
    distidealneg = (np.sum(np.square(weightednormDecisionMat - np.meshgrid(idealneg, np.ones(((np.shape(DecisionMatrix)[0]), 1)))[0]), axis=1)) ** 0.5
    #relative closeness ratio
    closeness = (distidealneg) / (distidealpos + distidealneg)
    # closeness = list of closeness ratios, index=alternatives
    #df = pd.DataFrame(data=[[AlternativeNames[i], closeness[i], [round(weightednormDecisionMat[i][j], 4) for j in range(np.shape(weightednormDecisionMat)[1])]] for i in range(np.shape(DecisionMatrix)[0])]
    #                  , columns=['Name', 'Closeness', 'Weighted Normalised Decision Matrix Values'])
    df1 = pd.DataFrame(data=[[AlternativeNames[i], closeness[i]] for i in range(np.shape(DecisionMatrix)[0])]
                      , columns=['Name', 'Closeness'])
    df2 = pd.DataFrame(np.round(weightednormDecisionMat, 4), columns=CriteriaNames)
    df = df1.join(df2)
    df = df.set_index('Name')
    df = df.sort_values('Closeness', ascending=False)
    if printstuff:
        print(df)
    if plotstuff or plotspider:
        import matplotlib.pyplot as plt
        if plotspider:
            plt.figure(figsize=(6, 6))
            label_loc = np.linspace(0, 2*np.pi, num=len(CriteriaNames)+1)
            for i in range(len(AlternativeNames)):
                plt.polar(label_loc, [*weightednormDecisionMat[i], weightednormDecisionMat[i][0]], label=AlternativeNames[i])
            lines, labels = plt.thetagrids((np.degrees(label_loc[:-1])), labels=CriteriaNames)
            plt.title('Radar Map - Weighted and Normalised Criteria Values')
            plt.legend(bbox_to_anchor=(1.3, 0.5))
            plt.show()
        if plotstuff:
            fig, axs = plt.subplots(nrows=1, ncols=len(CriteriaNames)+1)
            axs[0].bar(df.index, df['Closeness'])
            axs[0].set_title('Closeness', fontdict={'fontsize': 6})
            axs[0].tick_params(axis='both', which='major', labelsize=8, direction='out', labelrotation=90, pad=0.01)
            for i in range(len(CriteriaNames)):
                axs[i+1].bar(df.index, df[CriteriaNames[i]]) 
                axs[i+1].set_title(f'{CriteriaNames[i]}', fontdict={'fontsize': 6})
                axs[i+1].tick_params(axis='both', which='major', labelsize=8, direction='out', labelrotation=90, pad=0.01)
                # axs[i+1].set_ylabel('Values weighted and normalised')
            plt.show()
    
    return(df)
    

if __name__ == '__main__':
    arr = np.array([[1, 2, 3], [5, 6, 7], [3, 4, 5], [8, 9, 10]])
    UserPriorityMatrix(['bruh', 'bruh2', 'bruh3'])
    Topsis(arr, [0.1, 0.2, 0.3], printstuff=True, plotstuff=True, plotspider=True)
    
