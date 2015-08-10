import re
import numpy as np
import pandas as pd
import csv as csv 
#Import all the data

data_1314 = pd.read_csv('/Users/ChristopherSchon/Documents/PythonPractice/HTFT/PL1314.csv')
data_1213 = pd.read_csv('/Users/ChristopherSchon/Documents/PythonPractice/HTFT/PL1213.csv')
data_1112 = pd.read_csv('/Users/ChristopherSchon/Documents/PythonPractice/HTFT/PL1112.csv')


data = pd.concat([data_1112, data_1213, data_1314], axis=0, ignore_index=True)

# Add important stat Full time goal difference 
data['FTGD'] = data['FTHG'] - data['FTAG'] #Full time goal difference

#Make resultOdds DF

resultOdds = data[['B365A', 'B365D', 'B365H', 'BSA', 'BSD', 'BSH', 'BWA', 'BWD', 'BWH', 'BbAvA',
       'BbAvD', 'BbAvH', 'BbMxA', 'BbMxD', 'BbMxH', 'GBA', 'GBD', 'GBH', 'IWA', 'IWD', 'IWH', 'LBA', 'LBD',
       'LBH', 'PSA', 'PSD', 'PSH', 'SBA', 'SBD', 'SBH', 'SJA',
       'SJD', 'SJH', 'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH']]

#Makes average odds of home, draw, away result
def makeAvgOdds(regex, resultOdds):
	
	result = []
	for odds in resultOdds.columns.values:
		match = re.match(regex,odds)
		if match != None:
			result.append(odds)

	return resultOdds[ result ].mean( axis = 1) 


homeOdds = makeAvgOdds('([^/]*H)$', resultOdds)
awayOdds = makeAvgOdds('([^/]*A)$', resultOdds)
drawOdds = makeAvgOdds('([^/]*D)$', resultOdds)

data['homeOdds'] = homeOdds
data['awayOdds'] = awayOdds
data['drawOdds'] = drawOdds

#Finalise trimmed training data 
FTGD_train = data['FTGD'].values
train_data = data.drop(['HomeTeam','AwayTeam','Div','Referee','FTR','HTR','Date','FTGD','FTHG','FTAG'], axis = 1)
train_data = train_data.drop(resultOdds, axis = 1)
train_data = train_data.fillna(train_data.mean())

#Set up test data 
test_Data = pd.read_csv('/Users/ChristopherSchon/Documents/PythonPractice/HTFT/PL1415.csv')

testResultOdds = test_Data[['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','LBH','LBD','LBA','PSH','PSD','PSA','WHH','WHD','WHA','SJH','SJD','SJA','VCH','VCD','VCA','BbMxA','BbMxD','BbMxH','BbAvH','BbAvA','BbAvD']]
testHomeOdds = makeAvgOdds('([^/]*H)$', testResultOdds)
testAwayOdds = makeAvgOdds('([^/]*A)$', testResultOdds)
testDrawOdds = makeAvgOdds('([^/]*D)$', testResultOdds)

test_Data['homeOdds'] = homeOdds
test_Data['awayOdds'] = awayOdds
test_Data['drawOdds'] = drawOdds

test_Data['FTGD'] = test_Data['FTHG'] - test_Data['FTAG']
FTGD_test = test_Data['FTGD'].values
HTGD_test = test_Data['HTHG'] - test_Data['HTAG']

test_Data = test_Data.fillna(test_Data.mean())
test_Data = test_Data.drop(testResultOdds, axis = 1)

# Set up fixture columns for output into excel file
test_fixtures = test_Data['HomeTeam'] + " vs. " + test_Data['AwayTeam']
train_fixtures = data['HomeTeam'] + "vs. " + data['AwayTeam']

# Drop non needed columns 
test_Data = test_Data.drop(['FTGD','HomeTeam','AwayTeam','Div','Referee','FTR','HTR','Date','FTHG', 'FTAG'], axis = 1)
test_Data = test_Data[train_data.columns.values]

train_data = train_data.values
test_data = test_Data.values


print 'Training...'

forest = RandomForestClassifier(n_estimators=1000)
forest = forest.fit( train_data, FTGD_train )


print 'Predicting...'
output = forest.predict(test_Data).astype(float)
goal_error = output - FTGD_test

predictions_file = open("HTFTForest1.csv", "wb")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["Fixtures","Half time GD", "Expected FT Goal Difference", "Actual FT Goal Difference", "Goal error (Pred - Actual)"])

open_file_object.writerows(zip(test_fixtures, HTGD_test, output,FTGD_test, goal_error))

predictions_file.close()

print 'Done.'



