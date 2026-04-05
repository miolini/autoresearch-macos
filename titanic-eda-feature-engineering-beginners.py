import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import time

start = time.time()


train = pd.read_csv('titanic-data/train.csv')
test = pd.read_csv('titanic-data/test.csv')
joined = pd.concat([train, test], axis=0)




# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes = axes.flatten()

# joined.groupby(['Embarked', 'Survived']).size().unstack().plot(kind='bar', ax=axes[0])
# joined.groupby(['Pclass', 'Survived']).size().unstack().plot(kind='bar', ax=axes[1])
# joined.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', ax=axes[2])


# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# survived_data = joined[joined['Survived'] == 1]['Age']
# not_survived_data = joined[joined['Survived'] == 0]['Age']

# sns.histplot(survived_data, bins=25, color='blue', label='Survived', kde=True, ax=axes[0])
# sns.histplot(not_survived_data, bins=25, color='red', label='Not Survived', kde=True, ax=axes[0])

# axes[0].set_xlabel('Age')
# axes[0].set_ylabel('Frequency')
# axes[0].set_title('Histogram of Age with Survival')
# axes[0].legend()

# survived_data = joined[joined['Survived'] == 1]['Fare']
# not_survived_data = joined[joined['Survived'] == 0]['Fare']

# sns.histplot(survived_data, bins=25, color='blue', label='Survived', kde=True, ax=axes[1])
# sns.histplot(not_survived_data, bins=25, color='red', label='Not Survived', kde=True, ax=axes[1])

# axes[1].set_xlabel('Fare')
# axes[1].set_ylabel('Frequency')
# axes[1].set_title('Histogram of Fare with Survival')
# axes[1].legend()



# sns.catplot(data=joined, x='SibSp', y='Survived', kind='bar')


# sns.catplot(data=joined, x='Parch', y='Survived', kind='bar')


# sns.catplot(data=joined, x='Sex', y='Age', kind='box')
# sns.catplot(data=joined, x='SibSp', y='Age', kind='box')
# sns.catplot(data=joined, x='Parch', y='Age', kind='box')


# # 3. Feature Engineering

# ### 3.1 Missing Age: Impute based on similar Sex and Class
joined['Age'] = joined.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# ### 3.2 Embarked
joined[joined['Embarked'].isnull()]
joined['Embarked'].fillna("S", inplace=True) # From Google I have found that they both deparated from Southampton (S)


# ### 3.3 Fare: Filling the missing value in Fare with the median Fare of 3rd class alone passenger
joined[joined['Fare'].isnull()]
med_fare = joined.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
joined['Fare'] = joined['Fare'].fillna(med_fare)


# ### 3.4 Extract Deck from Cabin
# Set missing Cabin values as "M" for missing
joined['Deck'] = joined['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else "M")

# Passenger in the T deck is changed to A
joined.loc[joined[joined['Deck'] == 'T'].index, 'Deck'] = 'A'  


# Group decks with similar class/survival rates together
joined['Deck'] = joined['Deck'].replace(['A', 'B', 'C'], 'ABC')
joined['Deck'] = joined['Deck'].replace(['D', 'E'], 'DE')
joined['Deck'] = joined['Deck'].replace(['F', 'G'], 'FG')

joined.drop('Cabin', inplace=True, axis=1)


# ### 3.5 Fare & Age: Binning into groups

joined['Fare'] = pd.qcut(joined['Fare'], 13)

joined['Age'] = pd.qcut(joined['Age'], 10)


# ### 3.6 Family Size

joined['Family_Size'] = joined['SibSp'] + joined['Parch'] + 1



# Create new feature of family size
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
joined['Family_Size_Grouped'] = joined['Family_Size'].map(family_map)


# ### 3.7 Ticket Frequency

joined['Ticket_Frequency'] = joined.groupby('Ticket')['Ticket'].transform('count')


# ### 3.8 Extract Title from Name


joined['Title'] = joined['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
# g = sns.countplot(data=joined, x='Title')
# g = plt.setp(g.get_xticklabels(), rotation=45) 


# ### 3.9 Married


joined['Is_Married'] = 0
# joined['Is_Married'].loc[joined['Title'] == 'Mrs'] = 1
joined.loc[joined['Title'] == 'Mrs', 'Is_Married'] = 1

# ### 3.10 Titles: Binning titles together


joined['Title'] = joined['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
joined['Title'] = joined['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# ### 3.11 Extract Surname



def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

joined['Family'] = extract_surname(joined['Name'])


# ### 3.12 New Features: Family Survival Rate, Ticket Survival Rate


train = joined[joined['Survived'].notna()]
test = joined[joined['Survived'].isna()]


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]
non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]

df_family_survival_rate = train.groupby('Family')[['Survived', 'Family_Size']].median()
df_ticket_survival_rate = train.groupby('Ticket')[['Survived', 'Ticket_Frequency']].median()



family_rates = {}
ticket_rates = {}

# Checking a family exists in both training and test set, and has members more than 1
for i in range(len(df_family_survival_rate)):
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

# Checking a ticket exists in both training and test set, and has members more than 1
for i in range(len(df_ticket_survival_rate)):
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]


mean_survival_rate = np.mean(train['Survived'])


# #### Family Survival Rate

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(train)):
    if train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
train['Family_Survival_Rate'] = train_family_survival_rate
train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
test['Family_Survival_Rate'] = test_family_survival_rate
test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA


# #### Ticket Survival Rate

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(train)):
    if train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
train['Ticket_Survival_Rate'] = train_ticket_survival_rate
train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
test['Ticket_Survival_Rate'] = test_ticket_survival_rate
test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA


# #### Overall Survival Rate

for df in [train, test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2   


# # 4. Preprocessing

# ### 4.1 Label Encoding

non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in [train, test]:
    for feature in non_numeric_features:        
        df[feature] = LabelEncoder().fit_transform(df[feature])


# ### 4.2 One Hot Encoding

cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
encoded_features = []

for df in [train, test]:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

train = pd.concat([train, *encoded_features[:6]], axis=1)
test = pd.concat([test, *encoded_features[6:]], axis=1)


# ### 4.3 Drop Irrelevant Features

joined = pd.concat([train, test], sort=True).reset_index(drop=True)
drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped',
             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']

joined.drop(columns=drop_cols, inplace=True)


# ### 4.4 Split back to Train and Test

train = joined[joined['Survived'].notna()]
test = joined[joined['Survived'].isna()]

test.drop('Survived', axis=1, inplace=True)

ytrain = train['Survived']
xtrain = train.drop('Survived', axis=1)


# ### 4.5 Standardising

# In[43]:


xtrain = StandardScaler().fit_transform(xtrain)
xtest = StandardScaler().fit_transform(test)

print('X_train shape: {}'.format(xtrain.shape))
print('y_train shape: {}'.format(ytrain.shape))
print('X_test shape: {}'.format(test.shape))


# # 5. Modelling

SEED = 42
N_ESTIMATORS = 1750
MAX_DEPTH = 7
MIN_SAMPLES_SPLIT = 6
MIN_SAMPLES_LEAF = 6
MAX_FEATURES = 'sqrt'

rf = RandomForestClassifier(criterion='gini',
                           n_estimators=N_ESTIMATORS,
                           max_depth=MAX_DEPTH,
                           min_samples_split=MIN_SAMPLES_SPLIT,
                           min_samples_leaf=MIN_SAMPLES_LEAF,
                           max_features=MAX_FEATURES,
                           oob_score=True,
                           random_state=SEED,
                           n_jobs=-1,
                           verbose=1) 


N = 5
oob = 0
probs = pd.DataFrame(np.zeros((len(xtest), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])

# importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=df_all.columns)
# fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(skf.split(xtrain, ytrain), 1):
    print('Fold {}\n'.format(fold))
    
    rf.fit(xtrain[trn_idx], ytrain[trn_idx])
    
#     # Computing Train AUC score
#     trn_fpr, trn_tpr, trn_thresholds = roc_curve(ytrain[trn_idx], leaderboard_model.predict_proba(xtrain[trn_idx])[:, 1])
#     trn_auc_score = auc(trn_fpr, trn_tpr)
    
#     # Computing Validation AUC score
#     val_fpr, val_tpr, val_thresholds = roc_curve(ytrain[val_idx], leaderboard_model.predict_proba(xtrain[val_idx])[:, 1])
#     val_auc_score = auc(val_fpr, val_tpr)  
      
#     scores.append((trn_auc_score, val_auc_score))
#     fprs.append(val_fpr)
#     tprs.append(val_tpr)
    
    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = rf.predict_proba(xtest)[:, 0]
    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = rf.predict_proba(xtest)[:, 1]

#      importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_
        
    oob += rf.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, rf.oob_score_))   
    
print('Average OOB Score: {}'.format(oob))




# # 6. Submission


class_survived = [col for col in probs.columns if col.endswith('Prob_1')]
probs['1'] = probs[class_survived].sum(axis=1) / N
probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / N

probs['pred'] = 0
pos = probs[probs['1'] >= 0.5].index
probs.loc[pos, 'pred'] = 1

y_pred = probs['pred'].astype(int)




# submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# submission_df['Survived'] = y_pred.values

# submission_df.to_csv('submission.csv', header=True, index=False)
# submission_df.head(10)

end = time.time()


print("---")
print(f"oob:          {oob:.6f}")
print(f"time: {end-start:.1f}")
print(f"N_folds:      {N}")
print(f"seed:      {SEED}")
print(f"n_estimaors:   {N_ESTIMATORS}")
print(f"max_depth:        {MAX_DEPTH}")
print(f"min_samples_split:     {MIN_SAMPLES_SPLIT}")
print(f"min_samples_leaf:            {MIN_SAMPLES_LEAF}")
print(f"max_features:            {MAX_FEATURES}")
