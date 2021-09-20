from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['Education_Level'] = self.dataset['Education_Level'].\
            replace('Unknown', self.dataset['Education_Level'].mode()[0])

        self.dataset['Income_Category'] = self.dataset['Income_Category']. \
            replace('Unknown', self.dataset['Income_Category'].mode()[0])

        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['Uneducated'], 0)
        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['High School'], 1)
        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['College'], 2)
        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['Graduate'], 3)
        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['Post-Graduate'], 4)
        self.dataset['Education_Level'] = self.dataset['Education_Level'].replace(['Doctorate'], 5)

        self.dataset['Income_Category'] = self.dataset['Income_Category'].replace(['Less than $40K'], 0)
        self.dataset['Income_Category'] = self.dataset['Income_Category'].replace(['$40K - $60K'], 1)
        self.dataset['Income_Category'] = self.dataset['Income_Category'].replace(['$60K - $80K'], 2)
        self.dataset['Income_Category'] = self.dataset['Income_Category'].replace(['$80K - $120K'], 3)
        self.dataset['Income_Category'] = self.dataset['Income_Category'].replace(['$120K +'], 4)

        self.dataset['Card_Blue'] = 0
        self.dataset['Card_Silver'] = 0
        self.dataset['Card_Gold'] = 0
        self.dataset['Card_Platinum'] = 0
        self.dataset.loc[self.dataset['Card_Category'] == 'Blue', 'Card_Blue'] = 1
        self.dataset.loc[self.dataset['Card_Category'] == 'Silver', 'Card_Silver'] = 1
        self.dataset.loc[self.dataset['Card_Category'] == 'Gold', 'Card_Gold'] = 1
        self.dataset.loc[self.dataset['Card_Category'] == 'Platinum', 'Card_Platinum'] = 1
        self.dataset['Status_Single'] = 0
        self.dataset['Status_Married'] = 0
        self.dataset['Status_Divorced'] = 0
        self.dataset.loc[self.dataset['Marital_Status'] == 'Single', 'Status_Single'] = 1
        self.dataset.loc[self.dataset['Marital_Status'] == 'Married', 'Status_Married'] = 1
        self.dataset.loc[self.dataset['Marital_Status'] == 'Divorced', 'Status_Divorced'] = 1
        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])
        drop_columns = ['CLIENTNUM',
                        'Marital_Status',
                        'Card_Category',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
        self.dataset = self.dataset.drop(drop_columns, axis=1)
        return self.dataset

