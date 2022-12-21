import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import originalTextFor
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import datetime as dt


class Segmentation:
    def __init__(self):
        self.products, self.productList, self.numOfProducts = self.get_products()
        self.correlationMatrix = self.get_correlationMatrix()
        self.matchedProducts = []
        self.rfm_agg = []
        self.df_rfm = []

    # Retrieve the entire products dataset
    def get_products(self):
        df = pd.read_csv("data.csv", encoding="ISO-8859-1")
        df.dropna(subset=['Description'],how='all',inplace=True)
        products = df.groupby(['Description'], as_index=False)['InvoiceDate'].max()

        productList = products['Description'].to_list()
        numOfProducts = len(productList)

        return products, productList, numOfProducts

    # Retrieve the already calculated data used for calculation of correlation matrix.
    def get_correlationMatrix(self):
        correlationData = np.loadtxt("CosinSimilarity.txt")
        return correlationData



    
    def get_matched_keywords(self, title):

        indices = pd.Series(self.products.index,index=self.products['Description']).drop_duplicates()
        indx = indices[title]

        #getting pairwise similarity scores
        sig_scores = list(enumerate(self.correlationMatrix[indx]))
        
        #sorting products
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        
        #100 most similar products score
        sig_scores = sig_scores[1:100]
        
        #product indexes
        product_indices = [i[0] for i in sig_scores]
        
        #Top 100 most similar products
        self.matchedProducts = self.products['Description'].iloc[product_indices]
        return self.matchedProducts

    def get_segments(self):
        data = pd.read_csv("data.csv", encoding="ISO-8859-1")
        data['date'] = pd.DatetimeIndex(data.InvoiceDate).date
        data['date'] = data['date'].astype('datetime64[ns]')
        data = data[data['UnitPrice']>0]
        data.dropna(subset=['CustomerID'],how='all',inplace=True)
        data[data.Description.isin(list(self.matchedProducts.unique()))]
        
        max_date = data['date'].max().date()
        recency = data.groupby(['CustomerID'],as_index=False)['date'].max()
        recency.columns = ['CustomerID','LastPurchaseDate']
        recency['Recency'] = recency.LastPurchaseDate.apply(lambda x : (max_date - x.date()).days)
        recency.drop(columns=['LastPurchaseDate'],inplace=True)

        frequency = data.copy()
        frequency.drop_duplicates(subset=['CustomerID','InvoiceNo'], keep="first", inplace=True) 
        frequency = frequency.groupby('CustomerID',as_index=False)['InvoiceNo'].count()
        frequency.columns = ['CustomerID','Frequency']

        data['Total_cost'] = data['UnitPrice'] * data['Quantity']
        monetary = data.groupby('CustomerID',as_index=False)['Total_cost'].sum()
        monetary.columns = ['CustomerID','Monetary']
        
        rf = recency.merge(frequency, left_on='CustomerID', right_on='CustomerID')
        rfm = rf.merge(monetary, left_on='CustomerID', right_on='CustomerID')
        rfm.set_index('CustomerID',inplace = True)
        rfm.sort_values(['Monetary'], ascending = False, inplace = True)
        
        # Creating labels for Recency and Frequency
        r_labels = range(4, 0, -1) 
        f_labels = range(1, 5)
        m_labels = range(1, 5)

        # Assign these labels to 4 equal percentile groups 
        r_groups = pd.qcut(rfm['Recency'].rank(method='first'), q=4, labels=r_labels)

        # Assign these labels to 4 equal percentile groups 
        f_groups = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)

        # Assign these labels to three equal percentile groups 
        m_groups = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=m_labels)

        # Create new columns R and F 
        df_rfm = rfm.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)

        def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
        df_rfm['RFM_Segment'] = df_rfm.apply(join_rfm, axis=1)
        rfm_count_unique = df_rfm.groupby('RFM_Segment')['RFM_Segment'].nunique()
        df_rfm['RFM_Score'] = df_rfm[['R','F','M']].sum(axis=1)

        def rfm_level(df):
            if df['RFM_Segment'] == '444':
                return 'STARS'
            
            elif df['RFM_Segment'] == '411':
                return 'NEW'
            
            else:     
                if df['M'] == 4:
                    return 'BIG SPENDER'
                
                elif df['F'] == 4:
                    return 'LOYAL'
                
                elif df['R'] == 4:
                    return 'ACTIVE'
                
                elif df['R'] == 1:
                    return 'LOST'
                
                elif df['M'] == 1:
                    return 'LIGHT'
                
                return 'REGULARS'
            
        # Create a new column RFM_Level
        df_rfm['RFM_Level'] = df_rfm.apply(rfm_level, axis=1)

        # Calculating average values for each RFM_Level, and return a size of each segment 
        rfm_agg = df_rfm.groupby('RFM_Level').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']}).round(0)

        rfm_agg.columns = rfm_agg.columns.droplevel()
        rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
        rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

        self.df_rfm = df_rfm
        # Reset the index
        self.rfm_agg = rfm_agg.reset_index()


