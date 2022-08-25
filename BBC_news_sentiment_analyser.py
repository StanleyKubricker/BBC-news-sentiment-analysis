#Import Modules
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup as bs
from datetime import date
from transformers import pipeline
import os
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd

#Instantiate NewsApiClient instance
news_api_key = '814ba6ded1ad42aeb0e6f08e01c73565'
newsapi = NewsApiClient(api_key=news_api_key)

#Instantiate huggingface transformers sentiment classifier
classifier = pipeline('sentiment-analysis')

class News_Sentiment_Analyser:
    def __init__(self):
        self.url_list = []
        self.date_list = []
        self.articles_list = []

    #Method to retrieve bbc articles on given topic using NewsApi API and scrape content using BeautifulSoup
    def get_bbc_articles(self, topic):
        bbc_url_list = []
        bbc_date_list = []
        bbc_articles_list = []

        #Use newsapi.get_everything method to return all bbc articles on given topic published within past month
        bbc_articles = newsapi.get_everything(q=str(topic),
                                              sources='bbc-news',
                                              domains='bbc.co.uk',
                                              language='en',
                                              sort_by='relevancy')

        #Find the date on which each article was published
        for i in bbc_articles['articles']:
            bbc_url_list.append(i['url'])
            pub_date = i['publishedAt']
            pub_date = pub_date[:pub_date.find('T')]
            pub_date = date.fromisoformat(pub_date)
            bbc_date_list.append(pub_date)

        #Find text from each article
        for u in bbc_url_list:
            article = requests.get(u)
            soup = bs(article.content, 'html.parser')
            #Navigating HTML
            body = soup.find(property=='ArticleWrapper')
            text = body.find_all('p', {'class' : 'ssrcss-1q0x1qg-Paragraph eq5iqo00'})
            text = [x.text for x in text if 'ssrcss-7uxr49-RichTextContainer e5tfeyi1' in str(x.parent)]
            bbc_articles_list.append(text)

        self.url_list = bbc_url_list
        self.date_list = bbc_date_list
        self.articles_list = bbc_articles_list

    #Use huggingface transformers sentiment analyser to find average sentiment of bbc news articles published on given topic
    def get_average_sentiment(self, topic):
        self.get_bbc_articles(topic)
        if len(self.articles_list) > 0:
            totals = []

            for article in self.articles_list:

                #classifier returns list of dictionaries, one for each line passed to classifier
                article_sentiments = classifier(article)

                total_sentiment = 0

                #For each dictionary, adjust total sentiment according to classification
                for sentiment_instance in article_sentiments:
                    if sentiment_instance['label'] == 'POSITIVE':
                        total_sentiment += sentiment_instance['score']
                    elif sentiment_instance['label'] == 'NEGATIVE':
                        total_sentiment -= sentiment_instance['score']

                totals.append(total_sentiment)

            #Return mean sentiment over all published articles
            return sum(totals) / len(self.articles_list)

        else:
            print('No articles found.')

    #Method for producing plots of daily sentiment of given topic over past 31 days
    def get_daily_sentiment_plots(self, topic):
        #Call get_bbc_articles method
        self.get_bbc_articles(topic)

        #Generate dataframe of dates and articles returned via get_bbc_articles method
        data_dict = {}

        for i in range(len(self.date_list)):
            if self.date_list[i] not in data_dict.keys():
                if len(self.articles_list[i]) != 0:
                    data_dict[self.date_list[i]] = [self.articles_list[i]]
            else:
                if len(self.articles_list[i]) != 0:
                    data_dict[self.date_list[i]].append(self.articles_list[i])

        df1 = pd.DataFrame({'Dates':data_dict.keys(), 'Articles':data_dict.values()})

        #Generate list of sentiements for each day
        sentiments = []
        #For each day, find average sentiment of all articles and append to sentiments list
        for i in df1['Articles']:
            daily_sentiment = []
            for article in i:
                article_sentiments = classifier(article)

                total_sentiment = 0

                for sentiment_instance in article_sentiments:
                    if sentiment_instance['label'] == 'POSITIVE':
                        total_sentiment += sentiment_instance['score']
                    elif sentiment_instance['label'] == 'NEGATIVE':
                        total_sentiment -= sentiment_instance['score']

                daily_sentiment.append(total_sentiment)
            sentiments.append(sum(daily_sentiment)/len(i))
        df1['Sentiment'] = sentiments

        #Generate rows for missing dates in the past 31 days
        df1["Dates"] = pd.to_datetime(df1["Dates"])
        df1.sort_values(by='Dates', inplace=True)
        df1.index = pd.DatetimeIndex(df1.Dates)
        current_date = datetime.today()
        earliest_date = current_date - timedelta(days=31)
        current_date = current_date.strftime('%Y-%m-%d')
        earliest_date = earliest_date.strftime('%Y-%m-%d')
        idx = pd.date_range(earliest_date, current_date)
        df1 = df1.reindex(idx, fill_value = 0)
        df1 = df1[['Articles','Sentiment']]

        #Count number of articles published on each date
        article_counts = []
        for i in df1.Articles:
            if type(i) != type([]):
                article_counts.append(0)
            else:
                article_counts.append(len(i))
        df1['Article Counts'] = article_counts

        #Generate Plotly plots of data
        # Creating trace1
        trace1 = go.Scatter(
                            x = df1.index,
                            y = df1.Sentiment,
                            mode = "lines+markers",
                            name = "Sentiment",
                            marker = dict(color = 'rgba(16, 112, 2, 0.8)'))

        # Creating trace2
        trace2 = go.Scatter(
                            x = df1.index,
                            y = df1['Article Counts'],
                            mode = "lines+markers",
                            name = "Article Count",
                            marker = dict(color = 'rgba(251, 0, 0, 0.8)'))

        data = [trace1]
        data2 = [trace2]

        layout = dict(title = f'Sentiment over past 31 days for BBC articles on {topic}',
                      xaxis= dict(title= 'Date',ticklen= 1,zeroline= False)
                     )

        layout2 = dict(title = f'Article count over past 31 days for BBC articles on {topic}',
                      xaxis= dict(title= 'Date',ticklen= 1,zeroline= False)
                     )

        fig = dict(data = data, layout = layout)
        iplot(fig)

        fig2 = dict(data = data2, layout = layout2)
        iplot(fig2)


sentiment_analyser = News_Sentiment_Analyser()
topic = str(input('Please enter topic of interest: '))
print('Average Sentiment is: ',sentiment_analyser.get_average_sentiment(topic))
sentiment_analyser.get_daily_sentiment_plots(topic)
