#Import Modules
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup as bs
import datetime
from transformers import pipeline
import os

#Instantiate NewsApiClient instance
newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_API_KEY'))

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
            pub_date = datetime.date.fromisoformat(pub_date)
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


sentiment_analyser = News_Sentiment_Analyser()
topic = str(input('Please enter topic of interest: '))
print('Average Sentiment is: ',sentiment_analyser.get_average_sentiment(topic))
