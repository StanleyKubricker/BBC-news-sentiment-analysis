# Determining the average sentiment of BBC News articles published on a given topic
When given a search term, this code retrieves the URL of all relevant BBC News articles published within the last 30 days using the NewsApi API (https://newsapi.org/).

These URL's are then accessed via the requests library (https://requests.readthedocs.io/en/latest/) and scraped using the Beautiful Soup library (https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to retrieve the content of the article.

Each article is then passed to a Hugging Face Transformers Sentiment Analysis Pipeline, which returns a classification and a score for each article. The score represnets how extreme the classification is; for example if two articles are classified as Positive, the one with the highest score can be considered "more positive".

These sentiment scores are then summed and averaged over the number of relevant articles returned to give a measure of the averge sentiment of BBC articles published on the given topic within the past 30 days.
