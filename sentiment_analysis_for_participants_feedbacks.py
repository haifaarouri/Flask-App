import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# def scrape_reviews(base_url, start_page=1, end_page=None):
#     all_reviews = []
#     current_page = start_page
    
#     while True:
#         url = f"{base_url}?page={current_page}"
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         review_containers = soup.find_all('div', class_='reviewText')
        
#         if not review_containers:
#             print(f"No more reviews found at page {current_page}.")
#             break
        
#         for review in review_containers:
#             review_text = review.get_text(separator=" ", strip=True).split(" Review collected by")[0]
#             all_reviews.append(review_text)
        
#         current_page += 1
        
#         if end_page and current_page > end_page:
#             break
    
#     return all_reviews

def analyze_feedback(feedback):
    return sentiment_score(feedback)