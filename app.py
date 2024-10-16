from flask import Flask, request, jsonify
import sentiment_analysis_for_participants_feedbacks as sentiment_analysis
import pandas as pd
import json
import pickle
import nltk
from nltk.corpus import wordnet
import spacy
from fuzzywuzzy import process
from langdetect import detect
from sentence_transformers import SentenceTransformer
import scipy.spatial

nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__) 

@app.route('/analyze', methods=['POST'])
def analyze():
    feedbacks_str = request.json.get('feedbacks', '[]')
    try:
        feedbacks = json.loads(feedbacks_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in 'feedbacks'"}), 400
    
    results = []
    for feedback in feedbacks:
        data = feedback['data']
        
        # Convert the data to a DataFrame
        headers = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=headers)
        
        # Calculate sentiment scores for each question
        sentiment_columns = {}
        for column in headers[3:]:  # Skip the first, second and third columns (Horodateur + sessionId + participantId)
            sentiment_column = f"{column}_sentiment"
            df[sentiment_column] = df[column].apply(lambda x: sentiment_analysis.analyze_feedback(str(x)[:512]))
            sentiment_columns[column] = sentiment_column
        
        # Convert the DataFrame back to a dictionary
        result = df.to_dict(orient='records')
        results.append({
            'data': result
        })
    
    return jsonify(results)


# Load the model and data
with open('model_knn.pkl', 'rb') as file:
    model_knn = pickle.load(file)

course_features_df = pd.read_pickle("course_features_df.pkl")

# Load the spaCy model
nlp_en = spacy.load('en_core_web_md')
nlp_fr = spacy.load('fr_core_news_md')
nlp_sp = spacy.load('es_core_news_md')
nlp_it = spacy.load('it_core_news_md')

def load_model(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return nlp_en  # english
        elif lang == 'fr':
            return nlp_fr  # french
        elif lang == 'es':
            return nlp_sp  # Spanish model
        elif lang == 'it':
            return nlp_it  # Italian model
        else:
            print(f"Warning: Language '{lang}' not supported. Falling back to English.")
            return spacy.load('en_core_web_md')  # Default to English model
    except Exception as e:
        print("Error loading model:", e)


# Function to calculate similarity (based on the meaning of words, not the words itselfs or the characters)
def get_similarity(course_name, query):
    nlp = load_model(course_name)
    return nlp(course_name).similarity(nlp(query))

# Function to search for similar courses
def search_courses(query, df, top_n=5):
    similarities = {course: get_similarity(course, query) for course in df.index}
    # Sort courses based on similarity scores in descending order
    sorted_courses = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_courses[:top_n]


# Function to perform fuzzy search (based on the words similarities or synonyms or the characters)
def fuzzy_search(query, df, top_n=5):
    course_names = df.index.tolist()
    results = process.extract(query, course_names, limit=top_n)
    return results


def combined_search(query, df, top_n=5):
    fuzzy_results = fuzzy_search(query, df, top_n=20)  # Get top 20 results from fuzzy search
    # Extract just the course names from fuzzy results
    course_names = [result[0] for result in fuzzy_results]
    # Filter df to include only these courses
    filtered_df = df.loc[course_names]
    # Now apply semantic search on this filtered DataFrame
    final_results = search_courses(query, filtered_df, top_n=top_n)
    return final_results


@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    course_name = content['course_name']
    n_recommendations = content['n_recommendations']

    # Check if the course exists in the DataFrame
    if course_name in course_features_df.index:
        # Directly make recommendations if the course exists
        recommendations = make_recommendations(model_knn, course_features_df, course_name, n_recommendations)
    else:
        # Use combined_search to find similar courses if the course does not exist
        similar_courses = combined_search(course_name, course_features_df, top_n=10)
        if similar_courses:
            # Take only the most similar course name
            most_similar_course = similar_courses[0][0]
            # Make recommendations based on the most similar course
            recommendations = make_recommendations(model_knn, course_features_df, most_similar_course, n_recommendations)
            # Combine recommendations and similar courses into a single dictionary
            recommendations["similar_courses"] = [course[0] for course in similar_courses]
        else:
            # Handle case where no similar courses are found
            recommendations = {"error": "Aucune formation similaire n’a été trouvé !"}

    return jsonify(recommendations)

def make_recommendations(model, data, course_name, n_recommendations):
    course_index = data.index.tolist().index(course_name)
    distances, indices = model.kneighbors(data.iloc[course_index, :].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
    recommended_courses = [data.index[i] for i in indices.flatten() if i != course_index]
    
    recommendations = {}
    for i, (course, distance) in enumerate(zip(recommended_courses, distances.flatten()[1:]), 1):
        recommendations[f"{i}"] = {"course": course, "distance": distance}
    
    return recommendations

    

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/find_matches', methods=['POST'])
def find_matches():
    data = request.json
    specialty_embeddings = model.encode(data['specialties'])
    requirement_embeddings = model.encode(data['requirements'])
    
    # Calculate cosine similarities
    similarities = scipy.spatial.distance.cdist(specialty_embeddings, requirement_embeddings, "cosine")
    similarity_scores = 1 - similarities  # Convert distances to similarities
    print(data['specialties'])
    print(data['requirements'])
    print(similarities)
    print(similarity_scores)
    return jsonify(similarity_scores.tolist())


if __name__ == '__main__':
    app.run(debug=True, port=5000)