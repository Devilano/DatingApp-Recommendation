from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
file_path = r'C:\Users\Dell\Desktop\DatingApp\profileA_df.csv'  # Use raw string to avoid escape character issues
profileA_df = pd.read_csv(file_path)

def preprocess_data(df):
    df = df.fillna('Unknown')
    categorical_features = ['status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity']
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature].astype(str))
    numerical_features = ['age']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df, label_encoders, scaler

profileA_df, label_encoders, scaler = preprocess_data(profileA_df)

def calculate_compatibility(user1, user2):
    score = 0
    max_score = 8  # Maximum possible score based on 8 criteria
    if user1['status'] == user2['status']:
        score += 1
    if user1['orientation'] == user2['orientation']:
        score += 1
    if user1['body_type'] == user2['body_type']:
        score += 1
    if user1['diet'] == user2['diet']:
        score += 1
    if user1['drinks'] == user2['drinks']:
        score += 1
    if user1['drugs'] == user2['drugs']:
        score += 1
    if user1['education'] == user2['education']:
        score += 1
    if user1['ethnicity'] == user2['ethnicity']:
        score += 1
    return (score / max_score) * 100  # Return score as a percentage

def find_matches(user_id, df, top_n=5):
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        return [], None  # Return empty list and None if user not found
    user_data = user_data.iloc[0]
    matches = []
    for idx, row in df.iterrows():
        if row['user_id'] != user_id and row['sex'] != user_data['sex'] and row['orientation'] == user_data['orientation']:
            compatibility_score = calculate_compatibility(user_data, row)
            matches.append((row['user_id'], compatibility_score))
    matches = sorted(matches, key=lambda x: x[1], reverse=True)[:top_n]
    return matches, user_data

def collaborative_filtering(user_id, df, n_components=2, top_n=5):
    user_sex = df.loc[df['user_id'] == user_id, 'sex'].values[0]
    opposite_sex_df = df[df['sex'] != user_sex]
    if opposite_sex_df.empty:
        print(f"No opposite sex users found for user_id {user_id}.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Construct interaction matrix based on a meaningful metric
    # Assuming 'interaction' column exists representing user interactions (likes, messages, etc.)
    if 'interaction' not in opposite_sex_df.columns:
        print(f"No interaction column found in the data.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    interaction_matrix = opposite_sex_df.pivot(index='user_id', columns='status', values='interaction').fillna(0)

    # If the number of users is less than or equal to n_components, SVD will fail
    if interaction_matrix.shape[0] <= n_components:
        print(f"Not enough data for collaborative filtering for user_id {user_id}.")
        return pd.DataFrame()  # Return an empty DataFrame

    svd = TruncatedSVD(n_components=n_components)
    matrix = svd.fit_transform(interaction_matrix)
    
    # Check if user exists in the interaction matrix
    if user_id not in interaction_matrix.index:
        print(f"No collaborative filtering data available for user_id {user_id}.")
        return pd.DataFrame()  # Return an empty DataFrame

    user_vector = matrix[interaction_matrix.index == user_id]
    
    similarities = cosine_similarity(user_vector, matrix)[0]
    similar_users = similarities.argsort()[::-1][1:top_n + 1]
    similar_user_ids = interaction_matrix.index[similar_users]
    return df[df['user_id'].isin(similar_user_ids)]

@app.route('/find_matches', methods=['POST'])
def get_matches():
    data = request.json
    user_id = data.get('user_id')
    top_n = data.get('top_n', 5)
    
    print(f"Received request for user_id: {user_id}, top_n: {top_n}")
    
    matches, user_data = find_matches(user_id, profileA_df, top_n)
    
    if user_data is None:
        print(f"No user found with user_id: {user_id}")
        return jsonify({'error': 'User not found'}), 404
    
    print(f"User data: {user_data}")
    print(f"Matches found: {matches}")
    
    match_ids = [match[0] for match in matches]
    match_details = profileA_df[profileA_df['user_id'].isin(match_ids)]
    match_details['compatibility_score'] = match_details['user_id'].apply(lambda x: f"{dict(matches)[x]:.2f}%")
    
    # Additional collaborative filtering matches
    collaborative_matches = collaborative_filtering(user_id, profileA_df, n_components=2, top_n=top_n)
    if collaborative_matches.empty:
        print(f"No collaborative filtering matches for user_id {user_id}.")
        collaborative_match_details = []
    else:
        print(f"Collaborative filtering matches for user_id {user_id}:")
        collaborative_match_details = collaborative_matches.to_dict(orient='records')
    
    return jsonify({
        'user_data': user_data.to_dict(),
        'matches': match_details.drop(columns=['compatibility_score']).to_dict(orient='records'),
        'collaborative_matches': collaborative_match_details
    })

@app.route('/search_matches/<int:user_id>', methods=['GET'])
def search_matches(user_id):
    top_n = int(request.args.get('top_n', 5))
    
    print(f"Received search request for user_id: {user_id}, top_n: {top_n}")
    
    matches, user_data = find_matches(user_id, profileA_df, top_n)
    
    if user_data is None:
        print(f"No user found with user_id: {user_id}")
        return jsonify({'error': 'User not found'}), 404
    
    print(f"User data: {user_data}")
    print(f"Matches found: {matches}")
    
    match_ids = [match[0] for match in matches]
    match_details = profileA_df[profileA_df['user_id'].isin(match_ids)]
    match_details['compatibility_score'] = match_details['user_id'].apply(lambda x: f"{dict(matches)[x]:.2f}%")
    
    # Additional collaborative filtering matches
    collaborative_matches = collaborative_filtering(user_id, profileA_df, n_components=2, top_n=top_n)
    if collaborative_matches.empty:
        print(f"No collaborative filtering matches for user_id {user_id}.")
        collaborative_match_details = []
    else:
        print(f"Collaborative filtering matches for user_id {user_id}:")
        collaborative_match_details = collaborative_matches.to_dict(orient='records')
    
    return jsonify({
        'user_data': user_data.to_dict(),
        'matches': match_details.drop(columns=['compatibility_score']).to_dict(orient='records'),
        'collaborative_matches': collaborative_match_details
    })

if __name__ == '__main__':
    app.run(debug=True)
