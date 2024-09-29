from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
# Load the dataset
df = pd.read_csv('/home/vaishnavi/Pycharm/Practice/project/Reviews.csv')

# Preprocess data
df = df[['Id', 'ProductId', 'Score', 'Summary', 'Text']]
df = df.iloc[:10000, :]
ratings_df = df[['Id', 'ProductId', 'Score']]

# Create pivot table
pivot_table = ratings_df.pivot_table(index='Id', columns='ProductId', values='Score', fill_value=0)


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        if user_id in pivot_table.index:
            user_ratings = pivot_table.loc[user_id, :].values.reshape(1, -1)
            user_item_similarity = cosine_similarity(user_ratings, pivot_table)
            similar_item_indices = user_item_similarity.argsort()[0, ::-1][:]

            recommendations = ratings_df[ratings_df['Id'].isin(similar_item_indices)].head()
        else:
            recommendations = pd.DataFrame()  # Or set to None

    return render_template('index.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)