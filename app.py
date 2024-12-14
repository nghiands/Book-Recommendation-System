from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load các file pickle
popular_df = pickle.load(open('popular_update.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

app = Flask(__name__)

# Trang Home
@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].round(2).values)
                           )


# Route cho book_detail
@app.route('/book_detail/<int:book_id>')
def book_detail(book_id):
    # Lấy thông tin sách hiện tại
    try:
        book = popular_df.iloc[book_id]
    except IndexError:
        return "Book ID không hợp lệ!", 404

    # Lấy tên sách hiện tại
    current_book_title = book['Book-Title']

    # Tạo ma trận TF-IDF cho các tiêu đề sách
    tfidf = TfidfVectorizer().fit_transform(popular_df['Book-Title'])

    # Tính điểm tương đồng cosine
    cosine_similarities = cosine_similarity(tfidf[book_id], tfidf).flatten()

    # Tìm các sách có điểm tương đồng cao
    similar_indices = cosine_similarities.argsort()[-5:][::-1]
    similar_books = popular_df.iloc[similar_indices]
    
    # Loại trừ sách hiện tại
    similar_books = similar_books[similar_books.index != book_id]

    # Truyền dữ liệu vào template
    return render_template('book_detail.html',
                           title=book['Book-Title'],
                           author=book['Book-Author'],
                           image=book['Image-URL-M'],
                           votes=book['num_ratings'],
                           rating=round(book['avg_rating'], 2),
                           similar_books=similar_books.to_dict(orient='records'))

# Giao diện recommend
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

# Hàm recommend sách dựa trên input của người dùng
@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    error_message = None  # Biến lưu trữ thông báo lỗi
    data = []  # Biến chứa dữ liệu sách

    # Nếu user chưa nhập gì
    if not user_input:
        error_message ="Chưa có danh sách nào để hiển thị."
    else:
        # Kiểm tra nếu user_input tồn tại trong pt.index
        if user_input in pt.index:
            index = np.where(pt.index == user_input)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

            for i in similar_items:
                item = []
                temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                
                data.append(item)

            print(f"Data: {data}")  # In dữ liệu sách ra console

        else:
            if user_input:
                error_message = "Không tìm thấy sách nào với tên đã nhập. Vui lòng nhập lại tên sách."

    return render_template('recommend.html', data=data, error_message=error_message,user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)


