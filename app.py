from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Try to import flask_cors but continue if not installed
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)

# ---------------------------
# Configuration & DB setup
# ---------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:Kiran%40123@localhost:3306/ecomm"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------------------
# Load CSV/TSV files (recommendations & dataset)
# ---------------------------
TRENDING_PATH = "models/trending_products.csv"
TRAIN_DATA_PATH = "models/clean_data.csv"

# Windows path to your uploaded Walmart dataset (update if you move it)
WALMART_TSV_PATH = r"C:\Users\kiran\PycharmProjects\E-Commerce-Product-Recommendation\models\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"

def safe_read(path):
    try:
        if not path or not os.path.exists(path):
            print(f"safe_read: path missing or does not exist: {path}")
            return pd.DataFrame()
        if str(path).lower().endswith(".tsv") or str(path).lower().endswith(".txt"):
            df = pd.read_csv(path, sep='\t', on_bad_lines='skip', engine='python')
        else:
            df = pd.read_csv(path, on_bad_lines='skip')
        print(f"safe_read: loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return pd.DataFrame()

trending_products = safe_read(TRENDING_PATH)
train_data = safe_read(TRAIN_DATA_PATH)
dataset = safe_read(WALMART_TSV_PATH)  # this is the reviews dataset for dashboard

# ---------------------------
# Database models
# ---------------------------
class Signup(db.Model):
    __tablename__ = "signup"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Signin(db.Model):
    __tablename__ = "signin"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()

# ---------------------------
# Utility helpers
# ---------------------------
def truncate(text: str, length: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= length else text[:length] + "..."

def content_based_recommendations(df: pd.DataFrame, item_name: str, top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # ensure name column exists
    if 'Name' not in df.columns:
        possible_name_cols = [c for c in df.columns if 'name' in c.lower() or 'title' in c.lower()]
        if not possible_name_cols:
            return pd.DataFrame()
        df = df.rename(columns={possible_name_cols[0]: 'Name'})

    df_copy = df.copy()
    df_copy['Tags'] = df_copy.get('Tags', pd.Series([''] * len(df_copy))).fillna('').astype(str)
    name_to_index = {n.strip().lower(): idx for idx, n in df_copy['Name'].astype(str).items()}
    lookup = item_name.strip().lower()

    if lookup not in name_to_index:
        return pd.DataFrame()

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_copy['Tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except Exception:
        return pd.DataFrame()

    item_index = name_to_index[lookup]
    sim_scores = list(enumerate(cosine_sim[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, s in sim_scores if i != item_index][:top_n]

    cols = []
    for c in ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']:
        if c in df_copy.columns:
            cols.append(c)
    if cols:
        return df_copy.iloc[top_indices][cols].reset_index(drop=True)
    else:
        return df_copy.iloc[top_indices].reset_index(drop=True)

# ---------------------------
# Dataset helpers for dashboard endpoints
# ---------------------------
def infer_columns(df: pd.DataFrame):
    cols = { 'id': None, 'name': None, 'category': None, 'rating': None, 'review_text': None, 'image': None }
    if df is None or df.empty:
        return cols

    lower = [c.lower() for c in df.columns]

    # id candidates
    for candidate in ['product_id', 'productid', 'id', 'sku', 'asin', 'product_code']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['id'] = df.columns[i]
                break
        if cols['id']: break

    # name/title candidates
    for candidate in ['product_name', 'producttitle', 'product_title', 'name', 'title']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['name'] = df.columns[i]
                break
        if cols['name']: break

    # category candidates
    for candidate in ['category', 'product_category', 'department', 'cat', 'producttype']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['category'] = df.columns[i]
                break
        if cols['category']: break

    # rating candidates
    for candidate in ['star_rating', 'starrating', 'rating', 'review_rating', 'score', 'stars', 'review_score']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['rating'] = df.columns[i]
                break
        if cols['rating']: break

    # review text candidates
    for candidate in ['review_body', 'review', 'review_text', 'text', 'review_body_text', 'comments']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['review_text'] = df.columns[i]
                break
        if cols['review_text']: break

    # image candidates
    for candidate in ['image', 'image_url', 'imageurl', 'thumbnail', 'img', 'image_link']:
        for i, c in enumerate(lower):
            if candidate == c or candidate in c:
                cols['image'] = df.columns[i]
                break
        if cols['image']: break

    # fallback: if no id but have name, use name as id
    if cols['id'] is None and cols['name'] is not None:
        cols['id'] = cols['name']

    return cols

def build_product_aggregates(df: pd.DataFrame):
    """
    Returns a DataFrame with columns: id, name, category, avg_rating, review_count, image (optional)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['id','name','category','avg_rating','review_count','image'])

    cols = infer_columns(df)
    id_col = cols['id']
    name_col = cols['name']
    cat_col = cols['category']
    rating_col = cols['rating']
    image_col = cols['image']

    group_cols = []
    if id_col:
        group_cols.append(id_col)
    if name_col and name_col != id_col:
        group_cols.append(name_col)
    if not group_cols:
        return pd.DataFrame(columns=['id','name','category','avg_rating','review_count','image'])

    # compute aggregates
    if rating_col:
        aggregated = df.groupby(group_cols).agg(
            avg_rating=pd.NamedAgg(column=rating_col, aggfunc=lambda x: float(pd.to_numeric(x, errors='coerce').dropna().mean()) if len(x)>0 else 0.0),
            review_count=pd.NamedAgg(column=rating_col, aggfunc=lambda x: int(pd.to_numeric(x, errors='coerce').dropna().shape[0]))
        ).reset_index()
    else:
        aggregated = df.groupby(group_cols).size().reset_index(name='review_count')
        aggregated['avg_rating'] = 0.0

    # normalize id/name
    if id_col:
        aggregated = aggregated.rename(columns={group_cols[0]: 'id'})
    else:
        aggregated['id'] = aggregated.index.astype(str)

    if len(group_cols) > 1:
        aggregated = aggregated.rename(columns={group_cols[1]: 'name'})
    else:
        if name_col and name_col in df.columns:
            sample_names = df[[group_cols[0], name_col]].drop_duplicates(group_cols[0]).rename(columns={group_cols[0]: 'id', name_col: 'name'})
            aggregated = aggregated.merge(sample_names, on='id', how='left')
        else:
            aggregated['name'] = aggregated['id']

    # category
    if cat_col and cat_col in df.columns:
        sample_cat = df[[group_cols[0], cat_col]].drop_duplicates(group_cols[0]).rename(columns={group_cols[0]:'id', cat_col:'category'})
        aggregated = aggregated.merge(sample_cat, on='id', how='left')
    else:
        aggregated['category'] = 'Uncategorized'

    # image: take one sample image per product if image column exists
    if image_col and image_col in df.columns:
        sample_img = df[[group_cols[0], image_col]].drop_duplicates(group_cols[0]).rename(columns={group_cols[0]:'id', image_col:'image'})
        aggregated = aggregated.merge(sample_img, on='id', how='left')
    else:
        aggregated['image'] = None

    aggregated = aggregated[['id','name','category','avg_rating','review_count','image']]
    aggregated['avg_rating'] = aggregated['avg_rating'].fillna(0).astype(float).round(2)
    aggregated['review_count'] = aggregated['review_count'].fillna(0).astype(int)

    # if local static images exist for product ids, prefer them (optional)
    def map_local_image(row):
        # try static/images/<id>.jpg or .png
        candidate_png = os.path.join('static', 'images', f"{row['id']}.png")
        candidate_jpg = os.path.join('static', 'images', f"{row['id']}.jpg")
        if os.path.exists(candidate_png):
            return f"/{candidate_png.replace(os.path.sep, '/')}"
        if os.path.exists(candidate_jpg):
            return f"/{candidate_jpg.replace(os.path.sep, '/')}"
        # else use dataset image if present (could be a URL)
        if pd.notna(row['image']) and row['image'] not in (None, ''):
            return row['image']
        return None

    aggregated['image'] = aggregated.apply(map_local_image, axis=1)

    return aggregated

# Pre-compute aggregates and log counts
product_aggregates = build_product_aggregates(dataset)
try:
    print(f"Precomputed product_aggregates: {len(product_aggregates)} unique products")
    if not product_aggregates.empty:
        sample = product_aggregates.head(3).to_dict(orient='records')
        print("Sample products:", sample)
except Exception:
    pass

# ---------------------------
# Routes
# ---------------------------
random_image_urls = [
    "static/img_1.png", "static/img_2.png", "static/img_3.png", "static/img_4.png",
    "static/img_5.png", "static/img_6.png", "static/img_7.png", "static/img_8.png",
]

@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(min(8, len(trending_products)))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template(
        'index.html',
        trending_products=trending_products.head(8) if not trending_products.empty else pd.DataFrame(),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=random.choice(price)
    )

@app.route("/main")
def main():
    return render_template('main.html')

@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not email or not password:
            flash("Please fill all fields.", "warning")
            return redirect(url_for('signup'))

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(min(8, len(trending_products)))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template(
            'index.html',
            trending_products=trending_products.head(8) if not trending_products.empty else pd.DataFrame(),
            truncate=truncate,
            random_product_image_urls=random_product_image_urls,
            random_price=random.choice(price),
            signup_message='User signed up successfully!'
        )
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('signinUsername', '').strip()
        password = request.form.get('signinPassword', '').strip()

        if not username or not password:
            flash("Please enter username and password.", "warning")
            return redirect(url_for('signin'))

        new_signin = Signin(username=username, password=password)
        db.session.add(new_signin)
        db.session.commit()

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(min(8, len(trending_products)))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template(
            'index.html',
            trending_products=trending_products.head(8) if not trending_products.empty else pd.DataFrame(),
            truncate=truncate,
            random_product_image_urls=random_product_image_urls,
            random_price=random.choice(price),
            signup_message='User signed in successfully!'
        )
    return render_template('signin.html')

@app.route("/recommendations", methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod', '').strip()
        try:
            nbr = int(request.form.get('nbr', 5))
        except (TypeError, ValueError):
            nbr = 5

        if not prod:
            message = "Please provide a product name to get recommendations."
            trending = train_data.head(8) if not train_data.empty else pd.DataFrame()
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(min(8, len(trending)))]
            return render_template('main.html', message=message, trending_products=trending,
                                   random_product_image_urls=random_product_image_urls, truncate=truncate)

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
        if content_based_rec is None or content_based_rec.empty:
            message = f"No recommendations available for product: {prod}"
            trending = train_data.head(8) if not train_data.empty else pd.DataFrame()
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(min(8, len(trending)))]
            return render_template('main.html', message=message, trending_products=trending,
                                   random_product_image_urls=random_product_image_urls, truncate=truncate,
                                   content_based_rec=pd.DataFrame())

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        trending = train_data.head(8) if not train_data.empty else pd.DataFrame()
        return render_template(
            'main.html',
            content_based_rec=content_based_rec,
            truncate=truncate,
            random_product_image_urls=random_product_image_urls,
            random_price=random.choice(price),
            trending_products=trending
        )

    return redirect(url_for('main'))

# ---------------------------
# API: products with pagination + filtering
# ---------------------------
@app.route("/api/products", methods=['GET'])
def api_products():
    global product_aggregates
    df = product_aggregates.copy() if (product_aggregates is not None) else pd.DataFrame()

    # query params
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 25))
    min_rating = request.args.get('min_rating', type=float)
    category = request.args.get('category', type=str)
    q = request.args.get('q', type=str)

    if min_rating is not None and 'avg_rating' in df.columns:
        df = df[df['avg_rating'] >= float(min_rating)]
    if category and 'category' in df.columns:
        df = df[df['category'].astype(str).str.lower() == category.strip().lower()]
    if q and 'name' in df.columns:
        ql = q.strip().lower()
        df = df[df.apply(lambda r: ql in str(r.get('id','')).lower() or ql in str(r.get('name','')).lower(), axis=1)]

    # ensure name exists
    if not df.empty and 'name' not in df.columns:
        df['name'] = df['id']

    total = len(df)
    # sort by avg_rating, then review_count
    if 'avg_rating' in df.columns:
        df = df.sort_values(by=['avg_rating', 'review_count'], ascending=[False, False])
    else:
        df = df.sort_values(by='review_count', ascending=False)

    # pagination slice
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end].copy() if not df.empty else pd.DataFrame(columns=df.columns)

    # normalize output field names
    def row_to_record(row):
        return {
            'id': row.get('id'),
            'name': row.get('name'),
            'category': row.get('category'),
            'avg_rating': float(row.get('avg_rating')) if row.get('avg_rating') is not None else None,
            'review_count': int(row.get('review_count')) if row.get('review_count') is not None else 0,
            'image': row.get('image') if 'image' in row.index else None
        }

    records = [row_to_record(r) for _, r in page_df.iterrows()]

    return jsonify({
        'records': records,
        'total': total,
        'page': page,
        'per_page': per_page
    })

@app.route("/api/categories", methods=['GET'])
def api_categories():
    if product_aggregates is None or product_aggregates.empty:
        return jsonify([])
    cats = product_aggregates['category'].fillna('Uncategorized').unique().tolist()
    cats = sorted([str(c) for c in cats])
    return jsonify(cats)

@app.route("/api/reviews", methods=['GET'])
def api_reviews():
    if dataset is None or dataset.empty:
        return jsonify([])

    cols = infer_columns(dataset)
    id_col = cols['id']
    name_col = cols['name']
    review_col = cols['review_text']
    rating_col = cols['rating']

    df = dataset.copy()

    product_id = request.args.get('product_id')
    product_name = request.args.get('product_name')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    if product_id and id_col:
        df = df[df[id_col].astype(str) == str(product_id)]
    if product_name and name_col:
        df = df[df[name_col].astype(str).str.contains(str(product_name), case=False, na=False)]

    start = (page - 1) * per_page
    end = start + per_page

    out = []
    for _, row in df.iloc[start:end].iterrows():
        out.append({
            'product_id': row[id_col] if id_col in row.index else None,
            'product_name': row[name_col] if name_col in row.index else None,
            'rating': float(row[rating_col]) if (rating_col in row.index and pd.notna(row[rating_col])) else None,
            'review': str(row[review_col]) if (review_col in row.index and pd.notna(row[review_col])) else None
        })
    return jsonify(out)

# ---------------------------
# Chart endpoints
# ---------------------------
@app.route("/api/category-average-ratings", methods=['GET'])
def api_category_avg_ratings():
    """Return average rating per category for charting."""
    global product_aggregates
    if product_aggregates is None or product_aggregates.empty:
        return jsonify([])

    df = product_aggregates.copy()
    out = df.groupby('category').agg(
        avg_rating=pd.NamedAgg(column='avg_rating', aggfunc=lambda x: float(pd.to_numeric(x, errors='coerce').dropna().mean()) if len(x)>0 else 0.0)
    ).reset_index().sort_values(by='avg_rating', ascending=False)
    records = out.to_dict(orient='records')
    return jsonify(records)

@app.route("/api/top-reviewed", methods=['GET'])
def api_top_reviewed():
    """Return top N products by review_count"""
    global product_aggregates
    if product_aggregates is None or product_aggregates.empty:
        return jsonify([])

    n = int(request.args.get('n', 10))
    df = product_aggregates.copy()
    df = df.sort_values(by='review_count', ascending=False).head(n)
    records = df[['id','name','category','avg_rating','review_count','image']].to_dict(orient='records')
    return jsonify(records)

@app.route("/api/refresh", methods=['POST'])
def api_refresh():
    global dataset, product_aggregates
    try:
        dataset = safe_read(WALMART_TSV_PATH)
        product_aggregates = build_product_aggregates(dataset)
        print(f"api_refresh: reloaded dataset rows={len(dataset)}, products={len(product_aggregates)}")
        return jsonify({"status":"ok", "message":"refreshed", "products": len(product_aggregates)}), 200
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)}), 500

# ---------------------------
# Dashboard route (renders Dashboard.html — note capital D if your file uses that name)
# ---------------------------
@app.route("/dashboard")
def dashboard_page():
    """
    Render the dashboard template. Dashboard template JS can still call /api/* endpoints,
    but we'll pass a few handy things precomputed so template can show immediate data if desired.
    """
    # categories list and top reviewed sample (for immediate rendering)
    cats = []
    top_reviewed = []
    category_avg = []

    if product_aggregates is not None and not product_aggregates.empty:
        cats = sorted(product_aggregates['category'].fillna('Uncategorized').unique().tolist())
        top_reviewed = product_aggregates.sort_values(by='review_count', ascending=False).head(10)[['id','name','category','avg_rating','review_count','image']].to_dict(orient='records')
        # small sample of category averages
        category_avg = product_aggregates.groupby('category').agg(avg_rating=pd.NamedAgg(column='avg_rating', aggfunc='mean')).reset_index().sort_values(by='avg_rating', ascending=False).to_dict(orient='records')

    # Render the template (use exact filename: 'Dashboard.html' if that's what you saved)
    # If your template file is lowercase 'dashboard.html' rename accordingly.
    return render_template('Dashboard.html', categories=cats, top_reviewed=top_reviewed, category_avg=category_avg)

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    # debug: print registered routes
    print("\n--- Registered routes ---")
    print(app.url_map)
    print("-------------------------\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
