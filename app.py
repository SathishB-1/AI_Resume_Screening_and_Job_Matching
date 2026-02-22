from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_pickle("job_data (1).pkl")
job_embeddings = np.load("job_embeddings (1).npy")


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        resume_text = request.form.get("resume_text")

        file = request.files.get("resume_file")

        if file and file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            resume_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

        if resume_text:
            resume_embedding = model.encode([resume_text])
            similarities = cosine_similarity(resume_embedding, job_embeddings)[0]

            top_indices = similarities.argsort()[-5:][::-1]

            for idx in top_indices:
                results.append({
                    "title": df.iloc[idx]['job_title'],
                    "location": df.iloc[idx].get('location', 'Not Provided'),
                    "company": df.iloc[idx].get('company', 'Not Provided'),
                    "score": round(similarities[idx] * 100, 2)
                })

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)