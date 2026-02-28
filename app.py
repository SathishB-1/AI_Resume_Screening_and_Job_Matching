from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load lighter model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

df = pd.read_pickle("job_data.pkl")
job_embeddings = np.load("job_embeddings.npy")


@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        resume_text = request.form.get("resume_text")
        file = request.files.get("resume_file")

        # PDF Upload
        if file and file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            resume_text = ""

            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    resume_text += txt

        if resume_text:

            resume_embedding = model.encode(
                [resume_text],
                normalize_embeddings=True
            )

            similarities = cosine_similarity(
                resume_embedding,
                job_embeddings
            )[0]

            top_indices = similarities.argsort()[-5:][::-1]

            for idx in top_indices:
                results.append({
                    "title": df.iloc[idx]["job_title"],
                    "location": df.iloc[idx].get(
                        "location", "Not Provided"
                    ),
                    "company": df.iloc[idx].get(
                        "organization", "Not Provided"
                    ),
                    "score": round(similarities[idx]*100, 2)
                })

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)