from flask import Flask, request, redirect, url_for, render_template
import json

from face_comparison.face_similarity import *

app = Flask(__name__)


@app.route("/api/page")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Handle the upload of a file."""
    base64_images = request.form

    source_string = base64_images['SourceImage[Bytes]']
    target_string = base64_images['TargetImage[Bytes]']
    similarity_threshold = base64_images['SimilarityThreshold']

    face_similarity = FaceSimilarity(source_string, target_string, similarity_threshold)

    return json.dumps(face_similarity.response_data)

