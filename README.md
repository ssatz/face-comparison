# face-comparison-service
This finds similarity between faces.
This app runs at port 5050.

# Dependency
    python 3.8.8
    
    flask==2.2.3
    numpy==1.24.2
    pillow==9.5.0
    opencv-python==4.7.0.72
    face_recognition==1.3.0

# Request Data Format
Request method : Post, Request URL : http://localhost/api/upload
	

    {
        'SourceImage': {'Bytes' :  base64string},
        'TargetImage': {'Bytes' :  base64string},
        'SimilarityThreshold' :  number
    } 

# Response Data Format
    {
        "FaceMatches": [
            {
                "Face": {
                    "BoundingBox": {
                        "Height": number,
                        "Left": number,
                        "Top": number,
                        "Width": number,
                    }
                },
                "Similarity": number
            }
    
        ],
        "SourceImageFace": {
            "BoundingBox": {
                "Height": number,
                "Left": number,
                "Top": number,
                "Width": number,
            }
        },
        "UnmatchedFaces": [
            {
                "Face": {
                    "BoundingBox": {
                        "Height": number,
                        "Left": number,
                        "Top": number,
                        "Width": number,
                    }
                },
                "Similarity": number
            }, 
        ]
    }


# Execute 
python runserver.py

