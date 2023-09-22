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
### Request method : Post, Request API : /api/upload

Example API : http://10.10.10.10:5050/api/upload	

    {
        'SourceImage': {'Bytes' :  base64string},
        'TargetImage': {'Bytes' :  base64string},
        'SimilarityThreshold' :  number
    } 
### Request method : Post, Request API : /api/upload/images
Example API : http://10.10.10.10:5050/api/upload/images	

    {
        'source_image': original_source_image_formdata,
        'target_image': original_target_image_formdata,
        'SimilarityThreshold' :  number
    } 

# Response Data Format(JSON)

    '{"Similarity": number}'

# Test page
We can easily test through web page. 

The URL of web page : /api/page

Example URL : http://10.10.10.10:5050/api/page

# Execute 
python runserver.py

