FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app

RUN pip install flask==2.2.3
RUN pip install numpy==1.24.2
RUN pip install pillow==9.5.0
RUN pip install opencv-python==4.7.0.72
RUN pip install face_recognition==1.3.0

# Run app.py when the container launches
CMD ["python", "./app/runserver.py"]