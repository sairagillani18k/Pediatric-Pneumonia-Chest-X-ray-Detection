# Pediatric-Pneumonia-Chest-X-ray-Detection
## Teachable Machine Image Classification Project

This project demonstrates how to create a machine learning model for detecting pediatric pneumonia using chest X-ray images. The model was trained using Google’s Teachable Machine, a user-friendly platform that allows anyone to build AI models without coding expertise. This README provides a comprehensive guide to replicating the project and running the model in a Python environment for inference.

## Project Overview

Pediatric pneumonia is a serious respiratory infection, particularly dangerous for children under the age of five. Early detection through chest X-rays is crucial for prompt and effective treatment. In this project, we trained a machine learning model using chest X-ray images to distinguish between healthy lungs and those affected by pneumonia. The model was exported to a TensorFlow SavedModel format and integrated into a Python script to perform real-time video analysis.

## Steps to Create and Use the Model

To follow the complete procedure, you can refer to my tutorial [here](https://github.com/sairagillani18k/No-Code-AI-Building-a-Simple-Image-Classifier-with-Google-s-Teachable-Machine).

### 1. Start a New Project

- Navigate to the [Teachable Machine](https://teachablemachine.withgoogle.com/train) website.
- Select the `Image Project` option.
- You can either start a new project or continue an existing one from your Google Drive.

### 2. Choose Your Model Type

- Select `Standard image model` for this project to utilize the full range of Teachable Machine’s image classification capabilities.

### 3. Train Your Model

- Upload chest X-ray images into two categories: `Normal` and `Pneumonia`.
- Click the `Train Model` button. The platform will train the model to recognize patterns in the X-ray images that correspond to either healthy lungs or pneumonia.
- You can see my training parameters in the image below:

  ![Results](/Pictures/image.png)

### 4. Export the Model

- After training, click on `Export Model`.
- Choose `TensorFlow SavedModel` as the export format.
- Download the exported model along with the `labels.txt` file to your local machine.

### 5. Running the Model in Python

- Install the required libraries:

    ```bash
    pip install tensorflow opencv-python numpy
    ```

- Enter the correct paths for the model files, labels.txt, and the video on which you want to perform inference.

- Enter the following command to run the script:

    ```bash
    python main.py
    ```

### 6. Results

- In this project, the trained model was used to detect pneumonia in pediatric patients through X-ray images. Our inference was performed on a video demonstrating the model's capability to identify pneumonia from X-ray frames.

- You can watch the inference demonstration in this video: [Inference Video](https://www.youtube.com/watch?v=HTwT0GTBgdI&pp=ygUfUGVkaWF0cmljIFBuZXVtb25pYSBDaGVzdCBYLXJheQ%3D%3D).

- Additionally, a sample video on which the inference was run can be found in this repository.

### Importance of This Project

Pneumonia is a leading cause of morbidity and mortality among children, particularly in low-resource settings where diagnostic tools are limited. This project contributes to the healthcare field by providing a machine learning model that can assist in the early detection of pneumonia, potentially leading to timely and life-saving treatment. The use of automated diagnostic tools like this can bridge the gap in healthcare accessibility and improve outcomes for pediatric patients worldwide.
