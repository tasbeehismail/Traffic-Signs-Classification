
# Traffic Signs Classification 

This repository is part of our final project for the Computer Vision Lab. My team and I developed this project by researching deep learning techniques, specifically CNNs, to classify traffic sign images using TensorFlow, Keras, and OpenCV.
#### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone git@github.com:tasbeehismail/Traffic-Signs-Classification.git
   cd Traffic-Signs-Classification
   ```

2. **Create a virtual environment:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python3 -c "import tensorflow, keras, cv2, pandas, numpy, matplotlib"
   ```
---

## Running the Project

### 1. Training the Model (`train.py`)

Once you've installed the dependencies and ensured the dataset is in place, you can begin training the model.

1. Make sure the `myData` folder is present, containing your images organized into subfolders (each representing a traffic sign class).
   ```
    ├── myData/                # Dataset folder containing subfolders for each traffic sign class
    │   ├── <class_name>/     
    │   ├── <class_name>/      
    │   └── ...
    │
    ```
2. To train the model, use the following command in your terminal:
   ```bash
   python3 train.py
   ```

   This will start training the CNN on your dataset, and the model will be saved as `model_trained.h5`.

### 2. Testing the Model (`test.py`)

To evaluate the performance of the trained model:

1. After training, you can run the testing script with:
   ```bash
   python3 test.py
   ```

   This will load the trained model and evaluate it on the test dataset, printing out the performance metrics.

---

