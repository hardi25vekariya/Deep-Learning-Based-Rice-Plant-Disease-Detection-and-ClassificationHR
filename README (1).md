# Rice Disease Detection

Compare 6 CNN models (MobileNetV2, ResNet50, VGG16, EfficientNetB0, InceptionV3, DenseNet121) on the
Kaggle rice disease dataset, then use the best model in a Flask web app.

## Files
- `train_models.ipynb` - training notebook (run on Google Colab with GPU)
- `app.py` - Flask web app
- `templates/index.html` - upload page
- `requirements.txt` - python packages for the web app
- `models/` - put `best_model.keras` and `class_names.json` here after training
- `build_notebook.py` - (optional) regenerates the notebook

## Step 1: Train (Google Colab)
1. Open https://colab.research.google.com -> File -> Upload notebook -> `train_models.ipynb`
2. Runtime -> Change runtime type -> **GPU (T4)**
3. Run all cells top to bottom. The dataset downloads automatically.
4. At the end `rice_best_model.zip` (best model + class names) and `rice_results.zip`
   (graphs + comparison table for your paper) are downloaded.

## Step 2: Run the web app (your computer)
1. Extract `rice_best_model.zip` and copy `best_model.keras` and `class_names.json` into the `models/` folder.
2. Install packages: `pip install -r requirements.txt`
   (use the same TensorFlow version as Colab, check the first output cell of the notebook)
3. Start: `python app.py`
4. Open http://127.0.0.1:5000 , upload a leaf image and see the disease.
