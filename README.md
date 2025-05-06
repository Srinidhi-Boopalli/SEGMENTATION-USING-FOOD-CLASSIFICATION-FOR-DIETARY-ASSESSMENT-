Food Segmentation and Classification (Food101 - 20 Categories)

This project performs **semantic segmentation** using **SLIC superpixels** and classifies food images using **Gabor features** and an **SVM classifier**. It is tailored for a subset of the **Food101 dataset**, using 20 predefined categories.


 Dataset

- Dataset: [Food-101](https://www.kaggle.com/datasets/kmader/food41])
- The project uses a subset of 20 categories 
- Each category can have up to 50 images for training/demo purposes 


 Model Overview

- **Segmentation**: `skimage.segmentation.slic()` (with `compactness=30`, `n_segments=100`)
- **Feature Extraction**: Gabor filter bank (4 orientations × 3 frequencies)
- **Classifier**: `sklearn.svm.SVC` (RBF kernel)


 Running the Code
 Setup

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib tqdm Pillow
```
 Execute

Edit the `DATASET_PATH` in the script to point to your dataset location:

```python
DATASET_PATH = "path_to_food101_subset"
```

Then run the script:

```bash
python food_segmentation_classification.py
```

The model will:

1. Load and preprocess images.
2. Segment them with SLIC.
3. Extract Gabor texture features.
4. Train an SVM model.
5. Report accuracy and show segmentation results.


Categories Used

```text
['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beignets',
 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche',
 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros',
 'clam_chowder', 'club_sandwich']
```


 Sample Visualization

The script displays SLIC superpixel segmentation on a sample image and predicts its class label.

To test a different category, change this line:

```python
sample_category = 'cheesecake'  # Or any other category from the list
```

Output

- Classification report (precision, recall, F1-score)
- Segmentation visualizations using both `slic_zero=True` and standard SLIC
- Optional prediction display
