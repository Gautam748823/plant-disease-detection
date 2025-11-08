# Model Files

## Download Trained Model

Due to GitHub file size limits, download the trained model from:
- [Google Drive Link](https://drive.google.com/drive/folders/1yyuCp70PndmiVrAgf_9aqd3nfUgO89uw?usp=drive_link)

## Files Needed

- `plant_disease_model_compatible.h5` — Trained CNN model (30–40 MB)  
- `class_labels.json` — Contains class label mappings (included in this repo)

## Instructions

1. Download the model from the link above.  
2. Place the `.h5` model file inside this **models/** folder.  
3. Make sure your folder looks like this:
models/
├── plant_disease_model_compatible.h5
├── class_labels.json
└── README.md

4. Once done, the Flask app will automatically load this model at runtime.
