# PCam Cancer Detector

Aetheris analyzes H&E lymph-node tissue patches with the released center-aware ResNet-34 checkpoint. The browser workstation includes eight reference patches, image upload, four-rotation inference, layer-4 Grad-CAM, threshold calibration and display-only optical enhancement. The Python workstation adds batch analysis, CSV export and model insights.

## Run in a Codespace

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the forwarded Streamlit port. Select **Python workstation** for the native PyTorch tools. The browser build is included. To rebuild it after editing the React source, run `npm ci` and `npm run build` inside `aetheris/`.

## Evaluation

The project originally reported approximately 98.8% accuracy on a 500-patch evaluation and approximately 300,000 training samples. These figures describe the reported experiment. The model produces research scores for individual tissue patches.

<img width="977" height="508" alt="image" src="https://github.com/user-attachments/assets/6ab3ab9e-eee1-42ba-9823-28a251e445b0" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dbb4f036-f61a-4da3-a75d-a6d40ea8a964" />
