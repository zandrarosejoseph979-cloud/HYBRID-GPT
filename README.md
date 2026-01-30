Hybrid Ensemble Vision System with GPT-4o and ConvNeXt

This repository contains the implementation of the project **"Hybrid Ensemble Vision System with GPT-4o and ConvNeXt for Robust Image Classification"**, developed as part of an academic research project and documented in IEEE format.

The system integrates multimodal reasoning using GPT-4o with deep visual feature extraction from ConvNeXt to achieve robust and efficient image classification under distribution shifts and limited data conditions.

---

 Project Highlights

* Progressive fine-tuning of GPT-4o using multi-resolution images
* CNN baseline models: ResNet-50 and ConvNeXt-Tiny
* Hybrid ensemble approach combining multimodal reasoning and CNN predictions
* Interactive Gradio web interface for real-time image classification
* Evaluation across accuracy, F1-score, inference time, and robustness

---

 Repository Structure

```
Hybrid-Ensemble-Vision-System/
├── data/              # Dataset and sample images
├── notebooks/         # Exploratory analysis notebooks
├── src/               # Training and inference scripts
├── app/               # Gradio web application
├── results/           # Output metrics, plots, and logs
├── docs/              # IEEE paper and system diagrams
├── experiments/       # Experiment tracking logs
├── requirements.txt   # Python dependencies
└── README.md          # Project overview
```

---

 Installation

```bash
git clone https://github.com/your-username/Hybrid-Ensemble-Vision-System.git
cd Hybrid-Ensemble-Vision-System
pip install -r requirements.txt
```


Running the System

 Train CNN Models

```bash
python src/train_resnet.py
python src/train_convnext.py


Fine-tune GPT-4o (API-based)

```bash
python src/gpt4o_finetune.py`

 Run Ensemble Inference

```bash
python src/inference.py --image data/sample_images/example.jpg


 Launch Web Interface

```bash
python app/app.py
```



 Results Summary

| Model         | Accuracy | F1-Score | Inference Time |
| ------------- | -------- | -------- | -------------- |
| GPT-4o        | 87%      | 0.86     | 0.7s           |
| ResNet-50     | 82%      | 0.81     | 1.3s           |
| ConvNeXt-Tiny | 79%      | 0.78     | 1.5s           |
| Hybrid Model  | **89%**  | **0.88** | **0.8s**       |


 Technologies Used

* Python
* PyTorch
* OpenAI API (GPT-4o)
* Gradio
* NumPy, OpenCV, Matplotlib




Sandra Rose Joseph
Department of Computer Science and Engineering
Vidya Academy of Science and Technology, Thrissur
📧 [zandrarosejoseph979@gmail.com](mailto:zandrarosejoseph979@gmail.com)

