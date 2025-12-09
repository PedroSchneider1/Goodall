# Automated behavioral analysis of primates via audio signal processing and machine learning

## Introduction
Analyzing data from biomonitoring can quickly become a complex and laborious task, making the development of efficient and automated solutions a priority. In this context, this project seeks to automate the detection and classification of coconut-breaking behavior by capuchin monkeys (*Sapajaus* spp.). Initially, events will be differentiated between coconut breakage and non-breakage, enabling the development of a superior pipeline capable of differentiating coconut species in the future.

## Methodology
The project will utilize a dataset of audio recordings collected in field in conjunction with [NeoPReGo](https://www.neoprego.org/), an organization that supports field research and environmental socioeducation about neotropical primates. The dataset will be preprocessed to extract relevant features, which will then be used to train machine learning models for classification. The performance of the models will be evaluated using standard metrics such as **accuracy**, **precision**, **recall**, **ROC-AUC curves** and **F1-score**.

### Pipeline Overview
1. **Data Collection**: Audio recordings of capuchin monkeys engaging in coconut-breaking behavior
2. **Preprocessing**: Noise reduction, normalization, and segmentation of audio signals
3. **Feature Extraction**: Extraction of features from relevant events using techniques such as wavelet transforms and spectral analysis
4. **Template Matching**: Creation of templates for coconut-breaking events and matching them against the dataset to identify potential occurrences
5. **Model Training**: Training machine learning models (Decision Trees and Neural Networks) on the extracted features and template matches
6. **Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, ROC-AUC curves, and F1-score

### Outcomes
The outcome of this project is a robust and efficient pipeline for the automated analysis of primate behavior based on audio signal processing and machine learning. This pipeline will not only facilitate the analysis of large datasets but also contribute to the understanding of primate behavior and ecology, particularly in relation to their interactions with their environment.  
At this moment, the project is focused on binary classification (coconut breakage vs. non-breakage), but it lays the groundwork for future multi-class classification tasks, such as differentiating between coconut species based on the acoustic signatures of the breakage events.

# Usage and replicability
This repository allows other researchers and practitioners to replicate the study on the processed original dataset, keeping the raw data in charge of NeoPReGo [^1]. That way, the project maintains the integrity and autonomy of the collaborating group while providing a clear and accessible framework for others to build upon and adapt to their own research needs. 

## Installation
This project requires Python 3.12 or higher and uses UV package manager. The guide for UV installation can be found [here](https://docs.astral.sh/uv/getting-started/installation/).  
After installing UV, you can set up the project environment by cloning the repository and installing it's dependencies:

```bash
git clone https://github.com/PedroSchneider1/Goodall.git
cd goodall
uv sync
```

To run the project, simply execute the following command:
```bash
uv run main.py
```

At this moment, some parameters are hardcoded in the `main.py` file, but in the future, a configuration file will be added to allow for more flexible experimentation.
The list of parameters that can be modified includes:
- `TRAIN_RATIO`: the proportion of the dataset used for training the model  
&emsp; &emsp; (**default**: 0.85).
- `TEMPLATES_RATIO`: the proportion of the dataset used for template creation  
&emsp; &emsp; (**default**: 0.08).
- `FEATURE_MODE`: the method used for feature extraction, either 'EACH' (using each feature individually) or 'POOLING' (applying statistical pooling to the features)  
&emsp; &emsp; (**default**: 'POOLING').
- `NUM_SLICES`: the number of slices the audio signal is divided into for feature extraction  
&emsp; &emsp; (**default**: 60).
- `NOISE_FACTOR`: the amount of noise added to the audio signal during data augmentation  
&emsp; &emsp; (**default**: 15).
- `WAVELET`: the type of wavelet used for feature extraction.  
&emsp; &emsp; (**default**: 'cmor1.5-1.0' *[Morlet wavelet with bandwidth=1.5 and center frequency=1.0]*)

In the `settings.py` file, you can also modify the original folders and files paths used.

## Results
Results are automatically saved to a directory following the pattern `run_yyyyMMdd_hhMMss/`. Each run directory will contain the trained model, evaluation metrics, and any generated plots or visualizations. This allows for easy comparison between different runs and configurations [^2].

## Conclusion
Current results show strong performance for binary classification (impact vs. non-impact), reaching near-human precision. Although species-level coconut classification is not yet reliable, the pipeline establishes a solid foundation for future multi-class acoustic event recognition.  
Results are also evaluated under the trade-off between accuracy and computational efficiency: faster models may be preferable for real-time field deployment, while more accurate but slower models remain suitable for offline analysis. The results are presented considering both aspects, allowing for informed decision-making based on the specific requirements of the application.


*Feature extraction time per group*:
<p align="center">
  <div style="width:600px; overflow:hidden; margin:auto;">
    <img src="https://i.imgur.com/rsDhjHX.png" 
         alt="Model Training Time" 
         width="600"
         style="margin-top:-75px;margin-bottom:-55px;">
  </div>
</p>

### Using the 'pooling' method

*Training time per model*:
<p align="center">
    <img src="https://i.imgur.com/CNcviKt.png" alt="Model Training Time" width="600">
</p>

*ROC-Curves*:
<p align="center">
    <img src="https://i.imgur.com/Ye8pj4W.png" alt="ROC-Curves" width="600">
</p>

*Metrics per model*:
<p align="center">
  <div style="width:600px; overflow:hidden; margin:auto;">
    <img src="https://i.imgur.com/SVVD8mo.png" 
         alt="Model Training Time" 
         width="600"
         style="margin-top:-50px;">
      <img src="https://i.imgur.com/yghlIDR.png" alt="Model Training Time" width="600">
  </div>
</p>

### Using the 'each' method

*Training time per model*:
<p align="center">
    <img src="https://i.imgur.com/DptAJbl.png" alt="Model Training Time" width="600">
</p>

*ROC-Curves*:
<p align="center">
    <img src="https://i.imgur.com/LcD7PbM.png" alt="ROC-Curves" width="600">
</p>

*Metrics per model*:
<p align="center">
  <div style="width:600px; overflow:hidden; margin:auto;">
    <img src="https://i.imgur.com/2lVW0Fq.png" 
         alt="Model Training Time" 
         width="600"
         style="margin-top:-50px;">
      <img src="https://i.imgur.com/IgkuLhS.png" alt="Model Training Time" width="600">
  </div>
</p>

## Authors
- **Author**: Pedro Schneider (SCHNEIDER, P.) - [LinkedIn](https://www.linkedin.com/in/pedroschneider1) / [Lattes](http://lattes.cnpq.br/2351530499593103) / [@PedroSchneider1](https://github.com/PedroSchneider1). Undergraduate student in Computer Science at the [FEI](https://portal.fei.edu.br/), Brazil (2024). Research interests include machine learning, signal processing, and their applications in ecology and conservation.
- **Advisor**: Rafael Luiz Testa (TESTA, R. L.) - [Lattes](http://lattes.cnpq.br/9428274086606707) / [@lapidarioz](https://github.com/lapidarioz). Doctor in Information Systems from the University of São Paulo ([USP](https://www5.usp.br/)), Brazil (2024). Research interests include multimedia processing, exploring themes such as facial expression synthesis, deepfake detection and biomonitoring.

## License
Goodall © 2026 by Pedro Schneider is licensed under Creative Commons Attribution-NonCommercial 4.0 International. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/

> There is not a public article yet.

[^1]: The raw data can be accessed by contacting NeoPReGo through their website: https://www.neoprego.org/

[^2]: The results of running the project may vary due to the stochastic nature of machine learning and neural network training. 