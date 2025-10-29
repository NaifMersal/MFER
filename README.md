# A Multimodal Framework for Emotion Recognition (MFER)

This repository contains the code for our research project on multimodal emotion recognition, combining audio and visual cues to predict human emotions. The framework leverages pre-trained models like Emotion2Vec for audio feature extraction and Vision Transformer (ViT) for visual feature extraction, fused together for improved accuracy.

## 📌 Abstract

Emotion recognition is critical in many fields, from customer services to human-computer interaction. This study proposes a multimodal framework that integrates images and audio using pre-trained models specifically designed for emotion recognition tasks. Our approach captures and combines features across modalities.

On the benchmark RAVDESS dataset, our method showed robust and effective performance, achieving results comparable to the state-of-the-art. Although the average accuracy did not exceed the state-of-the-art benchmark, our model produced competitive results and, by one fold, even exceeded the reported accuracy, highlighting its potential in specific scenarios. However, an attempt to validate the model on a custom-created Arabic dataset was unsuccessful, highlighting areas for further improvement.

> **Note on Arabic Results**: The suboptimal performance on our custom Arabic dataset is likely due to the limited quality and diversity of the recorded actors' performances — in short, the bad results are probably from the bad actors :)

## 🧩 Methodology

Our system follows a three-stage progressive training strategy:
1.  **Visual-Only Stage**: Train the ViT-based image arm on RAVDESS.
2.  **Audio-Only Stage**: First train the Emotion2Vec + GRU audio arm on a multi-language dataset (CREMA-D, ShEMO, SDC, URDU), then fine-tune it for 8 classes on RAVDESS.
3.  **Combined Modality Stage**: Fuse the trained audio and visual features and fine-tune the entire multimodal network on RAVDESS.

The architecture processes video input by extracting frames (for the image arm) and audio waves (for the audio arm). Features from both arms are concatenated and passed through a final classifier.

## 📊 Key Results

*   **RAVDESS Dataset (8 classes)**:
    *   Visual-Only Accuracy: ~88.7%
    *   Audio-Only Accuracy: ~89.2%
    *   **Multimodal Accuracy: 96.94%** (comparable to SoTA)
    *   *One fold achieved 98.6% accuracy.*

*   **Custom Arabic Dataset**:
    *   Performance was unsatisfactory across all modalities (Visual: ~18.8%, Audio: ~34.2%, Combined: ~26.1%), indicating challenges with generalization to new, culturally specific datasets and potentially lower data quality.

## 🗃️ Datasets Used

*   **Audio Training**: CREMA-D, Saudi Dialect Corpus (SDC), ShEMO, URDU.
*   **Image Training**: FER-2013 (for ViT pre-training).
*   **Multimodal Evaluation**: RAVDESS (benchmark).
*   **Custom Test Set**: Self-collected Arabic audio-visual dataset (8 classes).


## 📚 References

This work builds upon several key papers see the full [References](MFER.pdf) section in the original report for details.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
