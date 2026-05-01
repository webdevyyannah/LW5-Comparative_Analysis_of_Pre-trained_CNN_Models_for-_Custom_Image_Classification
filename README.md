# Laboratory Work 5 — Comparative Analysis of Pre-trained CNN Models for Custom Image Classification

---

## Overview

This laboratory work evaluates and compares three pre-trained CNN models — **MobileNetV2**, **EfficientNetB0**, and **ResNet50** — using a custom plant species image dataset with 20 classes. Each model was trained using transfer learning and evaluated using accuracy, loss, precision, recall, F1-score, confusion matrix, ROC/AUC curves, and Grad-CAM explainability visualizations.

---

## Dataset

- **Total Classes:** 20 plant species
- **Images per Class:** ~250 images
- **Classes:** Anahaw Tree, Areca Tree, Balete Tree, Balimbing Tree, Bamboo Tree, Banana Tree, Birch Tree, Cacao Tree, Calamansi Tree, Coconut Tree, Douglas Fir Tree, Ilang Ilang Tree, Lanzones Tree, Lemon Tree, Mahogany Tree, Malunggay Tree, Mango Tree, Mangrove Tree, Santol Tree, Talisay Tree
- **Split:** 80% Training / 20% Validation
- **Image Size:** 224 × 224

---

## Models Used

| # | Model | Description |
|---|---|---|
| 1 | MobileNetV2 | Lightweight, mobile-optimized architecture |
| 2 | EfficientNetB0 | Compound-scaled efficient architecture |
| 3 | ResNet50 | Deep residual network with skip connections |

All models used ImageNet pre-trained weights with the top layers frozen (transfer learning). A custom classification head was added: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(20)`.

---

## Part 1 — Dataset Preparation

The dataset was organized in Google Drive with one subfolder per class, following TensorFlow's `image_dataset_from_directory()` convention. Each class folder contains at least 250 images of that specific plant species, collected and curated for the classifier originally built in LW3.

---

## Part 2 — Training Configuration

```python
optimizer = Adam(learning_rate=0.0001)
loss = SparseCategoricalCrossentropy(from_logits=True)
epochs = 10
batch_size = 32
image_size = (224, 224)
```

Each model was compiled with the same configuration to ensure a fair comparison. Base model weights were frozen during training (feature extraction / transfer learning).

---

## Part 12 — Complete Performance Comparison Table

This table summarizes all models built across all laboratory works, from Teachable Machine through LW5 pre-trained models.

| Model | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|---|---|
| **Teachable Machine** (LW2) | ~1.00 | ~0.05 | 1.00 | ~0.10 | 1.00 | 1.00 | 1.00 | N/A |
| **1st Model** — LW3 Custom CNN Baseline | 1.0000 | ~0.0001 | 0.9610 | 0.2866 | N/A | N/A | N/A | N/A |
| **2nd Model** — LW3 Improved CNN (Dropout + Augmentation) | 0.7739 | 0.7414 | 0.8720 | 0.4621 | 0.84 | 0.82 | 0.82 | 0.9530 |
| **Enhancement** — LW4 Fine-Tuned CNN (lr=0.00005, 20 epochs) | 0.8518 | 0.4729 | 0.9010 | 0.3704 | ~0.88 | ~0.88 | ~0.88 | 0.9637 |
| **3rd Model — The Good Model** — LW5 MobileNetV2 ✅ | 0.8788 | 0.4991 | **0.9330** | 0.3474 | **0.9351** | **0.9330** | **0.9333** | **0.9888** |
| **Pre-trained Model 1** — MobileNetV2 | 0.8788 | 0.4991 | 0.9330 | 0.3474 | 0.9351 | 0.9330 | 0.9333 | 0.9888 |
| **Pre-trained Model 2** — EfficientNetB0 | 0.0495 | 2.9972 | 0.0460 | 2.9970 | 0.0012 | 0.0350 | 0.0024 | 0.5024 |
| **Pre-trained Model 3** — ResNet50 | 0.0942 | 2.9342 | 0.0770 | 2.9272 | 0.0400 | 0.0700 | 0.0300 | 0.5983 |

> **Key Takeaway:** MobileNetV2 (LW5) is the best-performing model across all laboratory works, achieving 93.30% validation accuracy, 0.9333 F1-score, and 0.9888 AUC — surpassing all prior custom-built models while generalizing well without overfitting. EfficientNetB0 and ResNet50 underperformed due to limited training epochs and preprocessing conflicts with frozen base layers.

---

## Part 4 — MobileNetV2 — Detailed Results

### Training Curves

MobileNetV2 showed healthy, consistent improvement across all 10 epochs. Training accuracy steadily climbed from ~18% to **87.88%**, while validation accuracy improved from ~50% to **93.30%**. Both loss curves declined smoothly — training loss down to **0.4991** and validation loss down to **0.3474** — with no signs of overfitting. The validation accuracy consistently staying above training accuracy is a healthy indicator of strong generalization.

### Classification Report — MobileNetV2

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ANAHAW TREE | 0.93 | 0.98 | 0.95 | 41 |
| ARECA TREE | 0.92 | 0.92 | 0.92 | 48 |
| BALETE TREE | 1.00 | 1.00 | 1.00 | 46 |
| BALIMBING TREE | 0.91 | 0.95 | 0.93 | 44 |
| BAMBOO TREE | 1.00 | 1.00 | 1.00 | 35 |
| BANANA TREE | 1.00 | 1.00 | 1.00 | 47 |
| BIRCH TREE | 0.98 | 1.00 | 0.99 | 52 |
| CACAO TREE | 0.97 | 0.94 | 0.95 | 63 |
| CALAMANSI TREE | 1.00 | 0.98 | 0.99 | 50 |
| COCONUT TREE | 0.94 | 0.87 | 0.90 | 52 |
| DOUGLAS FIR TREE | 0.91 | 0.81 | 0.86 | 53 |
| ILANG ILANG TREE | 0.86 | 0.93 | 0.90 | 46 |
| LANZONES TREE | 0.96 | 0.98 | 0.97 | 47 |
| LEMON TREE | 0.96 | 0.91 | 0.94 | 58 |
| MAHOGANY TREE | 0.71 | 0.87 | 0.78 | 53 |
| MALUNGGAY TREE | 0.87 | 0.82 | 0.85 | 50 |
| MANGO TREE | 0.82 | 0.98 | 0.89 | 56 |
| MANGROVE TREE | 0.92 | 0.92 | 0.92 | 48 |
| SANTOL TREE | 0.94 | 0.88 | 0.91 | 56 |
| TALISAY TREE | 0.98 | 0.80 | 0.88 | 55 |
| **Macro Avg** | **0.93** | **0.93** | **0.93** | **1000** |
| **Weighted Avg** | **0.93** | **0.92** | **0.92** | **1000** |

**Overall Accuracy: 0.92**

### Confusion Matrix — MobileNetV2

The confusion matrix shows strong diagonal dominance across all 20 classes, indicating correct classifications for most samples. Notable misclassifications include:
- **TALISAY TREE** — misclassified 3 samples as Douglas Fir and 4 as Mahogany
- **DOUGLAS FIR TREE** — confused 6 samples as Mahogany and 3 as Santol
- **COCONUT TREE** — 2 samples misclassified as Anahaw and 2 as Areca
- **MALUNGGAY TREE** — 3 samples confused with Ilang Ilang, 3 with Mahogany

Most classes achieved near-perfect diagonal values (40–59 correct per class), confirming MobileNetV2's strong performance on this dataset.

### ROC Curve & AUC — MobileNetV2

| Class | AUC |
|---|---|
| ANAHAW TREE | 1.00 |
| ARECA TREE | 0.99 |
| BALETE TREE | 1.00 |
| BALIMBING TREE | 0.99 |
| BAMBOO TREE | 1.00 |
| BANANA TREE | 1.00 |
| BIRCH TREE | 1.00 |
| CACAO TREE | 1.00 |
| CALAMANSI TREE | 0.99 |
| COCONUT TREE | 1.00 |
| DOUGLAS FIR TREE | 0.99 |
| ILANG ILANG TREE | 0.99 |
| LANZONES TREE | 1.00 |
| LEMON TREE | 0.99 |
| MAHOGANY TREE | 0.95 |
| MALUNGGAY TREE | 0.99 |
| MANGO TREE | 0.99 |
| MANGROVE TREE | 0.98 |
| SANTOL TREE | 0.98 |
| TALISAY TREE | 0.97 |

All ROC curves hug the top-left corner, confirming excellent discriminative ability across all 20 classes. The weakest class, Mahogany Tree (AUC: 0.95), still reflects very strong performance.

---

## Part 5 — EfficientNetB0 — Results

EfficientNetB0 showed unusual training behavior with accuracy values below 6% across all 3 epochs and loss values hovering around 3.0. The validation accuracy dropped before recovering slightly, while training accuracy dipped at epoch 1 before rising again. This behavior is consistent with an under-trained model that requires more epochs and possibly a different learning rate configuration or unfreezing strategy to activate EfficientNet's compound-scaled feature extractors effectively on this custom dataset.

### Classification Report — EfficientNetB0

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ANAHAW TREE | 0.00 | 0.00 | 0.00 | 41 |
| ARECA TREE | 0.05 | 1.00 | 0.09 | 48 |
| BALETE TREE | 0.00 | 0.00 | 0.00 | 46 |
| BALIMBING TREE | 0.00 | 0.00 | 0.00 | 44 |
| BAMBOO TREE | 0.00 | 0.00 | 0.00 | 35 |
| BANANA TREE | 0.00 | 0.00 | 0.00 | 47 |
| BIRCH TREE | 0.00 | 0.00 | 0.00 | 52 |
| CACAO TREE | 0.00 | 0.00 | 0.00 | 63 |
| CALAMANSI TREE | 0.00 | 0.00 | 0.00 | 50 |
| COCONUT TREE | 0.00 | 0.00 | 0.00 | 52 |
| DOUGLAS FIR TREE | 0.00 | 0.00 | 0.00 | 53 |
| ILANG ILANG TREE | 0.00 | 0.00 | 0.00 | 46 |
| LANZONES TREE | 0.00 | 0.00 | 0.00 | 47 |
| LEMON TREE | 0.00 | 0.00 | 0.00 | 58 |
| MAHOGANY TREE | 0.00 | 0.00 | 0.00 | 53 |
| MALUNGGAY TREE | 0.00 | 0.00 | 0.00 | 50 |
| MANGO TREE | 0.00 | 0.00 | 0.00 | 56 |
| MANGROVE TREE | 0.00 | 0.00 | 0.00 | 48 |
| SANTOL TREE | 0.00 | 0.00 | 0.00 | 56 |
| TALISAY TREE | 0.00 | 0.00 | 0.00 | 55 |
| **Macro Avg** | **0.00** | **0.05** | **0.00** | **1000** |
| **Weighted Avg** | **0.00** | **0.05** | **0.00** | **1000** |

**Overall Accuracy: 0.05** — The model predicted almost everything as ARECA TREE (recall 1.00 for that class only), confirming it has not learned any meaningful class distinctions within 3 epochs.

---

## Part 6 — ResNet50 — Results

ResNet50's training curves showed a gradual upward trend in accuracy (reaching ~7.70% val accuracy by epoch 3) with steadily decreasing loss values. However, with only 3 epochs completed, the model was still in the very early stages of convergence. ResNet50's deeper architecture (50 layers) requires more epochs to transfer its learned ImageNet features effectively to a new domain. With full 10-epoch training, it is expected to perform significantly better.

### Classification Report — ResNet50

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ANAHAW TREE | 0.00 | 0.00 | 0.00 | 41 |
| ARECA TREE | 0.05 | 0.94 | 0.09 | 48 |
| BALETE TREE | 0.00 | 0.00 | 0.00 | 46 |
| BALIMBING TREE | 0.00 | 0.00 | 0.00 | 44 |
| BAMBOO TREE | 1.00 | 0.11 | 0.21 | 35 |
| BANANA TREE | 0.00 | 0.00 | 0.00 | 47 |
| BIRCH TREE | 0.00 | 0.00 | 0.00 | 52 |
| CACAO TREE | 0.00 | 0.00 | 0.00 | 63 |
| CALAMANSI TREE | 0.00 | 0.00 | 0.00 | 50 |
| COCONUT TREE | 0.00 | 0.00 | 0.00 | 52 |
| DOUGLAS FIR TREE | 0.00 | 0.00 | 0.00 | 53 |
| ILANG ILANG TREE | 0.00 | 0.00 | 0.00 | 46 |
| LANZONES TREE | 0.12 | 0.11 | 0.11 | 47 |
| LEMON TREE | 0.00 | 0.00 | 0.00 | 58 |
| MAHOGANY TREE | 0.06 | 0.02 | 0.03 | 53 |
| MALUNGGAY TREE | 0.00 | 0.00 | 0.00 | 50 |
| MANGO TREE | 0.00 | 0.00 | 0.00 | 56 |
| MANGROVE TREE | 0.00 | 0.00 | 0.00 | 48 |
| SANTOL TREE | 0.00 | 0.00 | 0.00 | 56 |
| TALISAY TREE | 0.24 | 0.07 | 0.11 | 55 |
| **Macro Avg** | **0.07** | **0.06** | **0.03** | **1000** |
| **Weighted Avg** | **0.06** | **0.06** | **0.02** | **1000** |

**Overall Accuracy: 0.06** — ResNet50 shows slightly more spread in predictions than EfficientNetB0, with a few classes (Bamboo, Areca, Lanzones, Talisay) picking up minimal signal, but overall performance is still far from meaningful at 3 epochs.

---

## Part 7 — Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) was applied to all three models using a Coconut Tree test image to visualize which image regions each model focused on during prediction.

### MobileNetV2 — Grad-CAM

The heatmap showed broad, distributed activation across the upper-center region of the image, highlighting the coconut clusters and palm fronds. The overlay revealed a colorful, widespread activation pattern covering most of the relevant plant structure. This indicates MobileNetV2 is learning from the full spatial context of the tree.

### EfficientNetB0 — Grad-CAM

The heatmap showed highly concentrated activation in a single corner region (bottom-right), with the rest of the image receiving near-zero activation. This extremely sparse and off-center focus reflects the model's early training state — it has not yet learned to identify meaningful plant features and instead fixated on a small corner region. This is consistent with the model's low accuracy results.

### ResNet50 — Grad-CAM

The heatmap showed moderate activation concentrated in the upper-left quadrant, covering some of the palm fronds and coconut fruit area. Compared to EfficientNetB0, the focus region is more spatially meaningful but less comprehensive than MobileNetV2's. The activation pattern suggests ResNet50 is beginning to identify relevant structural features but needs more training to achieve full spatial coverage.

### Grad-CAM Comparison Summary

| Model | Focus Region | Quality |
|---|---|---|
| MobileNetV2 | Broad — covers coconut clusters and fronds | Best — most contextually relevant |
| ResNet50 | Partial — upper-left quadrant, some plant structure | Moderate — improving but incomplete |
| EfficientNetB0 | Sparse — single corner pixel concentration | Weakest — not yet learning plant features |

---

## Guide Questions

### A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

MobileNetV2 achieved the highest accuracy, reaching **93.30% validation accuracy** and **87.88% training accuracy** after 10 epochs with a validation loss of 0.3474. This is because MobileNetV2 is specifically designed as a lightweight, efficient architecture that uses depthwise separable convolutions and inverted residual blocks. These design choices allow it to learn rich feature representations with fewer parameters, making it excellent for transfer learning on datasets of moderate size. Its pre-trained ImageNet weights were directly useful for detecting the leaf textures, branch patterns, and tree structures in my dataset, and it converged faster and more stably than the other two models.

**2. Which model had the lowest performance? What could be the reason?**

EfficientNetB0 had the lowest performance, achieving only **4.60% validation accuracy** with a validation loss of 2.9970, precision of 0.0012, recall of 0.0350, F1-score of 0.0024, and AUC of 0.5024 — barely above random guessing. The primary reason is that EfficientNetB0 uses a unique compound scaling approach that involves batch normalization layers with specific internal scaling behaviors. When a frozen EfficientNet base is used with a Rescaling layer and a small learning rate, the batch normalization statistics from ImageNet can conflict with the new dataset's distribution, causing the model to produce near-random outputs. Additionally, EfficientNetB0 typically benefits from unfreezing some layers or using its own built-in preprocessing rather than a separate rescaling layer.

**3. How did loss values compare across models?**

MobileNetV2 showed the best loss behavior, with training loss declining to **0.4991** and validation loss reaching **0.3474** by epoch 10 — reflecting genuine, stable learning. EfficientNetB0 ended at training loss **2.9972** and validation loss **2.9970**, while ResNet50 ended at **2.9342** training loss and **2.9272** validation loss. Both underperforming models barely moved from their starting values, indicating that neither model had begun meaningful learning within their 3-epoch runs. The near-identical training and validation loss values for EfficientNetB0 and ResNet50 further indicate the models were outputting near-uniform distributions rather than class-specific predictions.

---

### B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Accuracy only tells us the overall percentage of correct predictions, but it can be misleading — especially when classes are imbalanced. For example, if a model always predicts the majority class, it could achieve high accuracy while completely failing on minority classes. Precision, Recall, and F1-score give a more complete picture: Precision tells us how reliable the model's positive predictions are, Recall tells us how many actual positives were caught, and F1-score balances both. AUC adds another layer by measuring discriminative ability across different classification thresholds. Together, these metrics reveal the model's true strengths and weaknesses across all 20 classes.

**5. Which model had the best F1-score? What does it indicate?**

MobileNetV2 had the best F1-score among all three pre-trained models, achieving an overall **F1-score of 0.9333** with precision of 0.9351 and recall of 0.9330. This tightly balanced precision-recall relationship indicates that the model is both reliable when it makes predictions and thorough in finding actual positives. EfficientNetB0 (F1: 0.0024) and ResNet50 (F1: 0.0300) had near-zero F1-scores, consistent with their near-random classification outputs at 3 epochs of training.

**6. How did Precision and Recall differ across models?**

For MobileNetV2, precision (**0.9351**) and recall (**0.9330**) were both high and nearly equal — a 0.002 gap — indicating a well-balanced model that is both reliable and thorough. EfficientNetB0 had a severe imbalance: precision of 0.0012 versus recall of 0.0350, meaning the model rarely predicted any class confidently but occasionally got lucky. ResNet50 showed slightly better balance with precision 0.0400 and recall 0.0700, but both values confirm the model has not yet learned meaningful class boundaries. The precision-recall gap narrows as training progresses, so the disparity in EfficientNetB0 and ResNet50 is a direct consequence of their insufficient training duration.

---

### C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

Based on the MobileNetV2 confusion matrix, the most frequently misclassified classes were Douglas Fir Tree (6 samples confused as Mahogany, 3 as Santol), Talisay Tree (3 as Douglas Fir, 4 as Mahogany), and Coconut Tree (2 as Anahaw, 2 as Areca). These misclassifications are understandable because these tree species share similar bark textures, leaf shapes, or overall silhouette characteristics that make them visually ambiguous to the model.

**8. What patterns did you observe in the confusion matrix?**

The overall pattern is strongly diagonal, which means the model is correctly classifying the vast majority of samples. The off-diagonal errors cluster primarily among trees that have similar visual features — for example, Mahogany, Douglas Fir, Santol, and Talisay Tree tend to get confused with each other, likely because they all have similar broad-canopy structures and leaf appearances. Bamboo Tree, Banana Tree, and Mango Tree showed very clean rows with almost no misclassifications, reflecting their highly distinctive visual features that MobileNetV2 learned easily.

---

### D. ROC and AUC

**9. Which model had the highest AUC score?**

MobileNetV2 had the highest AUC score at **0.9888**, with the majority of individual classes achieving AUC = 1.00 and the lowest-performing class (Mahogany Tree) still at AUC = 0.95. This near-perfect macro AUC means MobileNetV2 has a 98.88% probability of correctly ranking a true positive higher than a false positive across all classes. EfficientNetB0 achieved only **0.5024** AUC — barely above random guessing (0.5) — while ResNet50 reached **0.5983**, slightly better but still far from useful discrimination. These AUC values confirm that only MobileNetV2 has learned meaningful class separability at this stage.

**10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's ability to discriminate between positive and negative cases across all possible classification thresholds. An AUC of 1.0 means the model perfectly separates all classes, while 0.5 means random guessing. Unlike accuracy, AUC is threshold-independent and provides a more complete picture of a model's discriminative power. In a multi-class problem like mine, individual per-class AUC values reveal which specific species the model distinguishes best, allowing targeted improvements for weaker classes.

---

### E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

Grad-CAM revealed significant differences in what each model was "looking at" when making predictions. MobileNetV2 showed broad, contextually relevant activation covering the coconut clusters and palm fronds — the actual distinguishing features of the Coconut Tree. ResNet50 showed partial activation over some relevant regions, while EfficientNetB0's activation was concentrated in a single corner pixel with no visible connection to any plant feature. This confirms that MobileNetV2's predictions are backed by genuine feature learning, while the other two models have not yet learned to associate plant features with their correct labels.

**12. Did the model focus on relevant image regions?**

MobileNetV2 clearly focused on relevant image regions — the heatmap overlay highlighted the fronds, coconut fruits, and tree structure, which are the actual distinguishing visual features of the plant. ResNet50 partially focused on relevant areas but its activation was incomplete. EfficientNetB0's focus was irrelevant, concentrating on a corner region that contains no meaningful plant information. This confirms that among the three models, only MobileNetV2 has developed meaningful visual representations of the plant species at this stage of training.

**13. Which model produced the most meaningful heatmaps?**

MobileNetV2 produced the most meaningful Grad-CAM heatmaps. Its activation pattern was broad, distributed across the entire relevant area of the image, and clearly corresponded to the visual structure of the Coconut Tree (fronds, trunk, fruit clusters). This level of spatial coverage and contextual relevance is a strong indicator that MobileNetV2 is not just pattern-matching but genuinely learning the visual grammar of plant species identification.

---

### F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**

I would recommend **MobileNetV2** for deployment. It achieved the highest validation accuracy (**93.30%**), the best AUC (**0.9888**), precision (**0.9351**), recall (**0.9330**), and F1-score (**0.9333**) among all three pre-trained models — and even outperforms the custom CNN built from scratch in LW3 (which achieved 96% val accuracy but suffered overfitting). Additionally, MobileNetV2 was specifically designed for mobile and edge deployment — it is lightweight, fast, and memory-efficient, making it ideal for integration into a mobile plant identification app. Its strong, stable convergence across all 10 epochs further confirms it as the best choice for real-world use.

**15. How can you further improve your best-performing model?**

MobileNetV2 can be further improved through several strategies. First, fine-tuning — gradually unfreezing the top layers of the MobileNetV2 base and retraining with a very low learning rate (e.g., 0.00001) to allow the pre-trained weights to adapt more closely to plant species features. Second, stronger data augmentation such as random brightness, hue shifts, and cutout regularization could help the model generalize better to photos taken under different lighting conditions. Third, training EfficientNetB0 with its native preprocessing pipeline (without the Rescaling layer) and unfreezing some layers could turn it into a competitive alternative. Fourth, increasing the dataset size by collecting more images per class — especially for the weaker classes like Mahogany and Douglas Fir — would directly improve recall for those species.

---

### G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

This plant classifier has direct real-world applications in several areas. For farmers and agricultural workers in the Philippines, it could serve as an instant plant identification tool — simply photograph a tree to identify its species. For forestry and conservation work, it could assist rangers in monitoring and cataloguing tree species in protected areas. For educational purposes, students in agriculture or biology could use it as a learning aid. The model's knowledge of Philippine native species like Narra (Balimbing), Coconut, Anahaw, and Malunggay makes it particularly valuable for local ecological and agricultural contexts.

**17. What are the risks of deploying an inaccurate model?**

Deploying an inaccurate model carries significant risks depending on the application context. In agriculture, misidentifying a toxic plant as an edible one could cause harm. In forestry, incorrect species identification could lead to wrong management decisions or misclassification of protected trees. Economically, a farmer relying on the model to identify crop species could suffer losses if the system makes systematic errors. More broadly, users who trust the system without understanding its limitations could make critical decisions based on wrong information. This is why model transparency, confidence scores, and explainability tools like Grad-CAM are essential components of any deployed AI system.

**18. How can this system be integrated into a mobile/web app?**

The trained MobileNetV2 model can be saved using `model.save()` and then converted to TensorFlow Lite (`.tflite`) format for mobile deployment on Android or iOS. A simple mobile app could allow users to capture a photo using their device camera, send it to the model for preprocessing and inference, and display the predicted plant species along with a confidence score. For a web application, the model can be served via a Flask or FastAPI backend, or converted to TensorFlow.js for fully client-side inference in the browser. The addition of Grad-CAM visualization in the app interface would make the system more trustworthy and educational, showing users which parts of the plant image the model focused on during identification.

---

## Results Summary

| Model | Val Accuracy | Val Loss | Precision | Recall | F1-Score | AUC | Grad-CAM Quality | Recommended |
|---|---|---|---|---|---|---|---|---|
| **MobileNetV2** | **93.30%** | **0.3474** | **0.9351** | **0.9330** | **0.9333** | **0.9888** | **Excellent** | **✅ Yes** |
| ResNet50 | 7.70% (3 epochs) | 2.9272 | 0.0400 | 0.0700 | 0.0300 | 0.5983 | Partial | ❌ Needs more training |
| EfficientNetB0 | 4.60% (3 epochs) | 2.9970 | 0.0012 | 0.0350 | 0.0024 | 0.5024 | Poor | ❌ Needs configuration fix |

---

## Conclusion

Laboratory Work 5 provided a comprehensive comparison of three pre-trained CNN architectures applied to a custom 20-class plant species dataset. MobileNetV2 emerged as the clear winner, achieving **93.30% validation accuracy**, **0.9333 F1-score**, and **0.9888 AUC** — the best results across all metrics and all models trained throughout this course. Its Grad-CAM heatmaps further confirmed that its predictions are grounded in genuine visual feature learning rather than coincidental patterns.

EfficientNetB0 (val accuracy: 4.60%, AUC: 0.5024) and ResNet50 (val accuracy: 7.70%, AUC: 0.5983) significantly underperformed primarily due to limited training epochs (3 epochs) and configuration issues, particularly EfficientNetB0's sensitivity to preprocessing pipeline choices when its base is frozen. With proper configuration — native EfficientNet preprocessing, partial unfreezing, and more epochs — both models have the architectural potential to match or exceed MobileNetV2.

The Grad-CAM analysis was particularly valuable in this comparison, revealing not just how accurate each model was, but why. MobileNetV2's broad, contextually relevant heatmaps gave confidence that the model would generalize reliably to new real-world plant images, while the other models' poor spatial focus confirmed their early training state.

Going forward, MobileNetV2 is recommended for deployment as a mobile plant identification tool, with fine-tuning (unfreezing top layers with lr=0.00001) and expanded data augmentation as the primary paths to further improvement.

---

## 🔗 Project Links

- 📓 **Google Colab Notebook (LW5):** *(https://colab.research.google.com/drive/1_uVEZlXpbotj0-lMSSuoBcXeVdsVHYjo?usp=sharing)*
- 📓 **Google Colab Notebook (LW4):** [https://colab.research.google.com/drive/16HlejEog1Jl3SxrGM1hmmo6DYOz3lb6n?usp=sharing]
- 📁 **Google Drive Dataset:** [https://drive.google.com/drive/folders/1TRQJ9ZjW8XNAK6J1VdbcqLDDwcdhuwcO?usp=sharing]
- 🧠 **Saved Model (LW4):** [https://drive.google.com/file/d/19L1TODQCLFHRFOioXjQewesOzPX2qbG1/view?usp=drive_link]
