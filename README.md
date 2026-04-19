# Laboratory Work 5: Comparative Analysis of Pre-trained CNN Models for Custom Image Classification

## Complete Performance Comparison Table

| Model | Train Accuracy | Train Loss | Test/Val Accuracy | Test/Val Loss | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|---|---|
| **Teachable Machine** | ~1.00 | ~0.05 | 1.00 | ~0.10 | 1.00 | 1.00 | 1.00 | N/A |
| **1st Model** (LW1 - Fashion MNIST) | 0.9190 | 0.2200 | 0.8894 | 0.3780 | N/A | N/A | N/A | N/A |
| **2nd Model** (LW3 - Custom CNN Baseline) | 0.9992 | 0.0072 | 0.9600 | 0.2420 | 0.84 | 0.82 | 0.82 | 0.9530 |
| **Enhancement** (LW4 - Improved CNN) | 0.4260 | 1.8752 | 0.4700 | 1.7326 | N/A | N/A | N/A | N/A |
| **3rd Model - The Good Model** (MobileNetV2) | 0.8788 | 0.4991 | 0.9330 | 0.3474 | 0.9351 | 0.9330 | 0.9333 | 0.9888 |
| **Pre-trained Model 1** (MobileNetV2) | 0.8788 | 0.4991 | 0.9330 | 0.3474 | 0.9351 | 0.9330 | 0.9333 | 0.9888 |
| **Pre-trained Model 2** (EfficientNetB0) | 0.0495 | 2.9972 | 0.0460 | 2.9970 | 0.0012 | 0.0350 | 0.0024 | 0.5024 |
| **Pre-trained Model 3** (ResNet50) | 0.0942 | 2.9342 | 0.0770 | 2.9272 | 0.0400 | 0.0700 | 0.0300 | 0.5983 |

---

### Notes on Each Model

**Teachable Machine (LW2-A)**
- Platform: Google Teachable Machine
- Dataset: 20 tree species, 250 images per class (5,000 total)
- Parameters: 80 epochs, batch size 32, learning rate 0.00115
- Result: 100% accuracy on all 20 classes with zero misclassifications
- Note: Teachable Machine uses MobileNet internally as its base architecture
  with ImageNet pre-trained weights, which explains its perfect performance
  on a well-curated dataset.

**1st Model (LW1 - Fashion MNIST)**
- Architecture: Simple Dense Neural Network
  (Flatten → Dense 256 → Dense 64 → Dense 10)
- Dataset: Fashion MNIST (10 classes, 60,000 training / 10,000 test images)
- Parameters: 22 epochs, Adam optimizer
- Result: Train Accuracy 0.9190, Test Accuracy 0.8894, Test Loss 0.3780
- Note: This was a basic fully-connected neural network with no
  convolutional layers, trained on grayscale 28x28 images. The absence
  of CNN layers limited its feature extraction capability, explaining the
  lower accuracy compared to CNN-based models.

**2nd Model (LW3 - Custom CNN Baseline)**
- Architecture: Custom CNN
  (Rescaling → Conv2D 16 → MaxPool → Conv2D 32 → MaxPool →
  Conv2D 64 → MaxPool → Flatten → Dense 128 → Dense 20)
- Dataset: 20 tree species, 5,002 images (80/20 split)
- Parameters: 10 epochs, Adam optimizer
- Result: Train Accuracy 0.9992, Val Accuracy 0.9600, Val Loss 0.2420
- Note: This model showed signs of overfitting (train accuracy 99.92% vs
  val accuracy 96%), but still achieved strong validation performance.
  The LW4 evaluation further revealed per-class metrics:
  Precision 0.84, Recall 0.82, F1 0.82, AUC 0.9530.

**Enhancement (LW4 Activity 3 - Improved CNN)**
- Architecture: Improved CNN with Data Augmentation, Batch Normalization,
  Dropout (0.4 + 0.5), Dense 256, Early Stopping
- Dataset: 20 tree species, 5,002 images (80/20 split)
- Parameters: 20 epochs, Adam lr=0.0001, Early Stopping patience=3
- Result: Train Accuracy 0.4260, Val Accuracy 0.4700, Val Loss 1.7326
- Note: Despite lower accuracy numbers compared to the baseline, this
  model showed healthy generalization behavior — validation accuracy
  consistently exceeded training accuracy, indicating no overfitting.
  The lower accuracy is due to the conservative learning rate and the
  model needing more epochs to converge with the more complex architecture.

**3rd Model - The Good Model (MobileNetV2 - LW5)**
- Architecture: MobileNetV2 (frozen) + GlobalAveragePooling2D +
  Dense 128 (relu) + Dropout 0.5 + Dense 20
- Dataset: 20 tree species, 5,002 images (80/20 split)
- Parameters: 10 epochs, Adam lr=0.0001, Early Stopping patience=3
- Result: Train Accuracy 0.8788, Val Accuracy 0.9330,
  Precision 0.9351, Recall 0.9330, F1 0.9333, AUC 0.9888
- Note: MobileNetV2 is designated as "The Good Model" because it achieved
  the best balance of accuracy, generalization, and efficiency among all
  models tested across all laboratory works. Its ImageNet pre-trained
  weights provided strong feature representations that transferred
  effectively to the 20-class tree classification task.

---

## Guide Questions and Answers

---

### A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

MobileNetV2 achieved the highest accuracy among the three models, reaching a validation
accuracy of 0.9330 and a training accuracy of 0.8788 after 10 epochs. This superior
performance can be attributed to MobileNetV2's architecture, which uses depthwise
separable convolutions and inverted residuals with linear bottlenecks — a design that is
highly efficient for feature extraction on custom image datasets. The ImageNet pre-trained
weights of MobileNetV2 transferred well to the 20-class tree dataset, allowing the frozen
base to provide strong feature representations that the new classification head could learn
from effectively. Additionally, MobileNetV2's relatively lightweight architecture made it
less prone to convergence issues under the transfer learning setup used in this experiment.

---

**2. Which model had the lowest performance? What could be the reason?**

EfficientNetB0 had the lowest performance, with a validation accuracy of only 0.0460 and
an AUC score of 0.5024 — barely above random chance. This poor performance is likely due
to EfficientNetB0's built-in rescaling behavior. EfficientNet models expect inputs to be
preprocessed using their own internal preprocessing pipeline, and adding an additional
Rescaling(1./255) layer on top of the model causes a conflict where pixel values become
incorrectly scaled, preventing the model from learning meaningful features. This is a known
compatibility issue with EfficientNet models in Keras when combined with manual rescaling
layers. ResNet50 similarly struggled for comparable reasons, as it also benefits from its
own specific preprocessing that was not applied in this setup.

---

**3. How did loss values compare across models?**

The loss values varied dramatically across the three models. MobileNetV2 showed a steady
and consistent decrease in both training loss (from 2.8283 in Epoch 1 to 0.4991 in Epoch
10) and validation loss (from 2.1252 to 0.3474), indicating healthy and stable convergence.
In contrast, EfficientNetB0 maintained a nearly flat loss throughout all 3 epochs before
early stopping triggered, with training loss remaining at approximately 2.9972 and
validation loss at 2.9970 — showing no meaningful learning. ResNet50 also showed minimal
improvement, with training loss decreasing only slightly from 3.1917 to 2.9342 across 3
epochs before early stopping halted training, and validation loss barely changing. The stark
contrast in loss trajectories highlights how critical proper preprocessing is when using
pre-trained models with transfer learning.

---

### B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Accuracy alone is not sufficient to evaluate a model because it does not reveal how the
model performs on individual classes, particularly in multi-class classification scenarios.
A model can achieve misleadingly high accuracy by simply predicting the majority class
repeatedly while completely failing on minority classes. For example, EfficientNetB0
achieved an accuracy of 0.0460 and predicted almost everything as BAMBOO TREE (recall
of 1.00 for that class), but had zero performance on all other 19 classes. Metrics like
Precision, Recall, F1-score, and AUC provide a more complete picture by measuring
per-class performance, false positive rates, false negative rates, and the model's overall
discriminative ability across all possible classification thresholds.

---

**5. Which model had the best F1-score? What does it indicate?**

MobileNetV2 achieved the best F1-score of 0.9333 (weighted average), with individual
class F1-scores ranging from 0.82 for MAHOGANY TREE to 1.00 for BAMBOO TREE. This
high F1-score indicates that MobileNetV2 maintained a strong balance between Precision
and Recall across all 20 tree classes — it was not only correctly identifying most of the
true positives but also avoiding a high number of false positives and false negatives. A
weighted F1-score of 0.9333 on a 20-class custom dataset is an excellent result that
demonstrates the effectiveness of MobileNetV2's ImageNet pre-trained features for
distinguishing between visually similar tree species.

---

**6. How did Precision and Recall differ across models?**

For MobileNetV2, Precision and Recall were consistently high and well-balanced across
all 20 classes, with weighted Precision at 0.9351 and weighted Recall at 0.9330. The
closest gap between Precision and Recall was observed in classes like BAMBOO TREE
(both 1.00) and BALETE TREE (Precision: 0.96, Recall: 1.00). The most imbalanced class
was MAHOGANY TREE with Precision of 0.79 and Recall of 0.85, suggesting some
misclassification in both directions for that class. For EfficientNetB0, both Precision
and Recall were effectively 0.00 across all classes except BAMBOO TREE, which had a
Recall of 1.00 but a Precision of only 0.04 — meaning the model predicted BAMBOO TREE
for nearly every sample but happened to be right only 4% of the time. ResNet50 showed
a similar pattern of near-zero Precision and Recall across most classes, with slight
activity in BAMBOO TREE, BIRCH TREE, MAHOGANY TREE, and SANTOL TREE.

---

### C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

Based on the evaluation results, the classes that experienced the most misclassification
in the MobileNetV2 model were MAHOGANY TREE (Precision: 0.79, Recall: 0.85,
F1: 0.82), SANTOL TREE (F1: 0.88), MALUNGGAY TREE (F1: 0.88), and ILANG ILANG
TREE (F1: 0.90). These classes likely share visual similarities in leaf shape, bark
texture, or overall tree structure that make them difficult to distinguish even for a
well-trained model. For EfficientNetB0 and ResNet50, virtually all classes were
misclassified as the models failed to learn discriminative features, with most predictions
collapsed into a single class.

---

**8. What patterns did you observe in the confusion matrix?**

In the MobileNetV2 confusion matrix, the dominant pattern was strong diagonal
activation, indicating that most predictions were correct. The off-diagonal
misclassifications were sparse and scattered, with no single class pair showing
consistently high confusion. The classes with the weakest diagonal values were
MAHOGANY TREE, MALUNGGAY TREE, and SANTOL TREE, confirming their lower
F1-scores. In contrast, the confusion matrices for EfficientNetB0 and ResNet50 showed
highly abnormal patterns where nearly all predictions were concentrated in one or two
columns (BAMBOO TREE for EfficientNetB0, and BAMBOO TREE, BIRCH TREE, and
SANTOL TREE for ResNet50), with the true diagonal being almost entirely empty. This
pattern clearly indicates that these two models failed to learn meaningful class
distinctions and were essentially guessing based on class bias.

---

### D. ROC and AUC

**9. Which model had the highest AUC score?**

MobileNetV2 achieved the highest overall AUC score of 0.9888, indicating excellent
discriminative ability across all 20 tree classes. EfficientNetB0 had an AUC of 0.5024
and ResNet50 had an AUC of 0.5983, both of which are close to 0.5 — the score expected
from a random classifier — confirming that these two models effectively failed to learn
meaningful class boundaries on the given dataset.

---

**10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's ability to correctly rank a positive
sample higher than a negative sample across all possible classification thresholds,
regardless of the specific threshold chosen. An AUC of 1.0 represents a perfect
classifier, while an AUC of 0.5 represents a random classifier. MobileNetV2's AUC of
0.9888 means it can correctly distinguish between the correct tree class and all other
classes 98.88% of the time, which is a very strong result for a 20-class problem. AUC
is particularly valuable in multi-class problems because it evaluates the model's
discriminative power independently of class imbalance, making it a more reliable metric
than accuracy when classes have unequal sample sizes.

---

### E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

Grad-CAM revealed the specific spatial regions of the input image that each model
relied upon when making its classification decision. By computing gradients of the
predicted class score with respect to the final convolutional layer outputs
(Conv_1 for MobileNetV2, top_conv for EfficientNetB0, and conv5_block3_3_conv for
ResNet50), Grad-CAM produced heatmaps that highlighted which parts of the coconut
tree image were most influential in driving each model's prediction. For MobileNetV2,
the heatmap activation reflected the model's ability to identify relevant structural
features of the tree, which aligns with its high classification accuracy. For
EfficientNetB0 and ResNet50, the Grad-CAM outputs were less meaningful given that
these models failed to converge properly.

---

**12. Did the model focus on relevant image regions?**

For MobileNetV2, the Grad-CAM overlay on the coconut tree image showed activation
distributed across the tree's visual structure, including the frond and trunk regions,
suggesting that the model was attending to visually meaningful parts of the image for
its classification. While the heatmap was not perfectly localized to a single feature,
the distribution of activation over tree-relevant regions is consistent with the model's
high classification performance. For EfficientNetB0 and ResNet50, the Grad-CAM
heatmaps were not reliable indicators of meaningful decision regions since both models
essentially failed to learn class-discriminative features, and their activations reflect
random or biased feature responses rather than genuine visual understanding.

---

**13. Which model produced the most meaningful heatmaps?**

MobileNetV2 produced the most meaningful and interpretable Grad-CAM heatmaps among
the three models. Since MobileNetV2 successfully learned to classify the 20 tree classes
with 93.3% validation accuracy, its Grad-CAM activations on the coconut tree image
reflected genuine learned features relevant to the classification task. The overlay
showed visible activation patterns over the structural elements of the tree rather than
uniform or scattered activations. EfficientNetB0 and ResNet50, having failed to
converge, produced heatmaps that lacked meaningful spatial interpretation since their
feature extractors did not develop class-discriminative representations for this dataset.

---

### F. Model Comparison and Improvement

**14. Which model would you recommend for deployment? Why?**

MobileNetV2 is the clear recommendation for deployment. It achieved the highest
validation accuracy of 93.30%, the best F1-score of 0.9333, and the highest AUC score
of 0.9888 among the three models tested. Beyond its performance metrics, MobileNetV2
is specifically designed for efficiency on resource-constrained environments such as
mobile devices and edge computing platforms. Its depthwise separable convolution
architecture results in significantly fewer parameters and faster inference times
compared to heavier models, making it practical for real-world deployment in mobile or
web applications for tree species identification. The combination of high accuracy,
strong generalization, and computational efficiency makes MobileNetV2 the optimal
choice for this 20-class tree classification task.

---

**15. How can you further improve your best-performing model?**

MobileNetV2's performance can be further improved through several strategies. First,
fine-tuning can be applied by unfreezing the top layers of the MobileNetV2 base and
training them with a very low learning rate (e.g., 1e-5) to allow the pre-trained
features to adapt more specifically to the tree dataset. Second, expanding the dataset
by adding more images per class, particularly for underperforming classes like
MAHOGANY TREE and MALUNGGAY TREE, would improve class-level performance.
Third, more aggressive data augmentation techniques such as random brightness
adjustment, cutout augmentation, and mixup could improve generalization further.
Fourth, experimenting with a larger classification head with additional Dense layers
and Batch Normalization could improve the model's ability to separate the 20 classes.
Finally, increasing the number of training epochs with a learning rate scheduler that
reduces the learning rate on plateau could help the model converge to a better minimum.

---

### G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

The MobileNetV2-based tree classifier can be applied in several real-world scenarios.
In forestry and environmental management, the model can be deployed as a mobile
application that allows rangers, researchers, and citizens to identify tree species by
simply taking a photo, supporting biodiversity monitoring and illegal logging detection.
In agriculture and agroforestry, farmers can use the system to identify trees in mixed
farming environments for crop management and yield planning. In education, the
application can serve as an interactive learning tool for students studying botany,
ecology, or environmental science. In urban planning, the system can assist city
planners in cataloging and monitoring urban tree populations. The model's lightweight
architecture makes it particularly suitable for deployment in remote or off-grid areas
where internet connectivity is limited, as it can run inference directly on mobile devices
without requiring a server connection.

---

**17. What are the risks of deploying an inaccurate model?**

Deploying an inaccurate tree classification model carries several significant risks. In
conservation and forestry, misidentifying a protected or endangered tree species could
lead to illegal logging going undetected, contributing to deforestation and biodiversity
loss. In agriculture, incorrectly classifying a tree species could lead to wrong
management decisions, inappropriate use of pesticides or fertilizers, and reduced crop
yields. From a legal and regulatory perspective, misclassification could result in
incorrect documentation for environmental impact assessments or compliance reports,
potentially exposing organizations to legal liability. In public safety, misidentifying
invasive tree species could allow them to spread unchecked, threatening native
ecosystems. Additionally, deploying a model with known biases — such as the poor
performance on MAHOGANY TREE and MALUNGGAY TREE observed in this study —
without disclosing those limitations could erode user trust and lead to decisions made
on false confidence.

---

**18. How can this system be integrated into a mobile or web app?**

The MobileNetV2 model saved in Keras format can be integrated into a mobile or web
application through several approaches. For mobile deployment, the model can be
converted to TensorFlow Lite (TFLite) format using TensorFlow's conversion tools,
reducing its size and enabling on-device inference on Android and iOS platforms without
requiring internet connectivity. The TFLite model can then be embedded into a native
Android app using Java/Kotlin or an iOS app using Swift, with a camera interface that
captures images and passes them through the model for real-time classification. For
web deployment, the model can be converted to TensorFlow.js format and hosted as a
Progressive Web App (PWA), allowing users to classify tree species directly in their
browser without installing any software. Alternatively, the Keras model can be deployed
as a REST API using frameworks such as FastAPI or Flask, hosted on cloud platforms
like Google Cloud Run or AWS Lambda, and consumed by a React or Flutter front-end
application that sends images and receives predictions in JSON format.
