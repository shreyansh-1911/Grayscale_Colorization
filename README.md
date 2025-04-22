# 🎨 Image Colorization using U-Net Architecture (Deep Learning)

This project focuses on colorizing grayscale images using a U-Net-based convolutional neural network (CNN). The model predicts pixel-level color information by classifying each pixel into one of 313 quantized color bins in the **ab color space**. The network is trained on grayscale and corresponding color images, with a custom data generator, preprocessing pipeline, and class-rebalanced loss.

---

## 📌 Objective

To develop a deep learning model that can automatically colorize black & white images using semantic understanding, by converting grayscale inputs into color outputs using pixel-wise classification into color bins.

---

## 🧠 Methodology

1. **Data Preparation**  
   - Dataset contains grayscale (`L` channel) and corresponding color images (`ab` channels).
   - The color space is converted from RGB to **Lab** color space for better perceptual representation.
   - Each `ab` value of every pixel is quantized into one of **313 color bins** using a precomputed color dictionary.
   - Created a custom data generator for batch-wise loading and preprocessing.

2. **Model Architecture**  
   - Used a modified **U-Net** architecture for pixel-wise classification.
   - Input: Single-channel grayscale (`L`) image of shape (224, 224, 1).
   - Output: Softmax probabilities over 313 bins for each pixel → shape: (224, 224, 313).

3. **Loss Function**  
   - Used **Categorical Cross-Entropy Loss** between predicted probabilities and one-hot encoded color bin labels.
   - Applied **class rebalancing** using a precomputed class histogram to give rare colors more importance.

4. **Training**  
   - Trained using batches of (grayscale input, one-hot bin labels).
   - Optimizer: Adam  
   - Evaluation: Visual inspection of colorized results on test images.

---

## 🏗 Model Architecture (U-Net)

- **Encoder:**  
  - 4 downsampling blocks (Conv + ReLU + MaxPooling)  
- **Bottleneck:**  
  - Two convolutional layers
- **Decoder:**  
  - 4 upsampling blocks (ConvTranspose + skip connections from encoder)
- **Final Layer:**  
  - 1x1 Conv → Softmax over 313 classes (per pixel)

<img src="https://raw.githubusercontent.com/zhixuhao/unet/master/img/u-net-architecture.png" width="500" />

---

## 📊 Results

- ✅ Successfully learned colorization of basic objects (e.g., red, yellow tones).
- ⚠️ Model struggled with distinguishing **green and blue** shades due to insufficient training diversity or class imbalance.
- 🔧 Deployed an interactive **image colorization app** for testing trained model outputs on custom images.
