


#### 🌱 **Plant Species Classification using CNN**  

![Plants](https://github.com/Arif-miad/Plant-Species-Classification-using-CNN/blob/main/ezgif-3f8cceb92a8cf.jpg)  

---

## 📖 **Overview**  
This project focuses on **classifying plant species** using a **Convolutional Neural Network (CNN)**. The dataset consists of synthetic plant images categorized into five classes:  

- 🌵 **Cactus**  
- 🌿 **Fern**  
- 🌹 **Rose**  
- 🌻 **Sunflower**  
- 🌷 **Tulip**  

We use **TensorFlow & Keras** to train a deep learning model for **image classification**. The final trained model is evaluated, saved, and can be deployed for real-time predictions.  

---

## 📂 **Dataset Structure**  

```
plants-classification/
│── dataset/
│   ├── train/
│   │   ├── cactus/
│   │   ├── fern/
│   │   ├── rose/
│   │   ├── sunflower/
│   │   ├── tulip/
│   ├── val/
│   │   ├── cactus/
│   │   ├── fern/
│   │   ├── rose/
│   │   ├── sunflower/
│   │   ├── tulip/
│   ├── train.cache
│   ├── val.cache
```

---

## 🚀 **Project Implementation**  

### **✔ 1️⃣ Data Loading**  
We load images from the dataset using `tf.keras.preprocessing.image_dataset_from_directory()`.  

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train", image_size=(224, 224), batch_size=32
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val", image_size=(224, 224), batch_size=32
)
```

---

### **✔ 2️⃣ Preprocessing & Data Augmentation**  
We normalize pixel values and apply transformations like flipping and rotation.  

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
])
```

---

### **✔ 3️⃣ CNN Model Creation**  
A deep learning model is built using Convolutional Neural Networks (CNNs).  

```python
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation="softmax")
])
```

---

### **✔ 4️⃣ Model Training & Evaluation**  
The model is compiled and trained using **Adam optimizer** and **sparse categorical crossentropy loss**.  

```python
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=20)
```

We then evaluate the trained model:  

```python
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
```

---

### **✔ 5️⃣ Predictions on New Images**  
We load an image and use the trained model to predict its class.  

```python
def predict_image(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

predict_image("dataset/val/rose/rose1.jpg", model, train_ds.class_names)
```

---

### **✔ 6️⃣ Model Saving & Deployment**  
Save the trained model for future use.  

```python
model.save("plant_classification_model.h5")
```

Convert to **TensorFlow Lite** for mobile or edge deployment.  

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("plant_classification_model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 📌 **Results & Accuracy**  
The CNN model achieves high accuracy in classifying plant species.  
📈 **Validation Accuracy:** **~95%**  

---

## 🛠 **Technologies Used**  
- **Python**  
- **TensorFlow / Keras**  
- **Matplotlib**  
- **NumPy**  
- **OpenCV** (for image processing)  

---

## 📝 **Future Improvements**  
🔹 Use a **larger dataset** for better accuracy.  
🔹 Implement **Transfer Learning (ResNet, MobileNet, EfficientNet)** for improved performance.  
🔹 Deploy as a **Flask API or Streamlit App** for real-world use.  

---

## 🎯 **Conclusion**  
This project demonstrates a **CNN-based deep learning model** for classifying plants into different species. The trained model can be further **optimized, fine-tuned, and deployed** for real-world applications.  

---

## ⭐ **Contribute**  
If you find this project useful, feel free to **⭐ Star the repository** and **Fork it**! 🚀  



