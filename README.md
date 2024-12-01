# 🏠 Furniture Classification Project

<div align="center">
<img src="https://img.freepik.com/free-vector/set-isolated-furniture-interior-decor-icons-with-images-soft-furniture-with-tables-carpet-vector-illustration_1284-70894.jpg?t=st=1732607764~exp=1732611364~hmac=d62519eb8431ff441e689d1b8747099c3be45e03d9a15177e20079da566c2b70&w=996" alt="Furniture Classification" width="500"/>
</div>

## 📝 Description
A deep learning project to classify different types of furniture using PyTorch and CNN architecture.

## 🏷️ Classes
Our model classifies **7 different types** of furniture:
- 🪑 Chair
- 🗄️ Cupboard
- ❄️ Fridge
- 🪟 Table
- 📺 TV
- 🛏️ Bed
- 🛋️ Sofa

## 📊 Dataset Information
| Split | Percentage | Images per Class |
|-------|------------|-----------------|
| Training | 60% | 120 |
| Validation | 30% | 60 |
| Testing | 10% | 20 |

**Total Images per Class**: 200  
**Image Dimensions**: 177 x 177 pixels

## 🏗️ Model Architecture
Using ***SimpleCNN*** with the following structure:
```python
Input (3, 177, 177)
    │
    ├── Conv2d + BatchNorm + LeakyReLU
    │
    ├── 4x ConvBlock (Conv + BatchNorm + MaxPool)
    │
    ├── Adaptive MaxPool
    │
    └── Fully Connected Layers
         └── Output (7 classes)
