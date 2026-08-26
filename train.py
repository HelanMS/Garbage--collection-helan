"""
Waste Classification CNN + Tableau Export
Run: python train.py
"""

# ============================================================
# CONFIG — edit these two paths for your machine
# ============================================================
SOURCE_DIR = r'C:\Users\helan\Downloads\cnn_project\Garbage classification\Garbage classification'   # folder that directly contains cardboard/, glass/, metal/, paper/, plastic/, trash/
SPLIT_DIR = r'C:\Users\helan\Downloads\cnn_project'                # folder containing the three .txt split files

# ============================================================
# STEP 1: Organize dataset into train/val/test using official splits
# ============================================================
import os, re, shutil

OUTPUT_DIR = 'data_split'
SPLIT_FILES = {
    'train': 'one-indexed-files-notrash_train.txt',
    'val': 'one-indexed-files-notrash_val.txt',
    'test': 'one-indexed-files-notrash_test.txt',
}

def class_from_filename(filename):
    return re.match(r'^([a-zA-Z]+)', filename).group(1)

counts = {'train': 0, 'val': 0, 'test': 0}
missing = []

for split, fname in SPLIT_FILES.items():
    split_path = os.path.join(SPLIT_DIR, fname)
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name = line.split()[0]
            cls = class_from_filename(image_name)

            src_path = os.path.join(SOURCE_DIR, cls, image_name)
            dst_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, image_name)

            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                counts[split] += 1
            else:
                missing.append(src_path)

print('Copied:', counts)
if missing:
    print(f'WARNING: {len(missing)} files not found. First few:')
    for p in missing[:5]:
        print(' ', p)

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
print('Classes found:', sorted(os.listdir(TRAIN_DIR)))

# ============================================================
# STEP 2: Load data with augmentation
# ============================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2]
)

eval_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=42
)

val_gen = eval_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=42, shuffle=False
)

test_gen = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=42, shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print('Classes:', class_names)
print('Train:', train_gen.samples, '| Val:', val_gen.samples, '| Test:', test_gen.samples)

# ============================================================
# STEP 3: Build model (MobileNetV2 transfer learning)
# ============================================================
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# STEP 4: Train (Phase 1: frozen base, Phase 2: fine-tune)
# ============================================================
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
]

history_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# ============================================================
# STEP 5: Plot & save training curves
# ============================================================
import matplotlib
matplotlib.use('Agg')  # avoids display issues when running as a plain script
import matplotlib.pyplot as plt
import pandas as pd

acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(acc, label='train'); axes[0].plot(val_acc, label='val')
axes[0].set_title('Accuracy'); axes[0].legend()
axes[1].plot(loss, label='train'); axes[1].plot(val_loss, label='val')
axes[1].set_title('Loss'); axes[1].legend()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

epoch_df = pd.DataFrame({
    'epoch': range(1, len(acc) + 1),
    'train_accuracy': acc,
    'val_accuracy': val_acc,
    'train_loss': loss,
    'val_loss': val_loss
})
epoch_df.to_csv('training_history.csv', index=False)
print('Saved training_curves.png and training_history.csv')

# ============================================================
# STEP 6: Evaluate on test set
# ============================================================
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

test_gen.reset()
pred_probs = model.predict(test_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes
confidences = np.max(pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved confusion_matrix.png')

# ============================================================
# STEP 7: Export everything Tableau needs
# ============================================================
import datetime, random

filenames = test_gen.filenames

records = []
base_date = datetime.date(2026, 1, 1)
for i in range(len(filenames)):
    records.append({
        'image_id': filenames[i],
        'true_label': class_names[y_true[i]],
        'predicted_label': class_names[y_pred[i]],
        'confidence': round(float(confidences[i]), 4),
        'correct': bool(y_true[i] == y_pred[i]),
        'bin_location': random.choice(['Campus A', 'Campus B', 'Downtown Hub', 'Mall Kiosk', 'Park Station']),
        'date': (base_date + datetime.timedelta(days=random.randint(0, 180))).isoformat()
    })

predictions_df = pd.DataFrame(records)
predictions_df.to_csv('predictions_for_tableau.csv', index=False)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_long = cm_df.reset_index().melt(id_vars='index', var_name='predicted_label', value_name='count')
cm_long = cm_long.rename(columns={'index': 'true_label'})
cm_long.to_csv('confusion_matrix_long.csv', index=False)

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
per_class_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'class'})
per_class_df = per_class_df[per_class_df['class'].isin(class_names)]
per_class_df.to_csv('per_class_metrics.csv', index=False)

print('\nDone! Files ready for Tableau:')
print(' - predictions_for_tableau.csv')
print(' - confusion_matrix_long.csv')
print(' - per_class_metrics.csv')
print(' - training_history.csv')
print(' - confusion_matrix.png / training_curves.png')