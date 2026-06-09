import os
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, EfficientNetB7
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pathlib import Path

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        _limit_mb = int(os.environ.get("GPU_MEMORY_LIMIT_MB", "0"))
        if _limit_mb > 0:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=_limit_mb)]
            )
            print(f"✓ GPU memory hard-capped to {_limit_mb} MB")
    except RuntimeError as e:
        print(f"GPU memory growth setup error: {e}")
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision training enabled (float16 where possible)")
except:
    print("Note: Mixed precision not available on this GPU")


class Config:
    DATASET_DIR = r"C:\mini proj\56rmx5bjcr-1\KneeXrayData\ClsKLData\kneeKL224"
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.10
    TEST_SIZE = 0.20
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    BATCH_SIZE = 32
    EFFICIENTNETB7_MICRO_BATCH = 2
    GRAD_ACCUM_STEPS = 4
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MIN_DELTA = 0.0001
    EARLY_STOPPING_MODE = 'min'
    PREFETCH_BUFFER = 1
    FREEZE_BASE_MODEL = True
    MULTI_CLASS_LABELS = {0: 'Normal', 1: 'Doubtful', 2: 'Mild',
                          3: 'Moderate', 4: 'Severe'}
    NUM_CLASSES = 5
    OUTPUT_DIR = r"C:\mini proj\outputs"
    MODELS_DIR = os.path.join(OUTPUT_DIR, "trained_models")

    def __post_init__(self):
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(self.MODELS_DIR).mkdir(exist_ok=True)


config = Config()
Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
Path(config.MODELS_DIR).mkdir(exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE
MODEL_NAMES = ['VGG16', 'ResNet50', 'EfficientNetB7']


def prepare_dataset(dataset):
    return dataset.prefetch(config.PREFETCH_BUFFER)


def dataset_for_model(dataset, model_name):
    batch_size = (
        config.EFFICIENTNETB7_MICRO_BATCH
        if model_name == 'EfficientNetB7'
        else config.BATCH_SIZE
    )
    return (
        dataset
        .unbatch()
        .batch(batch_size, drop_remainder=False)
        .prefetch(config.PREFETCH_BUFFER)
    )


def load_dataset_from_directory(dataset_dir, img_height=224, img_width=224):
    train_data = keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, 'train'),
        seed=42,
        image_size=(img_height, img_width),
        batch_size=config.BATCH_SIZE,
        label_mode='int',
        shuffle=True
    )
    val_data = keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, 'val'),
        seed=42,
        image_size=(img_height, img_width),
        batch_size=config.BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )
    test_data = keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, 'test'),
        seed=42,
        image_size=(img_height, img_width),
        batch_size=config.BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )
    _PARALLEL = min(4, os.cpu_count() or 1)
    normalization_layer = layers.Rescaling(1./255)
    train_data = train_data.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=_PARALLEL)
    val_data   = val_data.map(  lambda x, y: (normalization_layer(x), y), num_parallel_calls=_PARALLEL)
    test_data  = test_data.map( lambda x, y: (normalization_layer(x), y), num_parallel_calls=_PARALLEL)
    return prepare_dataset(train_data), prepare_dataset(val_data), prepare_dataset(test_data)


def create_synthetic_dataset(num_samples_per_class=100, num_classes=5):
    X = []
    y = []
    for class_id in range(num_classes):
        for _ in range(num_samples_per_class):
            img = np.random.rand(224, 224, 3).astype(np.float32)
            X.append(img)
            y.append(class_id)
    X = np.array(X)
    y = np.array(y)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    n_train = int(0.7 * len(X))
    n_val = int(0.1 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(config.BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config.BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(config.BATCH_SIZE)
    return prepare_dataset(train_ds), prepare_dataset(val_ds), prepare_dataset(test_ds)


def create_base_model(model_name, num_classes=5):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False,
                          input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    elif model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(weights='imagenet', include_top=False,
                                   input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    base_model.trainable = not config.FREEZE_BASE_MODEL
    return base_model


def build_finetuned_model(base_model, model_name, num_classes=5):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model


def compile_model(model, learning_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def clear_gpu_memory():
    print("\n✓ Clearing GPU memory...")
    gc.collect()
    tf.keras.backend.clear_session()
    logical_gpus = tf.config.list_logical_devices('GPU')
    if logical_gpus:
        try:
            for gpu in logical_gpus:
                tf.config.experimental.reset_memory_stats(gpu.name)
            print("✓ GPU memory cleared successfully")
        except Exception as e:
            print(f"Note: Could not reset GPU memory stats ({e})")
    else:
        print("Note: No GPU detected for memory clearing")


def train_model(model, model_name, train_ds, val_ds, num_epochs=1000):
    if model_name == 'EfficientNetB7':
        return train_model_with_grad_accum(
            model, model_name, train_ds, val_ds, num_epochs
        )
    print(f"\nTraining {model_name}...")
    print(
        f"Early stopping enabled: monitor={config.EARLY_STOPPING_MONITOR}, "
        f"patience={config.EARLY_STOPPING_PATIENCE}"
    )
    early_stop = EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.EARLY_STOPPING_MODE,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[early_stop],
        verbose=1
    )
    print(f"✓ {model_name} training completed")
    return history


def train_model_with_grad_accum(model, model_name, train_ds, val_ds, num_epochs=1000):
    print(f"\nTraining {model_name} with gradient accumulation...")
    print(
        f"  micro-batch={config.EFFICIENTNETB7_MICRO_BATCH}, "
        f"accum-steps={config.GRAD_ACCUM_STEPS}, "
        f"effective-batch={config.EFFICIENTNETB7_MICRO_BATCH * config.GRAD_ACCUM_STEPS}"
    )
    print(
        f"  Early stopping: monitor={config.EARLY_STOPPING_MONITOR}, "
        f"patience={config.EARLY_STOPPING_PATIENCE}"
    )
    optimizer = model.optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    accum_steps = config.GRAD_ACCUM_STEPS

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct = tf.cast(tf.equal(preds, tf.cast(y, tf.int32)), tf.float32)
        return loss, correct

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_correct  = 0
        epoch_samples  = 0
        accum_grads = None
        step_in_accum = 0

        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss   = loss_fn(labels, logits)
                scaled_loss = optimizer.get_scaled_loss(loss) if hasattr(optimizer, 'get_scaled_loss') else loss
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            if hasattr(optimizer, 'get_unscaled_gradients'):
                grads = optimizer.get_unscaled_gradients(grads)
            if accum_grads is None:
                accum_grads = [tf.Variable(tf.zeros_like(g), trainable=False) for g in grads]
            for ag, g in zip(accum_grads, grads):
                ag.assign_add(g if g is not None else tf.zeros_like(ag))
            step_in_accum += 1
            epoch_loss_sum += float(loss)
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            epoch_correct  += int(tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.int32)))
            epoch_samples  += int(tf.shape(labels)[0])
            if step_in_accum >= accum_steps:
                avg_grads = [ag / float(accum_steps) for ag in accum_grads]
                optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))
                for ag in accum_grads:
                    ag.assign(tf.zeros_like(ag))
                step_in_accum = 0

        if step_in_accum > 0 and accum_grads is not None:
            avg_grads = [ag / float(step_in_accum) for ag in accum_grads]
            optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))

        train_loss = epoch_loss_sum / max(epoch_samples, 1)
        train_acc  = epoch_correct  / max(epoch_samples, 1)

        val_loss_sum = 0.0
        val_correct  = 0
        val_samples  = 0
        for images, labels in val_ds:
            batch_loss, correct = val_step(images, labels)
            val_loss_sum += float(batch_loss) * int(tf.shape(labels)[0])
            val_correct  += int(tf.reduce_sum(correct))
            val_samples  += int(tf.shape(labels)[0])

        val_loss = val_loss_sum / max(val_samples, 1)
        val_acc  = val_correct  / max(val_samples, 1)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} — "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss    = val_loss
            best_weights     = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    print(f"✓ {model_name} training completed")
    return history


def model_output_path(model_name, task_name):
    safe_task_name = task_name.replace(" ", "_").replace("(", "").replace(")", "")
    return os.path.join(config.MODELS_DIR, f"{model_name}_{safe_task_name}.keras")


def evaluate_model_on_dataset(model, model_name, test_ds, subset):
    print(f"\nEvaluating {model_name}...")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def predict_step(x):
        return model(x, training=False)

    test_labels = []
    pred_labels = []
    loss_sum    = 0.0
    n_samples   = 0

    for images, labels in test_ds:
        logits = predict_step(images)
        batch_loss = float(loss_fn(labels, logits))
        loss_sum  += batch_loss * int(tf.shape(labels)[0])
        n_samples += int(tf.shape(labels)[0])
        pred_labels.extend(np.argmax(logits.numpy(), axis=1))
        test_labels.extend(labels.numpy())

    test_labels = np.asarray(test_labels)
    pred_labels = np.asarray(pred_labels)
    avg_loss    = loss_sum / max(n_samples, 1)

    accuracy = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average='weighted')
    cm = confusion_matrix(test_labels, pred_labels)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_score': f1,
        'confusion_matrix': cm,
        'subset': subset
    }


def train_and_evaluate_models_multiclass(train_ds, val_ds, test_ds):
    results = []
    best_model_path = None
    best_accuracy = -1.0

    for model_name in MODEL_NAMES:
        print(f"\n{'='*80}")
        print(f"Building {model_name} for multi-class classification")
        print(f"{'='*80}")

        print(f"[Pre-build] Clearing GPU memory before {model_name}...")
        clear_gpu_memory()

        model_train_ds = dataset_for_model(train_ds, model_name)
        model_val_ds = dataset_for_model(val_ds, model_name)
        model_test_ds = dataset_for_model(test_ds, model_name)

        base_model = create_base_model(model_name, num_classes=config.NUM_CLASSES)
        model = build_finetuned_model(base_model, model_name, num_classes=config.NUM_CLASSES)
        model = compile_model(model, learning_rate=config.LEARNING_RATE)

        history = train_model(
            model, model_name, model_train_ds, model_val_ds, num_epochs=config.NUM_EPOCHS
        )

        result = evaluate_model_on_dataset(
            model, model_name, model_test_ds, subset='Multi-class (0-4)'
        )

        save_path = model_output_path(model_name, 'multiclass')
        model.save(save_path)
        result['model_path'] = save_path
        results.append(result)

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_path = save_path

        del history, model, base_model
        print(f"[Post-model] Clearing GPU memory after {model_name}...")
        clear_gpu_memory()
        print("-" * 80)

    return results, best_model_path


def find_last_conv_layer_name(model):
    base_model = model.layers[0]
    for layer in reversed(base_model.layers):
        output = getattr(layer, 'output', None)
        if output is not None and len(output.shape) == 4:
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    feature_model = tf.keras.models.Model(
        base_model.inputs,
        [last_conv_layer.output, base_model.output]
    )
    classifier_layers = model.layers[1:]
    with tf.GradientTape() as tape:
        last_conv_output, x = feature_model(img_array)
        for layer in classifier_layers:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    return heatmap.numpy(), preds.numpy()


def visualize_gradcam_samples(model, model_name, test_ds, num_samples=5):
    sample_images = []
    true_labels = []
    pred_labels = []

    for batch_images, labels in test_ds:
        predictions = model.predict(batch_images, verbose=0)
        batch_preds = np.argmax(predictions, axis=1)
        for img, true_label, pred_label in zip(batch_images.numpy(), labels.numpy(), batch_preds):
            sample_images.append(img)
            true_labels.append(true_label)
            pred_labels.append(pred_label)
        if len(sample_images) >= num_samples * 8:
            break

    images = np.asarray(sample_images)
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    last_conv_layer_name = find_last_conv_layer_name(model)

    if last_conv_layer_name is None:
        print(f"Could not find convolutional layer for {model_name}")
        return

    correct_indices = np.where(pred_labels == true_labels)[0]
    incorrect_indices = np.where(pred_labels != true_labels)[0]

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle(f'GradCAM Visualization - {model_name}\n(Row 1: Correct, Row 2: Incorrect)')

    for i in range(num_samples):
        if i < len(correct_indices):
            idx = correct_indices[i]
            img = images[idx:idx+1]
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            heatmap, _ = make_gradcam_heatmap(img, model, last_conv_layer_name)
            axes[0, i].imshow(img[0], cmap='gray')
            axes[0, i].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[0, i].set_title(f'True: {true_label}, Pred: {pred_label}')
            axes[0, i].axis('off')
        if i < len(incorrect_indices):
            idx = incorrect_indices[i]
            img = images[idx:idx+1]
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            heatmap, _ = make_gradcam_heatmap(img, model, last_conv_layer_name)
            axes[1, i].imshow(img[0], cmap='gray')
            axes[1, i].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f'True: {true_label}, Pred: {pred_label}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f'gradcam_{model_name}.png'), dpi=150)
    plt.close()
    print(f"GradCAM visualization saved for {model_name}")


def plot_confusion_matrix(result, output_name):
    cm = result['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix - {result['model']} {result['subset']}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, output_name), dpi=150)
    plt.close()


def create_binary_subsets(train_ds, val_ds, test_ds):
    def filter_classes(dataset, class_0, class_1):
        class_0 = tf.cast(class_0, tf.int32)
        class_1 = tf.cast(class_1, tf.int32)

        def keep_class(img, label):
            return tf.logical_or(tf.equal(label, class_0), tf.equal(label, class_1))

        def remap_label(img, label):
            return img, tf.cast(tf.equal(label, class_1), tf.int32)

        return (
            dataset
            .unbatch()
            .filter(keep_class)
            .map(remap_label, num_parallel_calls=AUTOTUNE)
            .batch(config.BATCH_SIZE)
            .prefetch(config.PREFETCH_BUFFER)
        )

    subsets = {}
    subset_configs = [
        ('subset_1', 0, 1),
        ('subset_2', 0, 2),
        ('subset_3', 0, 3),
        ('subset_4', 0, 4)
    ]
    for subset_name, class_0, class_1 in subset_configs:
        train_subset = filter_classes(train_ds, class_0, class_1)
        val_subset = filter_classes(val_ds, class_0, class_1)
        test_subset = filter_classes(test_ds, class_0, class_1)
        subsets[subset_name] = (train_subset, val_subset, test_subset, (class_0, class_1))
    return subsets


def train_and_evaluate_models_binary(binary_subsets):
    results = {}
    for subset_name, (train_ds, val_ds, test_ds, classes) in binary_subsets.items():
        print(f"\n{'='*80}")
        print(f"Training models for {subset_name} (Classes {classes[0]} vs {classes[1]})")
        print(f"{'='*80}")
        subset_results = []
        for model_name in MODEL_NAMES:
            print(f"\nBuilding {model_name}...")
            print(f"[Pre-build] Clearing GPU memory before {model_name}...")
            clear_gpu_memory()

            model_train_ds = dataset_for_model(train_ds, model_name)
            model_val_ds = dataset_for_model(val_ds, model_name)
            model_test_ds = dataset_for_model(test_ds, model_name)

            base_model = create_base_model(model_name, num_classes=5)
            model = build_finetuned_model(base_model, model_name, num_classes=2)
            model = compile_model(model, learning_rate=config.LEARNING_RATE)

            history = train_model(
                model, model_name, model_train_ds, model_val_ds, num_epochs=config.NUM_EPOCHS
            )

            result = evaluate_model_on_dataset(model, model_name, model_test_ds, subset=classes)
            save_path = model_output_path(model_name, subset_name)
            model.save(save_path)
            result['model_path'] = save_path
            subset_results.append(result)

            del history, model, base_model
            print(f"[Post-model] Clearing GPU memory after {model_name}...")
            clear_gpu_memory()
            print("-" * 80)

        results[subset_name] = subset_results
    return results


def main():
    print("="*80)
    print("Knee Osteoarthritis Classification - Deep Learning & XAI")
    print("="*80)
    print("\n✓ GPU Memory Optimization Enabled")
    print(f"✓ Batch Size: {config.BATCH_SIZE} (EfficientNetB7 uses micro-batch {config.EFFICIENTNETB7_MICRO_BATCH})")
    print(f"✓ Base models frozen: {config.FREEZE_BASE_MODEL}")

    print("\nLoading dataset...")
    try:
        train_ds, val_ds, test_ds = load_dataset_from_directory(config.DATASET_DIR)
        print(f"Dataset loaded from {config.DATASET_DIR}")
    except:
        print(f"Dataset not found at {config.DATASET_DIR}")
        print("Creating synthetic dataset for demonstration...")
        train_ds, val_ds, test_ds = create_synthetic_dataset(num_samples_per_class=50)
        print("Synthetic dataset created (50 samples per class)")

    print("\n" + "="*80)
    print("MULTI-CLASS CLASSIFICATION (5 classes: 0-4)")
    print("="*80)

    multiclass_results, best_model_path = train_and_evaluate_models_multiclass(
        train_ds, val_ds, test_ds
    )

    best_model_idx = np.argmax([r['accuracy'] for r in multiclass_results])
    best_multiclass_result = multiclass_results[best_model_idx]
    best_model_name = best_multiclass_result['model']

    print(f"\n{'='*80}")
    print(f"Best Multi-class Model: {best_model_name}")
    print(f"Accuracy: {best_multiclass_result['accuracy']:.4f}")
    print(f"{'='*80}")

    plot_confusion_matrix(best_multiclass_result, 'confusion_matrix_multiclass.png')

    print(f"\nGenerating GradCAM visualizations for {best_model_name}...")
    best_model = keras.models.load_model(best_model_path)
    best_test_ds = dataset_for_model(test_ds, best_model_name)
    visualize_gradcam_samples(best_model, best_model_name, best_test_ds)
    del best_model
    clear_gpu_memory()

    print("\n" + "="*80)
    print("BINARY CLASSIFICATION (Class 0 vs others)")
    print("="*80)

    print("\nCreating binary classification subsets...")
    binary_subsets = create_binary_subsets(train_ds, val_ds, test_ds)

    binary_results = train_and_evaluate_models_binary(binary_subsets)

    for subset_name, results_list in binary_results.items():
        best_idx = np.argmax([r['accuracy'] for r in results_list])
        best_result = results_list[best_idx]
        plot_confusion_matrix(best_result, f'confusion_matrix_{subset_name}.png')

    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    print("\nMulti-class Results:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Loss':<12} {'F1-Score':<12}")
    print("-" * 80)
    for result in sorted(multiclass_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model']:<20} {result['accuracy']:<12.4f} "
              f"{result['loss']:<12.4f} {result['f1_score']:<12.4f}")

    print("\n\nBinary Classification Best Results per Subset:")
    print("-" * 80)
    for subset_name, results_list in binary_results.items():
        best_result = max(results_list, key=lambda x: x['accuracy'])
        print(f"\n{subset_name} (Classes {best_result['subset']}):")
        print(f"  Best Model: {best_result['model']}")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  F1-Score: {best_result['f1_score']:.4f}")

    print("\n" + "="*80)
    print("Results saved to:", config.OUTPUT_DIR)
    print("="*80)


if __name__ == "__main__":
    main()
