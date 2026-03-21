import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from optax import exponential_decay
import flax.linen as nn
from flax.training import train_state

import torch
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ---------------------- Utilities ----------------------

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# ---------------------- DataLoader ----------------------

def get_dataloaders(batch_size=100, num_workers=0, val_split=0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)

    indices = np.arange(len(full_train_ds))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

    train_subset = data.Subset(full_train_ds, train_idx)
    val_subset = data.Subset(full_train_ds, val_idx)

    train_loader = data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=numpy_collate
    )

    val_loader = data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=numpy_collate
    )

    return train_loader, val_loader

# ---------------------- Model ----------------------

class MLP(nn.Module):
    layer_dims: list
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        for dim in self.layer_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
        x = nn.Dense(10)(x)  # output layer
        return x

# ---------------------- Training Utilities ----------------------

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, learning_rate=1e-3, layer_dims=[512, 256, 128]):
    model = MLP(layer_dims=layer_dims)
    dummy_input = jnp.ones([1, 28, 28], jnp.float32)
    params = model.init(rng, dummy_input)["params"]

    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=100,
        decay_rate=0.9,
        staircase=False
    )

    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-3)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ---------------------- Loss Functions ----------------------

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    return optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()

def mean_squared_error_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    probs = nn.softmax(logits)
    return jnp.mean((probs - one_hot) ** 2)

def compute_metrics(logits, labels, loss_fn):
    loss = loss_fn(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

# ---------------------- Training Functions ----------------------

def train_step(state, batch, loss_fn):
    images, labels = batch

    def loss_inner(params):
        logits = state.apply_fn({"params": params}, images)
        loss, _ = compute_metrics(logits, labels, loss_fn)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_inner, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    _, accuracy = compute_metrics(logits, labels, loss_fn)
    return state, loss, accuracy

train_step = jax.jit(train_step, static_argnames=["loss_fn"])

def eval_step(state, batch, loss_fn):
    images, labels = batch
    logits = state.apply_fn({"params": state.params}, images)
    _, accuracy = compute_metrics(logits, labels, loss_fn)
    return accuracy

eval_step = jax.jit(eval_step, static_argnames=["loss_fn"])

# ---------------------- Training Loop ----------------------

def train_model(num_epochs=10, batch_size=100, learning_rate=1e-3,
                loss_fn=cross_entropy_loss, label="", layer_dims=[512, 256, 128]):
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate, layer_dims)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, num_epochs + 1):
        epoch_accs = []
        for images, labels in train_loader:
            batch_np = (np.array(images), np.array(labels))
            state, loss, acc = train_step(state, batch_np, loss_fn)
            epoch_accs.append(acc)

        train_acc = np.mean(jax.device_get(np.array(epoch_accs)))

        val_accs, sizes = [], []
        for images, labels in val_loader:
            batch_np = (np.array(images), np.array(labels))
            acc = eval_step(state, batch_np, loss_fn)
            val_accs.append(acc)
            sizes.append(images.shape[0])

        val_acc = np.sum(np.array(val_accs) * np.array(sizes)) / np.sum(sizes)

        print(f"Epoch {epoch:02d} - Train Acc: {train_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%")

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # Plot accuracy
    plt.plot(np.arange(1, num_epochs + 1), np.array(train_accuracies) * 100, label=f"{label} - Train")
    plt.plot(np.arange(1, num_epochs + 1), np.array(val_accuracies) * 100, label=f"{label} - Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy: {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return state

# ---------------------- Evaluation and Visualization ----------------------

def evaluate_model(state, batch_size=100, loss_fn=cross_entropy_loss):
    _, test_loader = get_dataloaders(batch_size=batch_size)
    accs, sizes = [], []
    for images, labels in test_loader:
        batch_np = (np.array(images), np.array(labels))
        acc = eval_step(state, batch_np, loss_fn)
        accs.append(acc)
        sizes.append(images.shape[0])

    total_acc = np.sum(np.array(accs) * np.array(sizes)) / np.sum(sizes)
    print(f"Test Accuracy: {total_acc * 100:.2f}%")
    return total_acc

def visualize_predictions(state, num_examples=49):
    _, test_loader = get_dataloaders(batch_size=num_examples)
    images, labels = next(iter(test_loader))
    images_np = np.array(images)
    logits = state.apply_fn({"params": state.params}, images_np)
    preds = np.argmax(np.array(logits), axis=-1)

    plt.figure(figsize=(6, 6), dpi=140)
    for i in range(num_examples):
        plt.subplot(int(np.sqrt(num_examples)), int(np.sqrt(num_examples)), i + 1)
        plt.imshow(images_np[i, 0], cmap="gray")
        plt.title(f"P:{preds[i]} / T:{labels[i]}", fontsize=6)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ---------------------- Main ----------------------

def main():
    loss_fns = {
        "Cross-Entropy": cross_entropy_loss,
        "Cross-Entropy2": cross_entropy_loss,
        "Cross-Entropy3": cross_entropy_loss,
        "Mean Squared Error": mean_squared_error_loss,
        "Mean Squared Error2": mean_squared_error_loss,
        "Mean Squared Error3": mean_squared_error_loss
    }

    for loss_name, loss_fn in loss_fns.items():
        print(f"\n--- Training with {loss_name} Loss ---")
        state = train_model(
            num_epochs=10,
            batch_size=100,
            learning_rate=1e-3,
            loss_fn=loss_fn,
            label=loss_name
        )
        print(f"\n[{loss_name}] Evaluation on test set:")
        evaluate_model(state, loss_fn=loss_fn)
        print(f"[{loss_name}] Visualizing predictions:")
        visualize_predictions(state, num_examples=49)
        print("-" * 60)

if __name__ == "__main__":
    main()
