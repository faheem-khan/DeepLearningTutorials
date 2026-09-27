"""Tutorial 2 MLP Classifier."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, learning_rate_init=0.001)

training_start = perf_counter()
mlp.fit(X_train_scaled, y_train)
baseline_training_seconds = perf_counter() - training_start

y_pred = mlp.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

print("\nMLP Structure:")
print(f"Number of layers: {mlp.n_layers_}")
print(f"Number of outputs: {mlp.n_outputs_}")
print(f"Activation function: {mlp.get_params()['activation']}")
print(f"Output activation function: {mlp.out_activation_}")
print(f"Number of epochs: {mlp.n_iter_}")

architectures = ((10, 10), (20, 20), (30, 30), (40, 40), (50, 50))
learning_rates = (0.001, 0.01, 0.1)
model_results = {((10, 10), 0.001): (mlp, baseline_training_seconds)}

for architecture in architectures:
	for learning_rate in learning_rates:
		if architecture == (10, 10) and learning_rate == 0.001:
			continue

		model = MLPClassifier(
			hidden_layer_sizes=architecture,
			max_iter=1000,
			random_state=42,
			learning_rate_init=learning_rate,
		)
		training_start = perf_counter()
		model.fit(X_train_scaled, y_train)
		training_seconds = perf_counter() - training_start
		model_results[(architecture, learning_rate)] = (model, training_seconds)

print("\nResults by architecture and learning rate:")
for architecture in architectures:
	for learning_rate in learning_rates:
		model, training_seconds = model_results[(architecture, learning_rate)]
		predictions = model.predict(X_test_scaled)
		model_accuracy = accuracy_score(y_test, predictions)
		print(
			f"hidden layers={architecture}, learning rate={learning_rate}: "
			f"accuracy={model_accuracy:.2f}, epochs={model.n_iter_}, "
			f"final training loss={model.loss_curve_[-1]:.4f}, "
			f"training time={training_seconds:.4f} seconds"
		)

plot_directory = Path(__file__).resolve().parent / "plots"
plot_directory.mkdir(exist_ok=True)

for architecture in architectures:
	plt.figure(figsize=(8, 6))
	for learning_rate in learning_rates:
		model, _ = model_results[(architecture, learning_rate)]
		plt.plot(model.loss_curve_, label=f"Learning rate={learning_rate}")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Learning Rate Comparison for Hidden Layers {architecture}')
	plt.legend()
	plt.grid()
	plot_path = plot_directory / f"learning_rate_comparison_{architecture[0]}_neurons.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Saved plot: {plot_path}")

plt.figure(figsize=(9, 6))
bar_width = 0.24
positions = list(range(len(architectures)))
for rate_index, learning_rate in enumerate(learning_rates):
	offset = (rate_index - (len(learning_rates) - 1) / 2) * bar_width
	training_times = [
		model_results[(architecture, learning_rate)][1]
		for architecture in architectures
	]
	bars = plt.bar(
		[position + offset for position in positions],
		training_times,
		width=bar_width,
		label=f"Learning rate={learning_rate}",
	)
	for bar, training_time in zip(bars, training_times):
		plt.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height(),
			f"{training_time:.3f}",
			ha="center",
			va="bottom",
			fontsize=8,
		)

plt.xticks(positions, [str(architecture) for architecture in architectures])
plt.xlabel('Hidden layer sizes')
plt.ylabel('Training time (seconds)')
plt.title('Training Time by Architecture and Learning Rate')
plt.legend()
plt.grid(axis="y", alpha=0.3)
time_plot_path = plot_directory / "training_time_comparison.png"
plt.savefig(time_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved training-time plot: {time_plot_path}")