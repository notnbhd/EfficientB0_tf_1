Developed and trained a deep learning image classifier using PyTorch and EfficientNet-B0 architecture, leveraging transfer learning and fine-tuning techniques to achieve high accuracy on a custom food image dataset (e.g., noodles, sushi/pizza/steak).

Engineered an end-to-end ML pipeline: encompassing data acquisition (Food101 subsetting/optional web scraping), preprocessing (augmentation, normalization via torchvision.transforms), custom training/validation loops, and evaluation (accuracy ~95.7%).

Implemented robust training strategies: including learning rate scheduling (ReduceLROnPlateau) and early stopping based on validation accuracy to prevent overfitting and optimize model performance.

Deployed the trained classification model into a user-friendly web application using Flask, enabling real-time image uploads and predictions through a simple interface.

Managed model lifecycle: including saving the best performing model weights (torch.save), loading them for inference (torch.load), and structuring code for reproducible training and deployment (network.py, app.py).
