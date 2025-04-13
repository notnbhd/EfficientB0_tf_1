1. Clone the repo:
    git clone https://github.com/notnbhd/EfficientB0_tf_1

2. Create and activate a virtual environment (recommended):
    python -m venv .venv

3. Install dependencies:
    pip install -r requirements.txt

4. Build model:
    run main.ipynb 
        or model_builder.py
    * note: if you use the notebook, you are gonna need to run the create_data.ipynb in data_creation first.
    
    -- when the training process finished, 'best_model.pth' file will be generated.

5. run the web application:
    - run app.py file.

    - open your web browser and navigate to http://127.0.0.1:5000 (or the address provided in the terminal output).

    - **Upload an image** using the form.
    - Click **"Classify Image"**.
      The application will display the uploaded image and the predicted food class based on the model loaded in `network.py`.

## Model Details

*   **Architecture:** EfficientNet-B0 (using`torchvision.models`)
*   **Approach:** Transfer Learning (freezing most base layers, fine-tuning later layers and the classifier).
*   **Dataset:** Subset of Food101 (Pho, Ramen, Spaghetti Carbonara)
    * note: i builded a image scraper using google search api but it does not work well yet.
