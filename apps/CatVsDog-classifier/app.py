import platform
from pathlib import Path
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath  # Handle paths from Linux-trained model

import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage

# Load exported FastAI model
learn = load_learner('CatVsDog.pkl')

# Define prediction function
def predict(img):
    try:
        pil_img = PILImage.create(img)  # Wrap for FastAI
        _, _, probs = learn.predict(pil_img)
        return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    except Exception as e:
        return {"error": str(e)}

# Gradio UI
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog image"),
    outputs=gr.Label(label="Prediction"),
    title="🐱 vs 🐶 Classifier",
    description="Upload an image and see whether it's a cat or a dog!",
).launch(debug=True)
