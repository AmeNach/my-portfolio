import platform
from pathlib import Path
import gradio as gr
from fastai.vision.core import PILImage

# Platform patch for loading model trained on Unix from Windows
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

learn = None  # global placeholder

# Define prediction function
def predict(img):
    global learn
    if learn is None:
        from fastai.learner import load_learner
        try:
            learn = load_learner('CatVsDog.pkl')
        except Exception as e:
            return {"error": f"Model failed to load: {str(e)}"}

    try:
        pil_img = PILImage.create(img)
        _, _, probs = learn.predict(pil_img)
        return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    except Exception as e:
        return {"error": str(e)}

# Launch Gradio UI
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog image"),
    outputs=gr.Label(label="Prediction"),
    title="🐱 vs 🐶 Classifier",
    description="Upload an image and see whether it's a cat or a dog!",
).launch(server_name="0.0.0.0", server_port=7860)
