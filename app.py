import gradio as gr
from PIL import Image
from src.pipeline.predict_pipeline import Predictor

predictor = Predictor("artifacts/model.keras")

def predict_image(img):
    result = predictor.predict(img)
    output = (
        f"Gender: {result['gender']}\n"
        f"Race: {result['race']}\n"
        f"Age Group: {result['age_group']}"
    )
    return output

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Gender, Age & Race Prediction",
    description="Upload an image and get predictions.",
)

interface.launch(server_name="0.0.0.0")
