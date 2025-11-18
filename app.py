import gradio as gr
from PIL import Image
from src.pipeline.predict_pipeline import Predictor

"""predictor = Predictor("artifacts/model.keras")

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
    outputs=gr.Textbox(),
    title="Gender, Age & Race Prediction",
    description="Upload an face image and get predictions. Please only upload .jpg files.",
    allow_flagging="never"  
)"""


predictor = Predictor("artifacts/model.keras")

def predict_image(img):
    result = predictor.predict(img)
    output = (
        f"Gender: {result['gender']}\n"
        f"Race: {result['race']}\n"
        f"Age Group: {result['age_group']}"
    )
    return output

with gr.Blocks() as demo:
    gr.Markdown("# Gender, Age & Race Prediction")
    gr.Markdown("Upload an image and get predictions.")

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image")
        out = gr.Textbox(label="Prediction")

    btn = gr.Button("Predict")
    btn.click(predict_image, inputs=inp, outputs=out)

if __name__ == "__main__":
    
    demo.launch(share=True)
