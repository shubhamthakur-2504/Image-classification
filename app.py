import gradio as gr
from src.pipelines.prediction import ImageClassifier
from PIL import Image
classifier = ImageClassifier()
def classify_image(image):
    image_path = "artifacts/uploaded_image.jpg"
    image.save(image_path)
    label, output_path = classifier.predict(image_path)
    return label, Image.open(output_path)

demo = gr.Interface(
    fn = classify_image,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Textbox(label="Predicted Label"), gr.Image(label="Labeled Image")],
    title="Image Classification App",
    description="Upload an image and get the predicted label and labeled image."

)
if __name__ == "__main__":
    demo.launch(share=True)