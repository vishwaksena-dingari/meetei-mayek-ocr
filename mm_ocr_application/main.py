import os
import gradio as gr

# from sklearn.utils import shuffle
from image_to_text import image_to_text


def read_content(file_path):
    """
    read the content of target file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def mm_ocr(image, model):
    text = image_to_text(image, model)
    return text, len(text), len("".join(text.split()))


root_dir = os.path.dirname(__file__)
images_folder_path = f"{root_dir}/images"
example_images = [
    f"{images_folder_path}/{image}" for image in os.listdir(images_folder_path)
]
models = ["mobilenet", "vgg16", "vgg19", "resnet", "xception"]
# models = ["mobilenet", "vgg16", "vgg19", "resnet"]

font = [gr.themes.GoogleFont("Noto Sans Meetei Mayek")]

with gr.Blocks(theme=gr.themes.Default(font=font), title="Meetei-Mayek OCR") as demo:
    gr.HTML(read_content(f"{root_dir}/header.html"))

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
        with gr.Column():
            output_text = gr.TextArea(label="Generated Text", interactive=True)
            with gr.Row():
                text_length = gr.Textbox(label="Text Length")
                char_count = gr.Textbox(label="Meetei-Mayek Character Count")

    with gr.Row():
        with gr.Column():
            input_model = gr.Radio(models, value="mobilenet", label="Model")
            with gr.Row():
                clear_button = gr.Button(value="Clear")
                submit_button = gr.Button(value="Submit", variant="primary")
        with gr.Column():
            gr.Examples(
                # examples=[*shuffle(example_images)[:5]],
                examples=[*example_images[:5]],
                inputs=input_image,
                outputs=output_text,
            )

        submit_button.click(
            fn=mm_ocr,
            inputs=[input_image, input_model],
            outputs=[output_text, text_length, char_count],
        )
        clear_button.click(
            # fn=default,
            fn=lambda: [None, "mobilenet", None, None, None],
            inputs=None,
            outputs=[input_image, input_model, output_text, text_length, char_count],
        )


if __name__ == "__main__":
    demo.launch()
