import gradio as gr
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

def extract_colors(image, num_colors=5):
    image = image.resize((200, 200))  # Downsize for faster processing
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)

    hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in colors]
    return hex_colors

def plot_palette(hex_colors):
    fig, ax = plt.subplots(1, figsize=(5, 1))
    for i, color in enumerate(hex_colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    plt.xlim(0, len(hex_colors))
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def process_image(image):
    image = Image.fromarray(image)
    colors = extract_colors(image)
    palette_img = plot_palette(colors)
    return palette_img, "\n".join(colors)

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[
        gr.Image(type="pil", label="Generated Palette"),
        gr.Textbox(label="Hex Codes")
    ],
    title="AI Color Palette Generator",
    description="Upload an image and extract dominant colors using K-Means. Lightweight & Fast!"
)

if __name__ == "__main__":
    demo.launch()
