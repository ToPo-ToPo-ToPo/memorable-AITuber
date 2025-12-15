
import gradio as gr
import cv2
import numpy as np
from PIL import Image

def remove_green_screen(image):
    if image is None:
        return None
    
    # PIL画像をOpenCV形式(numpy array)に変換
    # PILはRGB、OpenCVはBGRなので変換が必要
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB) # アルファチャンネルがあれば一度捨てる

    # 画像をHSV色空間に変換（色の範囲指定がしやすいため）
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 緑色の範囲を定義 (色相:Hue, 彩度:Saturation, 明度:Value)
    # 背景の緑に合わせて調整が必要な場合があります
    lower_green = np.array([35, 100, 100]) # 緑色の下限
    upper_green = np.array([85, 255, 255]) # 緑色の上限

    # 緑色の部分をマスク（白黒画像）にする
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # マスクを反転（緑以外の部分を白にする）
    mask_inv = cv2.bitwise_not(mask)

    # RGBA（アルファチャンネル付き）画像を作成
    # 元の画像のRGBチャンネルに、作成したマスクをアルファチャンネルとして結合
    b, g, r = cv2.split(img_rgb)
    rgba = [b, g, r, mask_inv]
    dst = cv2.merge(rgba, 4)

    # NumPy配列をPIL画像に戻す
    return Image.fromarray(dst)

# Gradioインターフェース
with gr.Blocks() as demo:
    gr.Markdown("## クロマキー（緑背景）削除ツール")
    with gr.Row():
        inp = gr.Image(label="入力画像", type="pil")
        out = gr.Image(label="出力画像", type="pil")
    
    btn = gr.Button("実行")
    btn.click(remove_green_screen, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()