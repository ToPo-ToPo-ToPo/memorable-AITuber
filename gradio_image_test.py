
import gradio as gr
import cv2
import numpy as np
from PIL import Image

def remove_green_screen_rgb(image):
    if image is None:
        return None
    
    # 画像がRGBAなどの場合もあるため、判定用に一度RGBに統一します
    image_rgb = image.convert("RGB")
    img_np = np.array(image_rgb)

    # RGBでの緑色の範囲定義
    # R (赤): 0〜80   (緑に赤はほぼ混ざらないので低く)
    # G (緑): 150〜255 (明るい緑色だけを対象にする)
    # B (青): 0〜80   (★ここが重要！水色は青が混ざるので、青が低い範囲に限定する)
    
    lower_rgb = np.array([0, 150, 0])    # 最小値 [R, G, B]
    upper_rgb = np.array([80, 255, 80])  # 最大値 [R, G, B]

    # 指定した色の範囲をマスク（白黒画像）にする
    mask = cv2.inRange(img_np, lower_rgb, upper_rgb)

    # マスクを反転（緑以外の部分を白にする）
    mask_inv = cv2.bitwise_not(mask)

    # 元画像（RGB）にアルファチャンネル（透過情報）を追加してRGBAにする
    r, g, b = cv2.split(img_np)
    rgba = [r, g, b, mask_inv]
    dst = cv2.merge(rgba)

    return Image.fromarray(dst)

# Gradioインターフェース
with gr.Blocks() as demo:
    gr.Markdown("## クロマキー削除（RGB指定版）")
    with gr.Row():
        inp = gr.Image(label="入力画像", type="pil")
        out = gr.Image(label="出力画像", type="pil")
    
    btn = gr.Button("実行")
    btn.click(remove_green_screen_rgb, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()