import gradio as gr
import cv2
import numpy as np
from PIL import Image

def remove_green_screen_rgb_interactive(image, r_min, g_min, b_min, r_max, g_max, b_max):
    if image is None:
        return None
    
    # 画像をRGBに変換してNumPy配列化
    image_rgb = image.convert("RGB")
    img_np = np.array(image_rgb)

    # スライダーの値を使って範囲を定義
    lower_rgb = np.array([r_min, g_min, b_min])
    upper_rgb = np.array([r_max, g_max, b_max])

    # 指定した色の範囲をマスク作成
    mask = cv2.inRange(img_np, lower_rgb, upper_rgb)

    # マスクを反転
    mask_inv = cv2.bitwise_not(mask)

    # アルファチャンネル結合
    r, g, b = cv2.split(img_np)
    rgba = [r, g, b, mask_inv]
    dst = cv2.merge(rgba)

    return Image.fromarray(dst)

# Gradioインターフェース
with gr.Blocks() as demo:
    gr.Markdown("## クロマキー調整ツール (RGB指定)")
    gr.Markdown("スライダーを動かすと、リアルタイムで透過具合が変化します。")

    with gr.Row():
        input_img = gr.Image(label="入力画像", type="pil")
        output_img = gr.Image(label="出力画像", type="pil")

    # 設定用アコーディオン（開閉可能）
    with gr.Accordion("RGB範囲設定 (ここを調整)", open=True):
        gr.Markdown("### 緑とみなす範囲の下限 (Minimum)")
        with gr.Row():
            r_min = gr.Slider(0, 255, value=0, label="R (赤) Min")
            g_min = gr.Slider(0, 255, value=115, label="G (緑) Min") # 緑の下限は高めに
            b_min = gr.Slider(0, 255, value=0, label="B (青) Min")

        gr.Markdown("### 緑とみなす範囲の上限 (Maximum)")
        with gr.Row():
            r_max = gr.Slider(0, 255, value=80, label="R (赤) Max")
            g_max = gr.Slider(0, 255, value=255, label="G (緑) Max")
            # ★重要: 水色を守るならここの値を小さくする
            b_max = gr.Slider(0, 255, value=80, label="B (青) Max") 

    # すべての入力コンポーネントをリストにまとめる
    inputs = [input_img, r_min, g_min, b_min, r_max, g_max, b_max]

    # 画像が変わったとき、またはスライダーが動いたときに実行
    input_img.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    r_min.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    g_min.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    b_min.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    r_max.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    g_max.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)
    b_max.change(remove_green_screen_rgb_interactive, inputs=inputs, outputs=output_img)

if __name__ == "__main__":
    demo.launch()