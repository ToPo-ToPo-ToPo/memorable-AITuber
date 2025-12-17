
import cv2
import numpy as np
import os
import glob

# ==========================================
# 設定エリア
# ==========================================

# 1. 処理対象の画像が入っているフォルダ名
# ※このスクリプトと同じ場所にフォルダを置いてください
INPUT_FOLDER = "assets/characters"

# 2. 処理後の画像を保存するフォルダ名（自動作成されます）
OUTPUT_FOLDER = "assets/characters_v1"

# 3. 確定したベストなRGB透過設定値
# (R:0-80, G:115-255, B:0-80 で設定)
LOWER_RGB = np.array([0, 115, 0])   # 下限 [R, G, B]
UPPER_RGB = np.array([120, 255, 86]) # 上限 [R, G, B]

# ==========================================

def process_and_save(input_path, output_dir):
    """
    画像を読み込み、緑背景を透過して、リネームして保存する関数
    """
    # ファイル名を取得（例: image01.png）
    filename = os.path.basename(input_path)
    # 拡張子を除いた名前と拡張子を取得（例: image01, .png）
    name_only, ext = os.path.splitext(filename)
    
    # 新しいファイル名を作成（例: image01_v1.png）
    new_filename = f"{name_only}{ext}"
    # 出力先のフルパスを作成
    output_path = os.path.join(output_dir, new_filename)

    print(f"処理中...: {filename} -> {new_filename}")

    # OpenCVで画像を読み込む (デフォルトはBGR順)
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"  [エラー] 画像を読み込めませんでした。スキップします。: {input_path}")
        return

    # 判定のためにBGRからRGBに変換
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 指定した色の範囲でマスクを作成（対象範囲が白、それ以外が黒になる）
    mask = cv2.inRange(img_rgb, LOWER_RGB, UPPER_RGB)

    # マスクを反転（背景以外を白=透過しない、背景を黒=透過する にする）
    mask_inv = cv2.bitwise_not(mask)

    # アルファチャンネルを作成
    # 元のBGR画像をチャンネルごとに分離
    b, g, r = cv2.split(img_bgr)
    # B, G, R, A(反転マスク) の順で結合してBGRA画像を作成
    bgra = cv2.merge([b, g, r, mask_inv])

    # 画像をPNG形式で保存
    cv2.imwrite(output_path, bgra)
    print("  -> 完了")


def main():
    # 入力フォルダの存在確認
    if not os.path.exists(INPUT_FOLDER):
        print(f"エラー: 入力フォルダ '{INPUT_FOLDER}' が見つかりません。")
        return

    # 出力フォルダが存在しなければ作成する
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"出力フォルダ '{OUTPUT_FOLDER}' を作成しました。")

    # 入力フォルダ内の .png ファイルをすべて取得する
    # (*.png なので小文字の拡張子のみ対象です。大文字も含む場合は工夫が必要です)
    input_files = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

    if not input_files:
        print(f"エラー: '{INPUT_FOLDER}' の中にPNGファイルが見つかりませんでした。")
        return

    print(f"合計 {len(input_files)} 枚の画像を処理します。")
    print("-" * 40)

    # 各ファイルに対して処理を実行
    for file_path in input_files:
        process_and_save(file_path, OUTPUT_FOLDER)

    print("-" * 40)
    print(f"すべての処理が完了しました。出力フォルダ '{OUTPUT_FOLDER}' を確認してください。")


if __name__ == "__main__":
    main()