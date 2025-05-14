from ultralytics import YOLO
import cv2
import os
import shutil


def main():
    if not os.path.exists("models/yolo11n.engine"):
        # Load a YOLO11n PyTorch model
        model = YOLO("models/yolo11n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo11n.engine'

    # Load the exported TensorRT model
    trt_model = YOLO("models/yolo11n.engine", task="detect")

    # 入力ディレクトリ
    input_dir = "images"
    
    # 人が検出された/検出されなかった画像の振り分け先ディレクトリ
    person_dir = os.path.join(input_dir, "person")
    no_person_dir = os.path.join(input_dir, "no-person")
    
    # ディレクトリが存在しない場合は作成
    for directory in [person_dir, no_person_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 出力ディレクトリがなければ作成
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 画像ファイルのリストを取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)
    
    print(f"{len(image_files)}枚の画像が見つかりました")
    
    # 各画像に対して推論を実行
    for image_path in image_files:
        print(f"処理中: {image_path}")
        
        # 推論実行
        results = trt_model(image_path)
        result = results[0]
        
        # 検出結果を画像に描画
        result_image = result.plot()
        
        # 出力ファイルパスを設定
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{output_filename}")
        
        # 画像を保存
        cv2.imwrite(output_path, result_image)
        print(f"検出結果を保存しました: {output_path}")
        
        # 人が検出されたかどうかを確認
        person_detected = False
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            if class_name.lower() == 'person':
                person_detected = True
                break
        
        # 人の検出結果に基づいて画像を振り分け
        dest_dir = person_dir if person_detected else no_person_dir
        dest_path = os.path.join(dest_dir, os.path.basename(image_path))
        
        # 画像を移動
        shutil.move(image_path, dest_path)  # 移動操作
        print(f"画像を{'person' if person_detected else 'no-person'}ディレクトリに移動しました")


if __name__ == "__main__":
    main()
