# yolo11-jetson

参考：[https://docs.ultralytics.com/ja/guides/nvidia-jetson/](https://docs.ultralytics.com/ja/guides/nvidia-jetson/)

## Usage

まず，人の写り込みの有無で振り分けたい画像を`images`フォルダに配置します。

以下のコマンドを実行すると，`images`フォルダ内の画像を推論し，`results`フォルダに結果を保存します。

また，人を検出した画像は`images/person`，検出されなかった画像は`images/no-person`に移動されます。

```bash
docker compose up -d --build
```
