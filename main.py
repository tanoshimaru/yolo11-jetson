from ultralytics import YOLO


def main():
    # Load a YOLO11n PyTorch model
    model = YOLO("yolo11n.pt")

    # Export the model to TensorRT
    model.export(format="engine")  # creates 'yolo11n.engine'

    # Load the exported TensorRT model
    trt_model = YOLO("yolo11n.engine")

    # Run inference
    results = trt_model("https://ultralytics.com/images/bus.jpg")

    print(results[0])


if __name__ == "__main__":
    main()
