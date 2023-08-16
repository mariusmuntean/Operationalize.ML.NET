namespace Operationalize.ML.NET.Common;

public class LandmarkModelSettings
{
    public const string OnnxModelName = "lite-model_on_device_vision_classifier_landmarks_classifier_north_america_V1_1.onnx";
    public const string Input = "uint8_image_input";
    public const string Output = "transpose_1";

    public const string MlNetModelFileName = "landmark_classifier_onnx.zip";
    public const string LabelFileName = "landmarks_classifier_north_america_V1_label_map.csv";
}