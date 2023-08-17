namespace Operationalize.ML.NET.Services;

public interface INorthAmericanLandmarkPredictor
{
    List<LandmarkPrediction> PredictLandmark(Stream imageStream);
}

public record LandmarkPrediction(string LandmarkName, float Probability);