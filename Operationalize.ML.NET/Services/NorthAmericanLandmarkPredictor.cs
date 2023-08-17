using Microsoft.Extensions.ML;
using Operationalize.ML.NET.Common;

namespace Operationalize.ML.NET.Services;

internal class NorthAmericanLandmarkPredictor : INorthAmericanLandmarkPredictor
{
    private readonly PredictionEnginePool<LandmarkInput, LandmarkOutput> _predictionEnginePool;
    private readonly INorthAmericanLabelProvider _northAmericanLabelProvider;

    public NorthAmericanLandmarkPredictor(PredictionEnginePool<LandmarkInput, LandmarkOutput> predictionEnginePool, INorthAmericanLabelProvider northAmericanLabelProvider)
    {
        _predictionEnginePool = predictionEnginePool;
        _northAmericanLabelProvider = northAmericanLabelProvider;
    }

    public List<LandmarkPrediction> PredictLandmark(Stream imageStream)
    {
        var labels = _northAmericanLabelProvider.GetLabels();

        // Make prediction
        // Post process prediction - the output contains duplicates, so we should group by label and take the entry with the highest probability.
        // Docs - https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_north_america_V1/1
        var landmarkOutput = _predictionEnginePool.Predict(new LandmarkInput(imageStream));
        return landmarkOutput.Prediction
            .Zip(labels, (probability, landmarkName) => (LandmarkName: landmarkName, Probability: probability))
            .GroupBy(tuple => tuple.LandmarkName)
            .Select(group => new LandmarkPrediction(
                group.Key,
                group.MaxBy(tuple => tuple.Probability).Probability
            ))
            .OrderByDescending(prediction => prediction.Probability)
            .ToList();
    }
}