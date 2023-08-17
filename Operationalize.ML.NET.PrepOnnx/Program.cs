using System.Diagnostics;
using Microsoft.ML;
using Operationalize.ML.NET.Common;

// Configure ML model
var mlCtx = new MLContext();

var pipeline = mlCtx
    .Transforms
    // Adjust the image to the required model input size
    .ResizeImages(
        inputColumnName: nameof(LandmarkInput.Image),
        imageWidth: LandmarkInput.ImageWidth,
        imageHeight: LandmarkInput.ImageHeight,
        outputColumnName: "resized"
    )
    // Extract the pixels form the image as a 1D float array, but keep them in the same order as they appear in the image.
    .Append(mlCtx.Transforms.ExtractPixels(
        inputColumnName: "resized",
        interleavePixelColors: true,
        outputAsFloatArray: false,
        outputColumnName: LandmarkModelSettings.Input)
    )
    // Perform the estimation
    .Append(mlCtx.Transforms.ApplyOnnxModel(
            modelFile: "./" + LandmarkModelSettings.OnnxModelName,
            inputColumnName: LandmarkModelSettings.Input,
            outputColumnName: LandmarkModelSettings.Output
        )
    );

// Save ml model
var transformer = pipeline.Fit(mlCtx.Data.LoadFromEnumerable(new List<LandmarkInput>()));

mlCtx.Model.Save(transformer, null, LandmarkModelSettings.MlNetModelFileName);

// Load ml model
var mlCtx2 = new MLContext();
var loadedModel = mlCtx2.Model.Load(LandmarkModelSettings.MlNetModelFileName, out var _);
var predictionEngine = mlCtx2.Model.CreatePredictionEngine<LandmarkInput, LandmarkOutput>(loadedModel);

// Predict 
var sw = new Stopwatch();
sw.Start();
await using var imagesStream = File.Open("Landmarks/Statue_of_Liberty_7.jpg", FileMode.Open);
var prediction = predictionEngine.Predict(new LandmarkInput(imagesStream));
Console.WriteLine($"Prediction took: {sw.ElapsedMilliseconds}ms");

// Labels start from the second line and each contains the 0-based index, a comma and a name.
var labels = await File.ReadAllLinesAsync(LandmarkModelSettings.LabelFileName)
    .ContinueWith(lineTask =>
    {
        var lines = lineTask.Result;
        return lines
            .Skip(1)
            .Select(line => line.Split(",").Last())
            .ToArray();
    });

// Merge the prediction array with the labels. Produce tuples of landmark name and its probability.
var predictions = prediction.Prediction
        .Select((val, index) => (index, probabiliy: val))
        .Where(pair => pair.probabiliy > 0.55f)
        .Select(pair => (name: labels[pair.index], pair.probabiliy))
        .GroupBy(pair => pair.name)
        .Select(group => (name: group.Key, probability: group.Select((p) => p.probabiliy).Max()))
        .OrderByDescending(pair => pair.probability)
    ;

// Output
var predictionsString = string.Join(Environment.NewLine, predictions.Select(pair => $"name: {pair.name}, probability: {pair.probability}"));
Console.WriteLine(string.Join(Environment.NewLine, predictionsString));