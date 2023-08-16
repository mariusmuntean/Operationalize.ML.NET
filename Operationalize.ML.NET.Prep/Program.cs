using System.Diagnostics;
using Microsoft.ML;
using Operationalize.ML.NET.Common;

// Configure ml model
var mlCtx = new MLContext();

var pipeline = mlCtx
    .Transforms
    .ResizeImages(
        inputColumnName: nameof(LandmarkInput.Image),
        imageWidth: LandmarkImageSettings.ImageWidth,
        imageHeight: LandmarkImageSettings.ImageHeight,
        outputColumnName: "resized"
    )
    .Append(mlCtx.Transforms.ExtractPixels(
        inputColumnName: "resized",
        interleavePixelColors: true,
        outputAsFloatArray: false,
        outputColumnName: LandmarkModelSettings.Input)
    )
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
// Bitmap image = Image.FromFile("255023953.jpeg") as Bitmap;
var sw = new Stopwatch();
sw.Start();
// var prediction = predictionEngine.Predict(new Input() { Image = image });
var prediction = predictionEngine.Predict(new LandmarkInput(File.Open("Landmarks/Statue_of_Liberty_7.jpg", FileMode.Open)));
Console.WriteLine($"Prediction took: {sw.ElapsedMilliseconds}ms");

// Output
var labels = await File.ReadAllLinesAsync(LandmarkModelSettings.LabelFileName)
    .ContinueWith(lineTask =>
    {
        var lines = lineTask.Result;
        return lines
            .Skip(1)
            .Select(line => line.Split(",").Last())
            .ToArray();
    });
var predictions = prediction.Prediction
        .Select((val, index) => (index, probabiliy: val))
        .Where(pair => pair.probabiliy > 0.55f)
        .Select(pair => (name: labels[pair.index], pair.probabiliy))
        .GroupBy(pair => pair.name)
        .Select(group => (name: group.Key, probability: group.Select((p) => p.probabiliy).Max()))
        .OrderByDescending(pair => pair.probability)
    ;
var predictionsString = string.Join(Environment.NewLine, predictions.Select(pair => $"name: {pair.name}, probability: {pair.probability}"));
Console.WriteLine(string.Join(Environment.NewLine, predictionsString));