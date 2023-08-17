using Microsoft.AspNetCore.Mvc;
using Operationalize.ML.NET.Services;

namespace Operationalize.ML.NET.Controllers;

[ApiController]
[Route("[controller]")]
public class LandmarkPredictionController : ControllerBase
{
    private readonly INorthAmericanLandmarkPredictor _northAmericanLandmarkPredictor;

    public LandmarkPredictionController(INorthAmericanLandmarkPredictor northAmericanLandmarkPredictor)
    {
        _northAmericanLandmarkPredictor = northAmericanLandmarkPredictor;
    }

    [HttpPost("NorthAmerica")]
    public async Task<List<LandmarkPrediction>> Get(IFormFile image)
    {
        var prediction = _northAmericanLandmarkPredictor.PredictLandmark(image.OpenReadStream());
        return prediction;
    }
}