using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Operationalize.ML.NET.Services;

namespace Operationalize.ML.NET.Test;

public class NorthAmericanLandmarkPredictorTests
{
    private CustomWebAppFactory _customWebAppFactory;
    private INorthAmericanLandmarkPredictor _northAmericanLandmarkPredictor;

    [SetUp]
    public void SetUp()
    {
        _customWebAppFactory = new CustomWebAppFactory();
        _northAmericanLandmarkPredictor = _customWebAppFactory.Services.CreateScope().ServiceProvider.GetRequiredService<INorthAmericanLandmarkPredictor>();
    }

    [Test]
    public void PredictStatueOfLiberty()
    {
        // Given that I have a picture of the Statue of Liberty
        using var statueOfLiberty = File.Open("Statue_of_Liberty_7.jpg", FileMode.Open);

        // When I make a prediction on this image
        var predictions = _northAmericanLandmarkPredictor.PredictLandmark(statueOfLiberty);

        // Then the landmark name and probability are as expected
        predictions.Should().NotBeNullOrEmpty();
        predictions.Should().BeInDescendingOrder((prediction) => prediction.Probability, "The predictions should be in descending order");
        predictions.First().LandmarkName.Should().Be("Liberty Island");
        predictions.First().Probability.Should().BeInRange(0.917f, 0.9177f);
    }
}