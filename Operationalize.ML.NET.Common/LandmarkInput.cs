using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Operationalize.ML.NET.Common;

public class LandmarkInput
{
    public LandmarkInput(Stream imagesStream)
    {
        Image = MLImage.CreateFromStream(imagesStream);
    }

    [ImageType(width: LandmarkImageSettings.ImageWidth, height: LandmarkImageSettings.ImageHeight)]
    public MLImage Image { get; }
}