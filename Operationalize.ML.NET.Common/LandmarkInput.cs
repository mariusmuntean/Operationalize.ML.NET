using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace Operationalize.ML.NET.Common;

public class LandmarkInput
{
    public const int ImageWidth = 321;
    public const int ImageHeight = 321;

    public LandmarkInput(Stream imagesStream)
    {
        Image = MLImage.CreateFromStream(imagesStream);
    }

    [ImageType(width: ImageWidth, height: ImageHeight)]
    public MLImage Image { get; }
}