using Microsoft.ML.Data;

namespace Operationalize.ML.NET.Common;

public class LandmarkOutput
{
    [ColumnName(LandmarkModelSettings.Output)]
    public float[] Prediction { get; set; }
}