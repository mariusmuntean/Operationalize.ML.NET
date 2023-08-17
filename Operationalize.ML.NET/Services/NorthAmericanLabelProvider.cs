using System.Reflection;
using Operationalize.ML.NET.Common;

namespace Operationalize.ML.NET.Services;

internal class NorthAmericanLabelProvider : INorthAmericanLabelProvider
{
    private Lazy<string[]>? _lazyLabels;

    public string[] GetLabels()
    {
        _lazyLabels ??= new Lazy<string[]>(() =>
        {
            var labelFilePath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!,
                LandmarkModelSettings.LabelFileName);
            var labelLines = File.ReadAllLines(labelFilePath);
            return labelLines.Skip(1)
                .Select(line => line.Split(","))
                .Select(lineTokens => (Index: int.Parse(lineTokens[0]), LandmarkName: lineTokens[1]))
                .OrderBy(tuple => tuple.Index)
                .Select(tuple => tuple.LandmarkName)
                .ToArray();
        });

        return _lazyLabels.Value;
    }
}