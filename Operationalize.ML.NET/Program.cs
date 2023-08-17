using System.Reflection;
using Microsoft.Extensions.ML;
using Operationalize.ML.NET.Common;
using Operationalize.ML.NET.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services
    .AddPredictionEnginePool<LandmarkInput, LandmarkOutput>()
    .FromFile(Path.Combine(
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!,
        LandmarkModelSettings.MlNetModelFileName
    ));
builder.Services.AddSingleton<INorthAmericanLabelProvider, NorthAmericanLabelProvider>();
builder.Services.AddScoped<INorthAmericanLandmarkPredictor, NorthAmericanLandmarkPredictor>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();

// Just a hack to make this class accessible from the Test project
public partial class Program
{
}