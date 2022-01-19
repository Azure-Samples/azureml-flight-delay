using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxRuntime
{
    class Program
    {
        static void Main(string[] args)
        {
            UseApi();
        }

        static void UseApi()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "best_model.onnx");
            string inputDataPath = Path.Combine(Directory.GetCurrentDirectory(), "test_data.csv");

            // Optional : Create session options and set the graph optimization level for the session
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;

            using (var session = new InferenceSession(modelPath, options))
            {
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] inputData = LoadTensorFromFile(inputDataPath);
                var index = 0;
                foreach (var name in inputMeta.Keys)
                {
                    var tensor = new DenseTensor<float>(new[] { inputData[index] }, new[] { 1, 1 });
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    index++;
                }

                // Run the inference
                using (var results = session.Run(container))
                {
                    // Show the results
                    var resultList = results.ToList();
                    var parsedResult = bool.Parse(resultList[0].AsTensor<string>()[0]) ? "Delayed" : "On Time";
                    Console.WriteLine($"Prediction: {parsedResult} " +
                        $"(probability: {resultList[1].AsTensor<float>()[0]})");
                }
            }
        }

        static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            // Read data from file
            using (var inputFile = new StreamReader(filename))
            {
                inputFile.ReadLine(); // Skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }
    }
}
