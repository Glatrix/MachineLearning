using System;
using System.Data;
using System.Text.Json;

namespace SmallNN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] layers = { 1, 1, 1 };

            DataTable trainingData = new DataTable();

            trainingData.Columns.Add("Input1", typeof(double));
            trainingData.Columns.Add("Target", typeof(double));

            trainingData.Rows.Add(0.0, 0.9);
            trainingData.Rows.Add(0.1, 0.8);
            trainingData.Rows.Add(0.2, 0.7);
            trainingData.Rows.Add(0.3, 0.6);
            trainingData.Rows.Add(0.4, 0.5);
            trainingData.Rows.Add(0.5, 0.4);
            trainingData.Rows.Add(0.6, 0.3);
            trainingData.Rows.Add(0.7, 0.2);
            trainingData.Rows.Add(0.8, 0.1);
            trainingData.Rows.Add(0.9, 0.0);

            NeuralNetwork nn = new NeuralNetwork(layers);

            nn.Train(trainingData, 50000, 0.1);
            nn.Test(trainingData);
        }
    }

    public class NeuralNetwork
    {
        private double[][] _weights;
        private double[][] _biases;
        private int[] _layers;

        public NeuralNetwork(int[] layers)
        {
            _layers = layers;

            _weights = new double[layers.Length - 1][];
            _biases = new double[layers.Length - 1][];
            Setup();
        }

        public void Setup()
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = new double[_layers[i] * _layers[i + 1]];
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    _weights[i][j] = new Random().NextDouble() * 2 - 1;
                }
            }

            for (int i = 0; i < _biases.Length; i++)
            {
                _biases[i] = new double[_layers[i + 1]];
                for (int j = 0; j < _biases[i].Length; j++)
                {
                    _biases[i][j] = new Random().NextDouble() * 2 - 1;
                }
            }
        }

        // Forward pass
        public double[] Forward(double[] input)
        {
            double[] output = input;
            for (int i = 0; i < _weights.Length; i++)
            {
                output = output.BatchedDotProducts(_weights[i]).Add(_biases[i]);
                for (int j = 0; j < output.Length; j++)
                {
                    output[j] = Tanh(output[j]);
                }
            }
            return output;
        }

        public void Train(double[][] inputs, double[][] targets, int epochs, double learningRate)
        {
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    Backpropagate(inputs[j], targets[j], learningRate);
                }
            }
        }

        public void Train(DataTable table, int epochs, double learningRate)
        {
            for (int i = 0; i < epochs; i++)
            {
                foreach (DataRow row in table.Rows)
                {
                    double[] input = new double[_layers[0]];
                    double[] target = new double[_layers[_layers.Length - 1]];

                    for (int j = 0; j < _layers[0]; j++)
                    {
                        input[j] = Convert.ToDouble(row[j]);
                    }

                    for (int j = 0; j < _layers[_layers.Length - 1]; j++)
                    {
                        target[j] = Convert.ToDouble(row[_layers[0] + j]);
                    }

                    Backpropagate(input, target, learningRate);
                }
            }
        }

        public void Backpropagate(double[] input, double[] target, double learningRate)
        {
            // Clean Code for Backpropagation
            double[][] outputs = new double[_layers.Length][];
            double[][] errors = new double[_layers.Length][];
            double[][] deltas = new double[_layers.Length][];

            outputs[0] = input;

            // Forward pass
            for (int i = 1; i < _layers.Length; i++)
            {
                outputs[i] = outputs[i - 1].BatchedDotProducts(_weights[i - 1]).Add(_biases[i - 1]);
                for (int j = 0; j < outputs[i].Length; j++)
                {
                    outputs[i][j] = Tanh(outputs[i][j]);
                }
            }

            // Backward pass
            for (int i = _layers.Length - 1; i > 0; i--)
            {
                // Output layer
                if (i == _layers.Length - 1)
                {
                    errors[i] = target.AddMul(outputs[i], -1);
                    deltas[i] = new double[errors[i].Length];
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        deltas[i][j] = errors[i][j] * dTanh(outputs[i][j]);
                    }
                }
                // Hidden layers
                else
                {
                    errors[i] = deltas[i + 1].BatchedDotProducts(_weights[i]);
                    deltas[i] = new double[errors[i].Length];
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        deltas[i][j] = errors[i][j] * dTanh(outputs[i][j]);
                    }
                }
            }

            // Log Average Error
            double error = 0;
            for (int i = 0; i < errors[errors.Length - 1].Length; i++)
            {
                error += Math.Abs(errors[errors.Length - 1][i]);
            }
            Console.WriteLine($"Error: {error / errors[errors.Length - 1].Length}");

            // Update weights
            for (int i = 0; i < _weights.Length; i++)
            {
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    _weights[i][j] += outputs[i][j % _layers[i]] * deltas[i + 1][j / _layers[i]] * learningRate;
                }
            }

            // Update biases
            for (int i = 0; i < _biases.Length; i++)
            {
                for (int j = 0; j < _biases[i].Length; j++)
                {
                    _biases[i][j] += deltas[i + 1][j] * learningRate;
                }
            }
        }

        public void Test(DataTable table)
        {
            foreach (DataRow row in table.Rows)
            {
                double[] input = new double[_layers[0]];
                double[] target = new double[_layers[_layers.Length - 1]];

                for (int j = 0; j < _layers[0]; j++)
                {
                    input[j] = Convert.ToDouble(row[j]);
                }

                for (int j = 0; j < _layers[_layers.Length - 1]; j++)
                {
                    target[j] = Convert.ToDouble(row[_layers[0] + j]);
                }

                double[] output = Forward(input);
                Console.WriteLine($"Input: {JsonSerializer.Serialize(input)} Target: {JsonSerializer.Serialize(target)} Output: {JsonSerializer.Serialize(output)}");
            }
        }

        //private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        //private double dSigmoid(double x) => x * (1 - x);

        private double Tanh(double x) => Math.Tanh(x);
        private double dTanh(double x) => 1 - x * x;
    }

    public static class MatrixHelpers
    {
        public static double[] BatchedDotProducts(this double[] a, double[] b)
        {
            double[] result = new double[b.Length / a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                for (int j = 0; j < a.Length; j++)
                {
                    result[i] += a[j] * b[i * a.Length + j];
                }
            }
            return result;
        }

        public static double[] Add(this double[] a, double[] b)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        public static double[] AddMul(this double[] a, double[] b, double scalar)
        {
            double[] result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i] * scalar;
            }
            return result;
        }
    }
}
