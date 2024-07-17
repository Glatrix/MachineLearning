using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class NeuralNetwork
    {
        private int[] layers;
        private float[][] neurons;
        private float[][] biases;
        private float[][][] weights;

        private const float LeakyAlpha = 0.01f;

        private Random random = new Random();

        private Func<float, float> ActivationFunc;
        private Func<float, float> ActivationFuncDer;

        public bool Logging { get; set; } = false;

        public NeuralNetwork(int[] layerCounts, ActivationFunction activationFunc)
        {
            layers = layerCounts;
            InitializeNeurons();
            InitializeBiases();
            InitializeWeights();

            switch (activationFunc)
            {
                case ActivationFunction.ReLU:
                    ActivationFunc = ReLU;
                    ActivationFuncDer = ReLUDerivative;
                    break;
                case ActivationFunction.TanH:
                    ActivationFunc = TanH;
                    ActivationFuncDer = TanHDerivative;
                    break;
                case ActivationFunction.LeakyReLU:
                    ActivationFunc = LeakyReLU;
                    ActivationFuncDer = LeakyReLUDerivative;
                    break;
                case ActivationFunction.Sigmoid:
                default:
                    ActivationFunc = Sigmoid;
                    ActivationFuncDer = SigmoidDerivative;
                    break;
            }
        }

        private void Log(string s)
        {
            if (Logging)
            {
                Console.WriteLine(s);
            }
        }

        private void InitializeNeurons()
        {
            neurons = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                neurons[i] = new float[layers[i]];
            }
        }

        private void InitializeBiases()
        {
            biases = new float[layers.Length][];
            for (int i = 1; i < layers.Length; i++)
            {
                biases[i] = new float[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    biases[i][j] = (float)random.NextDouble() - 0.5f;
                }
            }
        }

        private void InitializeWeights()
        {
            weights = new float[layers.Length][][];
            for (int i = 1; i < layers.Length; i++)
            {
                weights[i] = new float[layers[i]][];
                for (int j = 0; j < layers[i]; j++)
                {
                    weights[i][j] = new float[layers[i - 1]];
                    for (int k = 0; k < layers[i - 1]; k++)
                    {
                        weights[i][j][k] = (float)random.NextDouble() - 0.5f;
                    }
                }
            }
        }

        public float[] Forward(float[] inputs)
        {
            if (inputs.Length != neurons[0].Length)
            {
                throw new ArgumentException($"Input length {inputs.Length} does not match the expected input layer size {neurons[0].Length}.");
            }

            neurons[0] = inputs;
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i]; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < layers[i - 1]; k++)
                    {
                        sum += neurons[i - 1][k] * weights[i][j][k];
                    }
                    neurons[i][j] = ActivationFunc(sum + biases[i][j]);
                }
            }
            return neurons[neurons.Length - 1];
        }

        public void Backpropagate(float[] expected, float learningRate)
        {
            float[][] gamma = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                gamma[i] = new float[layers[i]];
            }

            // Output layer gamma
            for (int i = 0; i < layers[layers.Length - 1]; i++)
            {
                gamma[gamma.Length - 1][i] = (neurons[neurons.Length - 1][i] - expected[i]) * ActivationFuncDer(neurons[neurons.Length - 1][i]);
            }

            // Hidden layer gamma
            for (int i = layers.Length - 2; i > 0; i--)
            {
                for (int j = 0; j < layers[i]; j++)
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < layers[i + 1]; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * weights[i + 1][k][j];
                    }
                    gamma[i][j] *= ActivationFuncDer(neurons[i][j]);
                }
            }

            // Update weights and biases
            for (int i = 1; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i]; j++)
                {
                    biases[i][j] -= gamma[i][j] * learningRate;
                    for (int k = 0; k < layers[i - 1]; k++)
                    {
                        weights[i][j][k] -= gamma[i][j] * neurons[i - 1][k] * learningRate;
                    }
                }
            }
        }

        public void Train(float[][] inputs, float[][] expectedOutputs, int epochs, float learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                int iters = inputs.GetLength(0) - 1;
                for (int i = 0; i < iters; i++)
                {
                    // Forward pass
                    float[] output = Forward(inputs[i]);

                    // Backpropagation
                    Backpropagate(expectedOutputs[i], learningRate);
                }
                Log($"Epoch {epoch + 1}/{epochs} completed.");
            }
        }

        private float Sigmoid(float value)
        {
            // Sigmoid Activation Function
            return 1 / (1 + (float)Math.Exp(-value));
        }

        private float SigmoidDerivative(float value)
        {
            // Derivative of Sigmoid Activation Function
            return value * (1 - value);
        }

        // Optional: Implementing ReLU activation function
        private float ReLU(float value)
        {
            return Math.Max(0, value);
        }

        private float ReLUDerivative(float value)
        {
            return value > 0 ? 1 : 0;
        }

        // Optional: Implementing TanH activation function
        private float TanH(float value)
        {
            return (float)Math.Tanh(value);
        }

        private float TanHDerivative(float value)
        {
            return 1 - value * value;
        }

        // Optional: Implementing Leaky ReLU activation function
        private float LeakyReLU(float value)
        {
            return value > 0 ? value : LeakyAlpha * value;
        }

        private float LeakyReLUDerivative(float value)
        {
            return value > 0 ? 1 : LeakyAlpha;
        }
    }

    public enum ActivationFunction
    {
        Sigmoid,
        ReLU,
        TanH,
        LeakyReLU
    }
}
