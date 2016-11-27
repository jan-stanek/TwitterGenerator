package lstm;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Lstm {
    private int lstmLayerSize = 200;                    //Number of units in each GravesLSTM layer
    private int miniBatchSize = 1;						//Size of mini batch to use when  training
    private int exampleLength = 64;					//Length of each training example sequence to use. This could certainly be increased
    private int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    private int numEpochs = 50;                          //Total number of training epochs
    private int nSamplesToGenerate = 4;                 //Number of samples to generate after each training epoch
    private int nCharactersToSample = 300;              //Length of each sample to generate


    private Set<Character> validChars;

    private CharIterator charIterator;
    private ComputationGraph net;
    private Random rng;


    public Lstm(Set<Character> validChars) {
        this.validChars = validChars;
        int nIn = validChars.size();
        int nOut = validChars.size();

        rng = new Random(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("first", new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation("tanh")
                        .build(),"input")
                .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation("tanh")
                        .build(),"first")
                .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .updater(Updater.RMSPROP)
                        .nIn(2*lstmLayerSize)
                        .nOut(nOut)
                        .build(),"first","second")
                .setOutputs("outputLayer")
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false)
                .backprop(true)
                .build();

        net = new ComputationGraph(conf);

        net.init();
        net.setListeners(new ScoreIterationListener(1));
    }

    public void train(List<String> texts) throws IOException {
        charIterator = new CharIterator(texts, validChars, miniBatchSize, exampleLength);

        for (int i = 0; i < numEpochs; i++) {
            while (charIterator.hasNext()) {
                net.fit(charIterator.next());
            }

            String[] samples = sampleCharactersFromNetwork(nCharactersToSample, nSamplesToGenerate);
            System.out.println("Epoch " + (i+1) + ":");
            for (String sample : samples) {
                System.out.println(sample);
            }
            System.out.println();

            charIterator.reset();
        }
    }

    public String generate() {
        String[] samples = sampleCharactersFromNetwork(nCharactersToSample, nSamplesToGenerate);

        return samples[0];
    }

    private String[] sampleCharactersFromNetwork(int charactersToSample, int numSamples) {
        String initialization = String.valueOf(charIterator.getRandomCharacter());

        INDArray initializationInput = Nd4j.zeros(numSamples, charIterator.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = charIterator.convertCharacterToIndex(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++)
            sb[i] = new StringBuilder(initialization);

        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput)[0];
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            INDArray nextInput = Nd4j.zeros(numSamples, charIterator.inputColumns());

            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[charIterator.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(charIterator.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput)[0];    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    public int sampleFromDistribution(double[] distribution) {
        double d = rng.nextDouble();
        double sum = 0.0;

        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) return i;
        }

        throw new IllegalArgumentException("Distribution is invalid.");
    }
}