package lstm;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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
    private int lstmLayerSize = 128;                    //Number of units in each GravesLSTM layer
    private int miniBatchSize = 32;						//Size of mini batch to use when  training
    private int exampleLength = 2000;					//Length of each training example sequence to use. This could certainly be increased
    private int tbpttLength = 100;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    private int numEpochs = 100;                          //Total number of training epochs

    private int nSamplesToGenerate = 1;                 //Number of samples to generate after each training epoch
    private int nCharactersToSample = 140;              //Length of each sample to generate


    private Set<Character> validChars;

    private CharIterator charIterator;
    private MultiLayerNetwork net;
    private Random rng;


    public Lstm(Set<Character> validChars) {
        this.validChars = validChars;
        int nIn = validChars.size();
        int nOut = validChars.size();

        rng = new Random(12345);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn).nOut(lstmLayerSize)
                        .activation("tanh")
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation("tanh")
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                        .nIn(lstmLayerSize).nOut(nOut)
                        .activation("softmax")
                        .build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();

        net = new MultiLayerNetwork(conf);

        net.init();
        net.setListeners(new StatsListener(statsStorage));
    }

    public void train(List<String> texts) throws IOException {
        charIterator = new CharIterator(texts, validChars, miniBatchSize, exampleLength);

        for (int i = 0; i < numEpochs; i++) {
            while (charIterator.hasNext()) {
                net.fit(charIterator.next());
            }

            System.out.println("Epoch " + (i+1) + ":");
            for (int j = 0; j < 10; j++) {
                String sample[] = sampleCharactersFromNetwork(nCharactersToSample, nSamplesToGenerate);
                System.out.println(sample[0]);
                System.out.println();
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
                initializationInput.putScalar(new int[]{j, idx, i}, 1f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++)
            sb[i] = new StringBuilder(initialization);

        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);

        for (int i = 0; i < charactersToSample; i++) {
            INDArray nextInput = Nd4j.zeros(numSamples, charIterator.inputColumns());

            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[charIterator.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1f);
                sb[s].append(charIterator.convertIndexToCharacter(sampledCharacterIdx));
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
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