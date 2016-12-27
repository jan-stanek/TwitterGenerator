import com.google.common.base.Charsets;
import com.google.common.io.Files;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


/**
 * Created by Admin on 04.12.2016.
 */
public class LstmTrain {
    public static int LSTM_LAYER_SIZE = 256;        //Number of units in each GravesLSTM layer
    public static int MINI_BATCH_SIZE = 32;		//Size of mini batch to use when  training
    public static int EXAMPLES_LENGTH = 140;		//Length of each training example sequence to use. This could certainly be increased
    public static int T_BP_T_T_LENGTH = 50;        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    public static int EPOCHS_COUNT = 1000;           //Total number of training epochs

    public static int SAMPLES_COUNT = 1;            //Number of samples to generate after each training epoch
    public static int SAMPLES_LENGTH = 140;         //Length of each sample to generate

    private static Map<Character, Integer> charToIndexMap;
    private static Character[] indexToCharArray;
    private static Set<Character> characterSet;

    public static void main(String[] args) {
        try {
            File dataFile = new File("tmp/data.txt");
            File netDir = new File("tmp/");

            String data = Files.toString(dataFile, Charsets.UTF_8);
            createCharacterSet(data);

            CharIterator charIterator = new CharIterator(data, characterSet.size(), charToIndexMap, indexToCharArray);

            Lstm lstm = new Lstm(charIterator);
            lstm.train(netDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void createCharacterSet(String data) {
        characterSet = new HashSet<>();

        for (char c : data.toCharArray()) {
            characterSet.add(c);
        }

        charToIndexMap = new HashMap<>();
        indexToCharArray = new Character[characterSet.size()];

        int i = 0;
        for (Character c : characterSet) {
            charToIndexMap.put(c, i);
            indexToCharArray[i] = c;
            i++;
        }
    }
}
