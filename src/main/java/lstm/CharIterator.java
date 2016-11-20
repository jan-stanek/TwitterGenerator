package lstm;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

public class CharIterator implements DataSetIterator {

    private int exampleLength;
    private int miniBatchSize;

    private Random rng;

    private Map<Character, Integer> charToIdMap;
    private ArrayList<Character> allChars;
    private LinkedList<Integer> examplesStarts;


    private char [] characters;


    public CharIterator(List<String> texts, int miniBatchSize, int exampleLength, Random rng) throws IOException {
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        int totalLength = texts.size();
        for (String text : texts) {
            totalLength += text.length();
        }

        characters = new char[totalLength];
        int index = 0;

        for (String s : texts) {
            char[] text = s.toCharArray();
            for (char character : text) {
                characters[index++] = character;
            }
            characters[index++] = '\n';
        }

        Set<Character> allCharsSet = new HashSet<>();
        allChars = new ArrayList<>();
        for (char c : characters) {
            allCharsSet.add(c);
        }
        for (char c : allCharsSet) {
            allChars.add(c);
        }

        charToIdMap = new HashMap<>();
        for (int i = 0; i < allChars.size(); i++)
            charToIdMap.put(allChars.get(i), i);

        initializeOffsets();
    }

    private void initializeOffsets() {
        examplesStarts = new LinkedList<>();

        int examplesCount = (characters.length - 1) / exampleLength - 2;

        for (int i = 0; i < examplesCount; i++) {
            examplesStarts.add(i * exampleLength);
        }

        Collections.shuffle(examplesStarts, rng);
    }

    public boolean hasNext() {
        return examplesStarts.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if (examplesStarts.size() == 0)
            throw new NoSuchElementException();

        int currentMinibatchSize = Math.min(num, examplesStarts.size());
        INDArray input = Nd4j.create(new int[]{currentMinibatchSize, allChars.size(), exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currentMinibatchSize, allChars.size(), exampleLength}, 'f');

        for (int i = 0; i < currentMinibatchSize; i++) {
            int start = examplesStarts.removeFirst();
            int end = start + exampleLength;
            int currentChar = charToIdMap.get(characters[start]);

            int c = 0;
            for (int j = start + 1; j < end; j++, c++) {
                int nextChar = charToIdMap.get(characters[j]);
                input.putScalar(new int[]{i, currentChar, c}, 1.0);
                labels.putScalar(new int[]{i, nextChar, c}, 1.0);
                currentChar = nextChar;
            }
        }

        return new DataSet(input, labels);
    }

    public char convertIndexToCharacter( int idx ){
        return allChars.get(idx);
    }

    public int convertCharacterToIndex( char c ){
        return charToIdMap.get(c);
    }

    public char getRandomCharacter(){
        return allChars.get((int) (rng.nextDouble()*allChars.size()));
    }

    public int totalExamples() {
        return (characters.length - 1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return allChars.size();
    }

    public int totalOutcomes() {
        return allChars.size();
    }

    public void reset() {
        examplesStarts.clear();
        initializeOffsets();
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - examplesStarts.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public void remove() {
        throw new UnsupportedOperationException("Not implemented");
    }
}